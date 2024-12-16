import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import f1_score, make_scorer, confusion_matrix, accuracy_score
from xgboost import XGBClassifier
import joblib
from scipy import stats
from sklearn.metrics import roc_curve, auc
from sklearn.utils import resample

class mlVoms():
    def __init__(self):
        print('Initialize class')
        if not os.path.exists('figures'):
            os.mkdir('figures')
        #merge concussion
        df_src_sacc = pd.read_csv('Saccades_SRC.csv')
        df_src_sp = pd.read_csv('SP_SRC.csv')
        df_src_VOR = pd.read_csv('VOR_SRC.csv')
        df_src_VMS = pd.read_csv('VMS_SRC.csv')
        combined_df_src = pd.merge(df_src_sacc, df_src_sp, on='ID', how='inner')
        combined_df_src = pd.merge(combined_df_src, df_src_VOR, on='ID', how='inner')
        combined_df_src = pd.merge(combined_df_src, df_src_VMS, on='ID', how='inner')

        #merge control
        df_con_sacc = pd.read_csv('Saccades_Control.csv')
        df_con_sp = pd.read_csv('SP_Control.csv')
        df_con_VOR = pd.read_csv('VOR_Control.csv')
        df_con_VMS = pd.read_csv('VMS_Control.csv')
        combined_df_healthy = pd.merge(df_con_sacc, df_con_sp, on='ID', how='inner')
        combined_df_healthy = pd.merge(combined_df_healthy, df_con_VOR, on='ID', how='inner')
        combined_df_healthy = pd.merge(combined_df_healthy, df_con_VMS, on='ID', how='inner')

        #remove all string values
        # combined_df_healthy['ID'].to_csv('ID_healthy.csv',index=False)
        # combined_df_src['ID'].to_csv('ID_src.csv',index=False)
        combined_df_src = combined_df_src.select_dtypes(include=[np.number])
        combined_df_healthy = combined_df_healthy.select_dtypes(include=[np.number])

        #add classifications
        combined_df_src['label'] = 1
        combined_df_healthy['label'] = 0

        #final concat
        self.final_data = pd.concat([combined_df_src,combined_df_healthy])
        print(self.final_data.shape)
        self.final_data = self.final_data.replace([float('inf'), -float('inf')], pd.NA)
        self.final_data = self.final_data.dropna() 
        print(self.final_data.shape)

        #remove outliers with z-score
        numeric_data = self.final_data.drop(columns=['label'], errors='ignore').select_dtypes(include=[np.number])
        numeric_data.replace([np.inf, -np.inf], np.nan, inplace=True)
        numeric_data.dropna(inplace=True)

        z_scores = stats.zscore(numeric_data)
        threshold = 3
        self.final_data = self.final_data[(np.abs(z_scores) < threshold).all(axis=1)]
        print(self.final_data.shape)
        print(self.final_data)

        #Balance the dataset
        class_0 = self.final_data[self.final_data['label'] == 0]
        class_1 = self.final_data[self.final_data['label'] == 1]

        # Determine which class is larger
        if len(class_0) > len(class_1):
            class_0_undersampled = resample(class_0, 
                                            replace=False,  # Sample without replacement
                                            n_samples=len(class_1),  # Match the size of class 1
                                            random_state=42)  # For reproducibility
            self.final_data = pd.concat([class_0_undersampled, class_1])
        else:
            class_1_undersampled = resample(class_1, 
                                            replace=False,  # Sample without replacement
                                            n_samples=len(class_0),  # Match the size of class 0
                                            random_state=42)  # For reproducibility
            self.final_data = pd.concat([class_0, class_1_undersampled])

        self.final_data = self.final_data.sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"Balanced dataset shape: {self.final_data.shape}")

    def visualize_data(self):
        numeric_columns = self.final_data.select_dtypes(include=['number']).columns.difference(['label'])
        num_columns = len(numeric_columns)
        
        n_cols = 3
        n_rows = math.ceil(num_columns / n_cols)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 5))
        axes = axes.flatten()

        for idx, column in enumerate(numeric_columns):
            ax = axes[idx]
            
            sns.histplot(self.final_data[self.final_data['label'] == 0][column], color='blue', label='Label 0', kde=True, stat="density", ax=ax, fill=True, alpha=0.4)
            sns.histplot(self.final_data[self.final_data['label'] == 1][column], color='red', label='Label 1', kde=True, stat="density", ax=ax, fill=True, alpha=0.4)
            
            ax.set_title(f'Histogram of {column} by Label')
            ax.set_xlabel(column)
            ax.set_ylabel('Density')
            ax.legend()
        
        for j in range(idx + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        
        plt.savefig(os.path.join('figures','dist_features.png'),dpi=400)
        plt.close()

    def ml_analysis(self):
        y = self.final_data['label']
        if not os.path.exists('xgboost_model.joblib'):
            X = self.final_data.drop(columns=['label']) 
            y = self.final_data['label']           

            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

            # Standardize and normalize the data
            scaler = StandardScaler()  # Standardize
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # normalizer = MinMaxScaler()  # Normalize to [0, 1] range
            # X_train_normalized = normalizer.fit_transform(X_train_scaled)
            # X_test_normalized = normalizer.transform(X_test_scaled)

            # Define the XGBoost classifier
            xgb = XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False)

            # Set up parameter grid for hyperparameter tuning
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0]
            }

            # Define the scoring metric
            f1_scorer = make_scorer(f1_score)

            grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, scoring=f1_scorer, cv=5, verbose=1, n_jobs=1)
            grid_search.fit(X_train_scaled, y_train)

            self.best_model = grid_search.best_estimator_
            self.best_model.fit(X_train_scaled, y_train)

            y_pred = self.best_model.predict(X_test_scaled)
            f1 = f1_score(y_test, y_pred)

            print(f"Best parameters: {grid_search.best_params_}")
            print(f"F1 Score on test set: {f1:.4f}")
            print(f"ACC Score on test set: {accuracy_score(y_test, y_pred)}")
            joblib.dump(self.best_model, 'xgboost_model.joblib')
            with open('model_results.txt', 'w') as f:
                f.write(f"Best parameters: {grid_search.best_params_}\n")
                f.write(f"F1 Score on test set: {f1:.4f}\n")

            cm = confusion_matrix(y_test, y_pred)

            # cm[0,0] = True Negative, cm[0,1] = False Positive
            # cm[1,0] = False Negative, cm[1,1] = True Positive
            tn, fp, fn, tp = cm.ravel()

            sensitivity = tp / (tp + fn)
            specificity = tn / (tn + fp)

            print(f"Sensitivity: {sensitivity:.4f}")
            print(f"Specificity: {specificity:.4f}")

            labels = ['Sensitivity', 'Specificity']
            values = [sensitivity, specificity]

            plt.figure(figsize=(8, 5))
            plt.bar(labels, values, color=['blue', 'green'])
            plt.title('Sensitivity and Specificity')
            plt.ylabel('Score')
            plt.ylim(0, 1)

            plt.savefig(os.path.join('figures','SN_SP.png'),dpi=400)
            plt.close()

            self.roc_curve_thresholds(y_test,X_test_scaled)
            # fpr, tpr, thresholds = roc_curve(y_test, y_pred)
            # roc_auc = auc(fpr, tpr)

            # # Plot ROC curve
            # plt.figure(figsize=(10, 6))
            # plt.plot(fpr, tpr, color='b', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')

            # # Plot the diagonal line (random classifier)
            # plt.plot([0, 1], [0, 1], color='gray', linestyle='--')

            # plt.title('Receiver Operating Characteristic (ROC) Curve')
            # plt.xlabel('False Positive Rate (1 - Specificity)')
            # plt.ylabel('True Positive Rate (Sensitivity)')
            # plt.legend(loc='lower right')

            # plt.savefig(os.path.join('figures','SN_SP_curve.png'),dpi=400)
            # plt.close()
            self.feature_importance()
            
        else:
            self.best_model = joblib.load('xgboost_model.joblib')
    
    def roc_curve_thresholds(self,y_test,X_test_scaled):
        y_probs = self.best_model.predict_proba(X_test_scaled)[:, 1]  # Probability of class 1 (concussion)

        # ROC Curve and AUC
        fpr, tpr, thresholds = roc_curve(y_test, y_probs)
        roc_auc = auc(fpr, tpr)

        # Plotting the ROC curve
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')

        # Plotting the sensitivity and specificity at various thresholds
        sensitivity = []
        specificity = []
        threshold_vals = np.arange(0, 1.1, 0.1)  # Custom thresholds from 0 to 1 in steps of 0.1

        for threshold in threshold_vals:
            y_pred = (y_probs >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            sens = tp / (tp + fn)  # Sensitivity = TP / (TP + FN)
            spec = tn / (tn + fp)  # Specificity = TN / (TN + FP)
            sensitivity.append(sens)
            specificity.append(spec)
            plt.scatter(1-spec, sens, marker='o', color='red', label=f'Threshold = {threshold:.1f}' if threshold == 0 else "")

        plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')
        plt.xlabel('1 - Specificity')
        plt.ylabel('Sensitivity')
        plt.title('ROC Curve with Custom Threshold Points')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join('figures','SN_SP_curve_raw.png'),dpi=400)
        plt.close()

        # Plot Sensitivity and Specificity at different thresholds
        plt.figure(figsize=(10, 6))
        plt.plot(threshold_vals, sensitivity, label="Sensitivity", marker='o')
        plt.plot(threshold_vals, specificity, label="Specificity", marker='x')
        plt.xlabel("Threshold")
        plt.ylabel("Value")
        plt.title("Sensitivity and Specificity at Different Thresholds")
        plt.legend()
        plt.savefig(os.path.join('figures','SN_SP_curve_all.png'),dpi=400)
        plt.close()
    def feature_importance(self):
        importances = self.best_model.feature_importances_

        feature_columns = [col for col in self.final_data.columns if col != 'label']

        importance_df = pd.DataFrame({
            'Feature': feature_columns,
            'Importance': importances
        })

        importance_df['Importance'] = importance_df['Importance'] / importance_df['Importance'].sum()

        importance_df = importance_df.sort_values(by='Importance', ascending=False)

        plt.figure(figsize=(12, 6))
        sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')

        plt.title('Feature Importance', fontsize=16)
        plt.xlabel('Normalized Importance', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.tight_layout()
        
        plt.savefig(os.path.join('figures', 'Feature_Importances.png'), dpi=400)
        plt.close()
        
    def run_analysis(self):
        self.visualize_data()
        self.ml_analysis()

def main():
    mlVoms().run_analysis()

if __name__ == "__main__":
    main()