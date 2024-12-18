import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import os
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import f1_score, make_scorer, confusion_matrix, accuracy_score
from xgboost import XGBClassifier
import joblib
from scipy import stats
from sklearn.metrics import roc_curve, auc
from sklearn.utils import resample
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.calibration import calibration_curve
from sklearn.utils import resample
from tqdm import tqdm

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

        #save final dataset
        os.makedirs('data',exist_ok=True)
        self.final_data.to_csv('data/dataset_for_ml.csv',index=False)

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

    def ml_analysis_no_feature_alteration(self):
        y = self.final_data['label']
        if not os.path.exists('xgboost_model.joblib'):
            X = self.final_data.drop(columns=['label']) 
            y = self.final_data['label']           

            # Standardize and normalize the data
            scaler = StandardScaler()
            x_scaled = scaler.fit_transform(X)

            min_max_scaler = MinMaxScaler(feature_range=(0, 1))
            x_normalized = min_max_scaler.fit_transform(x_scaled)
        
            #split the data 90/10
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(x_normalized , y, test_size=0.10, stratify=y, random_state=42)

            # normalizer = MinMaxScaler()  # Normalize to [0, 1] range
            # X_train_normalized = normalizer.fit_transform(X_train_scaled)
            # X_test_normalized = normalizer.transform(X_test_scaled)

            # Define the XGBoost classifier
            # xgb = XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False)

            # # Set up parameter grid for hyperparameter tuning
            # param_grid = {
            #     'n_estimators': [50, 100, 200],
            #     'max_depth': [3, 5, 7],
            #     'learning_rate': [0.01, 0.1, 0.2],
            #     'subsample': [0.6, 0.8, 1.0],
            #     'colsample_bytree': [0.6, 0.8, 1.0]
            # }

            # # Define the scoring metric
            # f1_scorer = make_scorer(f1_score)

            # grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, scoring=f1_scorer, cv=5, verbose=1, n_jobs=1)
            # grid_search.fit(X_train_scaled, y_train)

            # self.best_model = grid_search.best_estimator_
            # self.best_model.fit(X_train_scaled, y_train)

            # y_pred = self.best_model.predict(X_test_scaled)
            # f1 = f1_score(y_test, y_pred)

            # print(f"Best parameters: {grid_search.best_params_}")
            # print(f"F1 Score on test set: {f1:.4f}")
            # print(f"ACC Score on test set: {accuracy_score(y_test, y_pred)}")

            #USE CROSS VALIDATION ONLY, AS THE DATASET IS VERY SMALL
            xgb = XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False)

            # Set up parameter grid for hyperparameter tuning
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0]
            }

            # Use stratified k-fold cross-validation
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            f1_scores = []
            acc_scores = []
            self.best_model_overall = None
            #track the best F1 score across folds
            best_f1_score = 0  

            for train_index, val_index in tqdm(skf.split(self.X_train, self.y_train)):
                X_train, X_val = self.X_train[train_index, :], self.X_train[val_index, :]
                y_train, y_val = self.y_train.values[train_index], self.y_train.values[val_index]

                # X_train, X_val = self.X_train[train_index], self.X_train[val_index]
                # y_train, y_val = self.y_train[train_index], self.y_train[val_index]

                # Hyperparameter tuning with GridSearchCV
                grid_search = GridSearchCV(
                    estimator=xgb,
                    param_grid=param_grid,
                    scoring='f1',
                    cv=3,  #inner 3-fold cross-validation for hyperparameter tuning
                    verbose=3,
                    n_jobs=8
                )
                grid_search.fit(X_train, y_train)

                # Train the best model on the current fold
                best_model = grid_search.best_estimator_
                best_model.fit(X_train, y_train)

                # Predict on the validation set
                y_pred = best_model.predict(X_val)

                # Compute metrics
                f1 = f1_score(y_val, y_pred)
                acc = accuracy_score(y_val, y_pred)
                f1_scores.append(f1)
                acc_scores.append(acc)

                if f1 > best_f1_score:
                    best_f1_score = f1
                    self.best_model_overall = best_model  # Save the best model

            # Report average and standard deviation of the scores
            print(f"Average F1 Score: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
            print(f"Average Accuracy: {np.mean(acc_scores):.4f} ± {np.std(acc_scores):.4f}")

            joblib.dump(self.best_model_overall, 'xgboost_model.joblib')
            # with open('model_results.txt', 'w') as f:
            #     f.write(f"Best parameters: {grid_search.best_params_}\n")
            #     f.write(f"F1 Score on test set: {f1:.4f}\n")

            with open('model_results.txt', 'w') as f:
                f.write(f"Average F1 Score: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}\n")
                f.write(f"Average Accuracy: {np.mean(acc_scores):.4f} ± {np.std(acc_scores):.4f}\n")
                f.write(f"Best parameters: {grid_search.best_params_}\n")
                f.write(f"Best F1 Score: {best_f1_score:.4f}\n")

            y_pred = self.best_model_overall.predict(self.X_test)
            
            cm = confusion_matrix(self.y_test, y_pred)

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

            self.roc_curve_thresholds(self.y_test,self.X_test)
            self.calibration_analysis(y_test=self.y_test,y_probs=self.best_model_overall.predict_proba(self.X_test)[:, 1])
            self.precision_recall_analysis(y_test=self.y_test, y_probs=self.best_model_overall.predict_proba(self.X_test)[:, 1])

            mean_f1, lower_f1, upper_f1 = self.compute_confidence_intervals(self.y_test, y_pred, f1_score)
            #confidence intervals for Accuracy
            mean_acc, lower_acc, upper_acc = self.compute_confidence_intervals(self.y_test, y_pred, accuracy_score)
            print(f"F1 Score CI: Mean={mean_f1:.4f}, Lower={lower_f1:.4f}, Upper={upper_f1:.4f}")
            print(f"Accuracy CI: Mean={mean_acc:.4f}, Lower={lower_acc:.4f}, Upper={upper_acc:.4f}")

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
            self.best_model_overall = joblib.load('xgboost_model.joblib')
    
    def roc_curve_thresholds(self,y_test,X_test_scaled):
        y_probs = self.best_model_overall.predict_proba(X_test_scaled)[:, 1]  # Probability of class 1 (concussion)

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
        importances = self.best_model_overall.feature_importances_

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
        
    def precision_recall_analysis(self, y_test, y_probs):
        precision, recall, _ = precision_recall_curve(y_test, y_probs)
        avg_precision = average_precision_score(y_test, y_probs)

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='purple', lw=2, label =f'Avg Precision = {avg_precision:.2f}')
        plt.title('Precision-Recall Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend()
        plt.savefig(os.path.join('figures', 'precision_recall_curve.png'), dpi=400)
        plt.close()

    def calibration_analysis(self, y_test, y_probs):
        prob_true, prob_pred = calibration_curve(y_test, y_probs, n_bins=10)

        plt.figure(figsize=(8, 6))
        plt.plot(prob_pred, prob_true, marker='o', label='Calibration Curve')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
        plt.title('Calibration Curve')
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.legend()
        plt.savefig(os.path.join('figures', 'calibration_curve.png'), dpi=400)
        plt.close()

    def compute_confidence_intervals(self, y_true, y_pred, metric_function, n_bootstrap=1000):
        metric_values = []
        for _ in range(n_bootstrap):
            y_true_boot, y_pred_boot = resample(y_true, y_pred)
            metric_values.append(metric_function(y_true_boot, y_pred_boot))
        
        lower = np.percentile(metric_values, 2.5)
        upper = np.percentile(metric_values, 97.5)
        mean = np.mean(metric_values)
        return mean, lower, upper

    def run_analysis(self):
        self.visualize_data()
        self.ml_analysis_no_feature_alteration()

def main():
    mlVoms().run_analysis()

if __name__ == "__main__":
    main()