import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Simulated data for demonstration purposes
np.random.seed(42)
n_features = 18
n_samples = 558

# Generate mock data: SRC (label 1) and Control (label 0)
X = np.random.rand(n_samples, n_features)
y = np.array([1] * 279 + [0] * 279)

# Step 1: Visualize Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

plt.figure(figsize=(10, 5))
plt.bar(['Training Data', 'Test Data'], [len(y_train), len(y_test)], color=['blue', 'orange'])
plt.title('Train-Test Split (70/30)')
plt.ylabel('Number of Samples')
plt.savefig('train_test_split.png', dpi=300)
plt.show()

# Step 2: Standardization Visualization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

plt.figure(figsize=(12, 6))
sns.histplot(X_train[:, 0], kde=True, color='red', label='Before Scaling', alpha=0.6)
sns.histplot(X_train_scaled[:, 0], kde=True, color='blue', label='After Scaling', alpha=0.6)
plt.title('Feature Standardization (Example: Feature 1)')
plt.legend()
plt.savefig('feature_standardization.png', dpi=300)
plt.show()

# Step 3: Feature Importance Visualization (Mocked)
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train_scaled, y_train)
feature_importances = model.feature_importances_

plt.figure(figsize=(12, 6))
sns.barplot(x=feature_importances, y=[f'Feature {i+1}' for i in range(n_features)], palette='viridis')
plt.title('Feature Importance')
plt.xlabel('Normalized Importance')
plt.savefig('feature_importance.png', dpi=300)
plt.show()

# Step 4: Model Evaluation Metrics - Confusion Matrix Visualization
y_pred = model.predict(scaler.transform(X_test))
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Control (0)', 'SRC (1)'], yticklabels=['Control (0)', 'SRC (1)'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('confusion_matrix.png', dpi=300)
plt.show()

# Step 5: ROC Curve Visualization
y_probs = model.predict_proba(scaler.transform(X_test))[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
plt.scatter(1 - 0.8, 0.8, color='red', label='Custom Threshold Point (Sensitivity=0.8, Specificity=0.8)')
plt.title('ROC Curve with Custom Thresholds')
plt.xlabel('1 - Specificity (False Positive Rate)')
plt.ylabel('Sensitivity (True Positive Rate)')
plt.legend()
plt.savefig('roc_curve.png', dpi=300)
plt.show()
