import os  # Provides functionalities to interact with the operating system
import joblib  # Used for saving and loading machine learning models
import numpy as np  # Supports large, multi-dimensional arrays and mathematical functions
import seaborn as sns  # Used for data visualization, especially statistical graphics
import matplotlib.pyplot as plt  # Provides functions for plotting and visualization
import pandas as pd  # Used for data manipulation and analysis
from sklearn.datasets import load_breast_cancer  # Loads the breast cancer dataset
from sklearn.model_selection import train_test_split, GridSearchCV  # Splits data and performs hyperparameter tuning
from sklearn.ensemble import VotingClassifier, AdaBoostClassifier  # Implements ensemble learning models
from sklearn.svm import SVC  # Support Vector Classifier for classification tasks
from sklearn.linear_model import LogisticRegression  # Logistic Regression model
from sklearn.preprocessing import StandardScaler, MinMaxScaler  # Standardizes features by removing the mean and scaling to unit variance
from sklearn.decomposition import PCA  # Performs Principal Component Analysis (PCA) for dimensionality reduction
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix, roc_curve, auc, mean_absolute_error, mean_squared_error, f1_score  # Various metrics for model evaluation
from sklearn.impute import SimpleImputer  # To handle missing values if needed


# Step 1: Load and preprocess the data
data = load_breast_cancer()  # Load the breast cancer dataset
X = data.data  # Extract feature variables
y = data.target  # Extract target variable

# Convert to DataFrame to display the dataset
df = pd.DataFrame(X, columns=data.feature_names)  # Create DataFrame from features
df['target'] = y  # Add target column to DataFrame

# Display the first few rows of the dataset
print("Loaded Breast Cancer Dataset:")
print(df.head())  # Print the first few rows of the dataset

# Step 1.1: Preprocessing - Handling missing values (if any)
imputer = SimpleImputer(strategy='mean')  # Fill missing values with the mean (if any)
X = imputer.fit_transform(X)  # Apply imputer to the features

# Step 1.2: Feature Scaling - StandardScaler (already done later but doing it immediately here for preprocessing)
scaler = StandardScaler()  # Initialize Standard Scaler
X = scaler.fit_transform(X)  # Fit and transform all the features at once

# Step 2: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # 80-20 train-test split

df_train = pd.DataFrame(X_train, columns=data.feature_names)  # Convert training data into DataFrame for visualization
df_train['target'] = y_train  # Add target column to the DataFrame

# Step 3: Compute and visualize the correlation matrix before PCA
plt.figure(figsize=(12, 10))  # Set figure size
sns.heatmap(df_train.corr(), annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)  # Plot correlation heatmap
plt.title('Feature Correlation Matrix')  # Add title
plt.show()  # Display plot

# Step 4: Pair Plot for feature relationships (limited to a subset for better visualization)
sns.pairplot(df_train.iloc[:, :5].join(df_train[['target']]), hue='target', palette='coolwarm')  # Plot pairwise relationships
plt.show()  # Display plot

# Step 5: Dimensionality Reduction using PCA
pca = PCA(n_components=0.95)  # Keep 95% variance in PCA
X_train = pca.fit_transform(X_train)  # Fit and transform training data
X_test = pca.transform(X_test)  # Transform test data

# Print original and reduced feature count
print(f"Original number of features: {X.shape[1]}")  # Display original feature count
print(f"Number of features after PCA: {X_train.shape[1]}")  # Display reduced feature count

# Map PCA components to original features
feature_names = data.feature_names
component_mappings = {}

for i, component in enumerate(pca.components_):
    most_influential_feature = feature_names[np.argmax(np.abs(component))]
    component_mappings[f'PC{i+1}'] = most_influential_feature

# Print the mapping in the required format
print("\nPCA Components Mapped to Original Features:")
for pc, feature in component_mappings.items():
    print(f"{pc}: {feature}")


# Convert transformed training data into a DataFrame
pca_columns = [f'PC{i+1}' for i in range(X_train.shape[1])]
df_pca_train = pd.DataFrame(X_train, columns=pca_columns)
df_pca_train['target'] = y_train  # Add target column

# Convert transformed test data into a DataFrame
df_pca_test = pd.DataFrame(X_test, columns=pca_columns)
df_pca_test['target'] = y_test  # Add target column

# Display first few rows of the PCA-transformed dataset
print("PCA Transformed Training Dataset:")
print(df_pca_train.head())

# Step 7: Initialize base models
svm = SVC(probability=True, random_state=42)  # Support Vector Machine model
logreg = LogisticRegression(random_state=42)  # Logistic Regression model
adaboost = AdaBoostClassifier(random_state=42)  # AdaBoost model

# Step 8: Hyperparameter tuning
param_grid_svm = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}  # Define SVM hyperparameters
grid_svm = GridSearchCV(SVC(probability=True), param_grid_svm, cv=5)  # Grid search for SVM

param_grid_logreg = {'C': [0.1, 1, 10], 'solver': ['liblinear']}  # Define Logistic Regression hyperparameters
grid_logreg = GridSearchCV(LogisticRegression(), param_grid_logreg, cv=5)  # Grid search for Logistic Regression

param_grid_adaboost = {'n_estimators': [50, 100], 'learning_rate': [0.5, 1.0]}  # Define AdaBoost hyperparameters
grid_adaboost = GridSearchCV(AdaBoostClassifier(), param_grid_adaboost, cv=5)  # Grid search for AdaBoost

# Train models with best hyperparameters
grid_svm.fit(X_train, y_train)  # Train SVM model
grid_logreg.fit(X_train, y_train)  # Train Logistic Regression model
grid_adaboost.fit(X_train, y_train)  # Train AdaBoost model

# Step 9: Retrieve best models
svm_best = grid_svm.best_estimator_  # Get best SVM model
logreg_best = grid_logreg.best_estimator_  # Get best Logistic Regression model
adaboost_best = grid_adaboost.best_estimator_  # Get best AdaBoost model

# Step 10: Create Voting Classifier
voting_clf = VotingClassifier(
    estimators=[('svm', svm_best), ('logreg', logreg_best), ('adaboost', adaboost_best)],
    voting='soft'
)

# Step 11: Train the model
voting_clf.fit(X_train, y_train)  # Train Voting Classifier

# Step 12: Evaluate the model
y_pred = voting_clf.predict(X_test)  # Predict on test data
y_prob = voting_clf.predict_proba(X_test)[:, 1]  # Get probability scores

# Compute evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
f1 = f1_score(y_test, y_pred)

# Print evaluation results
print(f"Voting Classifier Accuracy: {accuracy * 100:.2f}%")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"F1-Score: {f1:.4f}")
print(classification_report(y_test, y_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.2f}")

# Step 13: Save the model and preprocessing objects
os.makedirs('models', exist_ok=True)  # Create directory to store models
joblib.dump(voting_clf, 'models/voting_model.pkl')  # Save trained model
joblib.dump(scaler, 'models/scaler.pkl')  # Save scaler
joblib.dump(pca, 'models/pca.pkl')  # Save PCA model

# Step 14: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)  # Compute confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])  # Plot confusion matrix
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Step 15: ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)  # Compute ROC curve
roc_auc = auc(fpr, tpr)  # Compute AUC score
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')  # Plot ROC curve
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')  # Plot random guess line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid()
plt.show()
