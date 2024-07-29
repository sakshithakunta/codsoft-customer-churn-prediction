import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

data = pd.read_csv('Churn_Modelling.csv')

print(data.info())
print(data.describe())
print(data.isnull().sum())

data.hist(bins=30, figsize=(15, 10))
plt.show()
sns.countplot(x='churn', data=data)
plt.show()
corr_matrix = data.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()
data['feature_with_missing_values'].fillna(data['feature_with_missing_values'].mean(), inplace=True)

label_encoders = {}
categorical_features = ['gender', 'plan_type', 'region']

for feature in categorical_features:
    le = LabelEncoder()
    data[feature] = le.fit_transform(data[feature])
    label_encoders[feature] = le
X = data.drop('churn', axis=1)
y = data['churn']

scaler = StandardScaler()
X = scaler.fit_transform(X)

(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=42)

log_reg = LogisticRegression()
rand_forest = RandomForestClassifier()
grad_boost = GradientBoostingClassifier()

log_reg.fit(X_train, y_train)
rand_forest.fit(X_train, y_train)
grad_boost.fit(X_train, y_train)

y_pred_log_reg = log_reg.predict(X_test)
y_pred_rand_forest = rand_forest.predict(X_test)
y_pred_grad_boost = grad_boost.predict(X_test)

models = {
    'Logistic Regression': y_pred_log_reg,
    'Random Forest': y_pred_rand_forest,
    'Gradient Boosting': y_pred_grad_boost
}

for model_name, y_pred in models.items():
    print(f"Model: {model_name}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"Precision: {precision_score(y_test, y_pred)}")
    print(f"Recall: {recall_score(y_test, y_pred)}")
    print(f"F1 Score: {f1_score(y_test, y_pred)}")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_pred)}\n")

best_model = grad_boost
