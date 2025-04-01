import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve, validation_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_curve, 
                             roc_auc_score, precision_recall_curve, average_precision_score, 
                             confusion_matrix, classification_report)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import seaborn as sns

# Tạo dữ liệu mẫu
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, 
                           n_redundant=5, n_clusters_per_class=2, random_state=42)

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Khởi tạo các mô hình
models = {
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'SVM': SVC(kernel='rbf', probability=True),
    'Decision Tree': DecisionTreeClassifier(max_depth=5),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=5),
    'Naive Bayes': GaussianNB()
}

# Huấn luyện và đánh giá các mô hình
results = {}
y_preds = {}
y_probs = {}

for name, model in models.items():
    # Huấn luyện mô hình
    model.fit(X_train_scaled, y_train)
    
    # Dự đoán
    y_pred = model.predict(X_test_scaled)
    y_preds[name] = y_pred
    
    # Xác suất dự đoán (cho đường cong ROC)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
    else:
        y_prob = model.decision_function(X_test_scaled)
    y_probs[name] = y_prob
    
    # Tính các thước đo
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Tính AUC-ROC
    auc_roc = roc_auc_score(y_test, y_prob)
    
    # Tính Average Precision (AP)
    ap = average_precision_score(y_test, y_prob)
    
    # Lưu kết quả
    results[name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'AUC-ROC': auc_roc,
        'Average Precision': ap
    }

# Hiển thị kết quả dưới dạng bảng
results_df = pd.DataFrame(results).T
print("Bảng so sánh hiệu năng của các mô hình:")
print(results_df)

# Vẽ biểu đồ so sánh độ chính xác
plt.figure(figsize=(12, 6))
results_df['Accuracy'].sort_values().plot(kind='barh')
plt.title('So sánh độ chính xác của các mô hình')
plt.xlabel('Độ chính xác')
plt.xlim(0, 1)
plt.grid(axis='x')
plt.tight_layout()
plt.show()

# Vẽ biểu đồ so sánh các thước đo
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC-ROC', 'Average Precision']
plt.figure(figsize=(15, 10))

for i, metric in enumerate(metrics):
    plt.subplot(2, 3, i+1)
    results_df[metric].sort_values().plot(kind='barh')
    plt.title(f'So sánh {metric}')
    plt.xlabel(metric)
    plt.xlim(0, 1)
    plt.grid(axis='x')

plt.tight_layout()
plt.show()

# Vẽ ma trận nhầm lẫn cho tất cả các mô hình
plt.figure(figsize=(15, 10))

for i, (name, y_pred) in enumerate(y_preds.items()):
    plt.subplot(2, 3, i+1)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Ma trận nhầm lẫn - {name}')
    plt.xlabel('Dự đoán')
    plt.ylabel('Thực tế')

plt.tight_layout()
plt.show()

# Vẽ đường cong ROC cho tất cả các mô hình
plt.figure(figsize=(10, 8))

for name, y_prob in y_probs.items():
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Đường cong ROC')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

# Vẽ đường cong Precision-Recall cho tất cả các mô hình
plt.figure(figsize=(10, 8))

for name, y_prob in y_probs.items():
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    plt.plot(recall, precision, label=f'{name} (AP = {ap:.3f})')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Đường cong Precision-Recall')
plt.legend(loc='lower left')
plt.grid(True)
plt.show()

# Thực hiện cross-validation
cv_results = {}
cv_scores = {}

for name, model in models.items():
    # Thực hiện 5-fold cross-validation
    scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    cv_scores[name] = scores
    cv_results[name] = {
        'Mean': scores.mean(),
        'Std': scores.std()
    }

# Hiển thị kết quả cross-validation
cv_df = pd.DataFrame(cv_results).T
print("\nKết quả 5-fold Cross-Validation:")
print(cv_df)

# Vẽ biểu đồ boxplot cho kết quả cross-validation
plt.figure(figsize=(12, 6))
cv_data = pd.DataFrame(cv_scores)
sns.boxplot(data=cv_data)
plt.title('Kết quả 5-fold Cross-Validation')
plt.ylabel('Độ chính xác')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# Vẽ đường cong học tập (learning curve) cho một mô hình (Random Forest)
plt.figure(figsize=(10, 6))
train_sizes, train_scores, test_scores = learning_curve(
    RandomForestClassifier(n_estimators=100, max_depth=5),
    X_train_scaled, y_train, cv=5, scoring='accuracy',
    train_sizes=np.linspace(0.1, 1.0, 10))

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.plot(train_sizes, train_mean, 'o-', color='r', label='Training score')
plt.plot(train_sizes, test_mean, 'o-', color='g', label='Cross-validation score')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='r')
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='g')
plt.xlabel('Số lượng mẫu huấn luyện')
plt.ylabel('Độ chính xác')
plt.title('Đường cong học tập - Random Forest')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

# Vẽ đường cong validation (validation curve) - thay đổi tham số max_depth cho Random Forest
param_range = np.arange(1, 11)
train_scores, test_scores = validation_curve(
    RandomForestClassifier(n_estimators=100),
    X_train_scaled, y_train, param_name="max_depth", param_range=param_range,
    cv=5, scoring='accuracy')

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(param_range, train_mean, 'o-', color='r', label='Training score')
plt.plot(param_range, test_mean, 'o-', color='g', label='Cross-validation score')
plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.1, color='r')
plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, alpha=0.1, color='g')
plt.xlabel('max_depth')
plt.ylabel('Độ chính xác')
plt.title('Đường cong validation - Random Forest')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()