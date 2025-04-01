import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.inspection import permutation_importance

# Tạo dữ liệu mẫu với nhiều đặc trưng hơn
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, 
                           n_redundant=2, n_clusters_per_class=2, random_state=42)

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Huấn luyện Random Forest
rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred = rf.predict(X_test)

# Đánh giá mô hình
print("Random Forest:")
print(f"Độ chính xác: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))

# Tính độ quan trọng của đặc trưng
importances = rf.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)
feature_names = [f"Feature {i}" for i in range(X.shape[1])]

# Vẽ độ quan trọng của đặc trưng
plt.figure(figsize=(12, 6))
plt.bar(feature_names, importances, yerr=std, align='center')
plt.xticks(rotation=90)
plt.title("Độ quan trọng của đặc trưng theo Random Forest")
plt.tight_layout()
plt.show()

# Tính permutation importance (thường chính xác hơn)
result = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42)
perm_importances = result.importances_mean
perm_std = result.importances_std

# Vẽ permutation importance
plt.figure(figsize=(12, 6))
plt.bar(feature_names, perm_importances, yerr=perm_std, align='center')
plt.xticks(rotation=90)
plt.title("Permutation Importance của đặc trưng")
plt.tight_layout()
plt.show()

# Thử nghiệm với các giá trị n_estimators khác nhau
n_trees = [1, 5, 10, 50, 100, 200]
train_scores = []
test_scores = []

for n in n_trees:
    rf = RandomForestClassifier(n_estimators=n, max_depth=5, random_state=42)
    rf.fit(X_train, y_train)
    
    train_score = accuracy_score(y_train, rf.predict(X_train))
    test_score = accuracy_score(y_test, rf.predict(X_test))
    
    train_scores.append(train_score)
    test_scores.append(test_score)

# Vẽ biểu đồ học tập
plt.figure(figsize=(10, 6))
plt.plot(n_trees, train_scores, 'o-', label='Tập huấn luyện')
plt.plot(n_trees, test_scores, 'o-', label='Tập kiểm tra')
plt.xlabel('Số lượng cây (n_estimators)')
plt.ylabel('Độ chính xác')
plt.title('Ảnh hưởng của số lượng cây đến hiệu suất Random Forest')
plt.legend()
plt.grid(True)
plt.show()

# Thử nghiệm với dữ liệu 2D để trực quan hóa đường biên quyết định
X_2d, y_2d = make_classification(n_samples=300, n_features=2, n_informative=2, 
                                n_redundant=0, n_clusters_per_class=1, random_state=42)
X_train_2d, X_test_2d, y_train_2d, y_test_2d = train_test_split(X_2d, y_2d, test_size=0.3, random_state=42)

# Huấn luyện Random Forest trên dữ liệu 2D
rf_2d = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf_2d.fit(X_train_2d, y_train_2d)

# Hàm trực quan hóa đường biên quyết định
def plot_decision_boundary(model, X, y, title):
    # Tạo lưới điểm
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
    # Dự đoán cho từng điểm trong lưới
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Vẽ đường biên quyết định
    plt.contourf(xx, yy, Z, alpha=0.3)
    
    # Vẽ các điểm dữ liệu
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', s=50)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(title)
    plt.xlabel("Đặc trưng 1")
    plt.ylabel("Đặc trưng 2")
    plt.colorbar(scatter)
    
    return plt

# Vẽ đường biên quyết định
plt.figure(figsize=(12, 6))
plot_decision_boundary(rf_2d, X_2d, y_2d, "Random Forest (100 cây, max_depth=5)")
plt.show()