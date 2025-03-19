import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Tạo dữ liệu mẫu
X, y = make_classification(n_samples=300, n_features=2, n_informative=2, 
                           n_redundant=0, n_clusters_per_class=1, random_state=42)

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Huấn luyện Decision Tree
tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred = tree.predict(X_test)

# Đánh giá mô hình
print("Decision Tree:")
print(f"Độ chính xác: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))

# Hàm trực quan hóa kết quả
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
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plot_decision_boundary(tree, X, y, "Decision Tree (max_depth=3)")

# Vẽ cây quyết định
plt.subplot(1, 2, 2)
plot_tree(tree, filled=True, feature_names=[f"X{i}" for i in range(2)], class_names=["0", "1"])
plt.title("Cấu trúc của Decision Tree")

plt.tight_layout()
plt.show()

# Thử nghiệm với các giá trị max_depth khác nhau
max_depths = [1, 2, 3, 5, 10, None]
plt.figure(figsize=(15, 10))

for i, depth in enumerate(max_depths):
    tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
    tree.fit(X_train, y_train)
    
    plt.subplot(2, 3, i+1)
    plot_decision_boundary(tree, X, y, f"Decision Tree (max_depth={depth})")
    accuracy = accuracy_score(y_test, tree.predict(X_test))
    plt.title(f"max_depth={depth}, accuracy={accuracy:.4f}")

plt.tight_layout()
plt.show()