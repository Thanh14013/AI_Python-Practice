import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Tạo dữ liệu mẫu
X, y = make_classification(n_samples=300, n_features=2, n_informative=2, 
                           n_redundant=0, n_clusters_per_class=1, random_state=42)

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Huấn luyện mô hình kNN với k=5
k = 5
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred = knn.predict(X_test)

# Đánh giá mô hình
accuracy = accuracy_score(y_test, y_pred)
print(f"Độ chính xác của mô hình kNN với k={k}: {accuracy:.4f}")
print("\nBáo cáo phân loại:")
print(classification_report(y_test, y_pred))

# Hàm trực quan hóa kết quả
def plot_decision_boundary(model, X, y):
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
    plt.title(f"Đường biên quyết định của kNN với k={k}")
    plt.xlabel("Đặc trưng 1")
    plt.ylabel("Đặc trưng 2")
    plt.colorbar(scatter)
    
    return plt

# Vẽ đường biên quyết định
plt.figure(figsize=(10, 6))
plot_decision_boundary(knn, X, y)
plt.show()

# Thử nghiệm với các giá trị k khác nhau
k_values = [1, 3, 5, 15, 30]
plt.figure(figsize=(15, 10))

for i, k in enumerate(k_values):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    
    plt.subplot(2, 3, i+1)
    plot_decision_boundary(knn, X, y)
    plt.title(f"kNN với k={k}")

plt.tight_layout()
plt.show()