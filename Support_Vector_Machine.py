import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_classification, make_circles
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# Tạo dữ liệu mẫu không phân tách tuyến tính
X, y = make_circles(n_samples=300, noise=0.1, factor=0.2, random_state=42)

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Huấn luyện SVM với kernel tuyến tính
svm_linear = SVC(kernel='linear', C=1.0)
svm_linear.fit(X_train_scaled, y_train)

# Huấn luyện SVM với kernel RBF
svm_rbf = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_rbf.fit(X_train_scaled, y_train)

# Huấn luyện SVM với kernel đa thức
svm_poly = SVC(kernel='poly', C=1.0, degree=3, gamma='scale')
svm_poly.fit(X_train_scaled, y_train)

# Dự đoán trên tập kiểm tra
y_pred_linear = svm_linear.predict(X_test_scaled)
y_pred_rbf = svm_rbf.predict(X_test_scaled)
y_pred_poly = svm_poly.predict(X_test_scaled)

# Đánh giá mô hình
print("SVM với kernel tuyến tính:")
print(f"Độ chính xác: {accuracy_score(y_test, y_pred_linear):.4f}")
print(classification_report(y_test, y_pred_linear))

print("\nSVM với kernel RBF:")
print(f"Độ chính xác: {accuracy_score(y_test, y_pred_rbf):.4f}")
print(classification_report(y_test, y_pred_rbf))

print("\nSVM với kernel đa thức:")
print(f"Độ chính xác: {accuracy_score(y_test, y_pred_poly):.4f}")
print(classification_report(y_test, y_pred_poly))

# Hàm trực quan hóa kết quả
def plot_decision_boundary(model, X, y, title):
    # Tạo lưới điểm
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
    # Chuẩn hóa dữ liệu lưới
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_scaled = scaler.transform(grid)
    
    # Dự đoán cho từng điểm trong lưới
    Z = model.predict(grid_scaled)
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
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plot_decision_boundary(svm_linear, X, y, "SVM với kernel tuyến tính")

plt.subplot(1, 3, 2)
plot_decision_boundary(svm_rbf, X, y, "SVM với kernel RBF")

plt.subplot(1, 3, 3)
plot_decision_boundary(svm_poly, X, y, "SVM với kernel đa thức")

plt.tight_layout()
plt.show()

# Thử nghiệm với các giá trị C khác nhau cho kernel RBF
C_values = [0.1, 1, 10, 100]
plt.figure(figsize=(15, 10))

for i, C in enumerate(C_values):
    svm = SVC(kernel='rbf', C=C, gamma='scale')
    svm.fit(X_train_scaled, y_train)
    
    plt.subplot(2, 2, i+1)
    plot_decision_boundary(svm, X, y, f"SVM (RBF) với C={C}")

plt.tight_layout()
plt.show()