import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
# from sklearn.multiclass import OneVsRestClassifier # Không cần cho đa lớp
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd

# --- Cấu hình ---
RESULTS_DIR = 'Results/baseline_transfer_multiclass' # Đổi tên thư mục kết quả để phân biệt
DATASET_PATH = 'Dataset/dataset_capec_transfer.csv'
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Tạo thư mục kết quả nếu chưa tồn tại
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- Tải và chuẩn bị dữ liệu ---
print(f"Đang tải dữ liệu từ: {DATASET_PATH}")
try:
    df = pd.read_csv(DATASET_PATH)
    print("Đã tải dữ liệu thành công.")
    print(f"Kích thước dữ liệu đọc được: {df.shape}")
    print("Các cột trong dữ liệu:", df.columns.tolist())

    # Giả định cột 'text' chứa văn bản và cột 'label' chứa nhãn
    if 'text' not in df.columns:
        print("Lỗi: Không tìm thấy cột 'text' trong file CSV.")
        # Thay thế 'text_column_name' bằng tên cột văn bản thực tế của bạn nếu nó khác
        # Thử dùng cột đầu tiên nếu không có cột 'text'
        if len(df.columns) > 0:
            text_column_name = df.columns[0]
            print(f"Sử dụng cột đầu tiên '{text_column_name}' làm cột văn bản.")
        else:
            raise ValueError("Không tìm thấy cột nào trong file CSV.")
    else:
        text_column_name = 'text'

    if 'label' not in df.columns:
         raise ValueError("Không tìm thấy cột 'label' trong file CSV.")

    X = df[text_column_name].astype(str).tolist() # Lấy cột văn bản

    # Lấy cột nhãn (đa lớp)
    y = df['label'].values # Lấy cột nhãn dưới dạng mảng NumPy

    # Lấy danh sách các nhãn duy nhất để sử dụng cho ma trận nhầm lẫn
    label_names = sorted(np.unique(y).tolist())

    print(f"Kích thước dữ liệu X (văn bản): {len(X)}")
    print(f"Kích thước dữ liệu y (nhãn): {y.shape}")
    print(f"Các nhãn duy nhất được tìm thấy: {label_names}")

except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy file dữ liệu tại {DATASET_PATH}.")
    print("Vui lòng kiểm tra lại đường dẫn và đảm bảo file tồn tại.")
    exit() # Dừng chương trình nếu không tìm thấy file
except ValueError as ve:
    print(f"Lỗi xử lý dữ liệu: {ve}")
    exit() # Dừng chương trình nếu có lỗi về cấu trúc dữ liệu
except Exception as e:
    print(f"Đã xảy ra lỗi khi tải hoặc xử lý dữ liệu: {e}")
    exit()

# TODO: Kết thúc phần tải và chuẩn bị dữ liệu

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y) # Sử dụng stratify cho phân loại đa lớp

print(f"Kích thước tập huấn luyện X: {len(X_train)}")
print(f"Kích thước tập kiểm tra X: {len(X_test)}")
print(f"Kích thước tập huấn luyện y: {y_train.shape}")
print(f"Kích thước tập kiểm tra y: {y_test.shape}")

# --- Khai báo Vectorizers và Models ---
vectorizers = {
    'BOW': CountVectorizer(max_features=1000), # Tăng max_features cho đa lớp có thể hữu ích
    'TFIDF': TfidfVectorizer(max_features=1000)
}

# Các mô hình, không cần OneVsRest nữa
models = {
    'NaiveBayes': MultinomialNB(),
    'DecisionTree': DecisionTreeClassifier(random_state=RANDOM_STATE),
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE), # Tăng n_estimators
    'LogisticRegression': LogisticRegression(max_iter=500, random_state=RANDOM_STATE), # Tăng max_iter
    'AdaBoost': AdaBoostClassifier(random_state=RANDOM_STATE)
}

# Bảng lưu độ chính xác tổng thể cho từng mô hình
overall_accuracy_results = {}

# --- Huấn luyện và đánh giá các mô hình ---
print("\n--- Bắt đầu huấn luyện và đánh giá các mô hình (Đa lớp) ---")

for vec_name, vectorizer in vectorizers.items():
    print(f"\nSử dụng Vectorizer: {vec_name}")

    # Vector hóa dữ liệu
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print(f"Kích thước dữ liệu sau vector hóa (train): {X_train_vec.shape}")
    print(f"Kích thước dữ liệu sau vector hóa (test): {X_test_vec.shape}")

    overall_accuracy_results[vec_name] = {}

    for model_name, model in models.items(): # Sử dụng model trực tiếp
        print(f"\n  Sử dụng Model: {model_name}")

        print("  Đang huấn luyện mô hình...")
        model.fit(X_train_vec, y_train) # Huấn luyện trực tiếp trên y_train
        print("  Huấn luyện hoàn tất.")

        # Đánh giá mô hình
        print("  Đang đánh giá mô hình...")
        y_pred = model.predict(X_test_vec)

        # Tính toán độ chính xác tổng thể (Accuracy)
        overall_acc = accuracy_score(y_test, y_pred)
        overall_accuracy_results[vec_name][model_name] = overall_acc
        print(f"  Độ chính xác (Accuracy) với {vec_name} + {model_name}: {overall_acc:.4f}")

        # --- Vẽ và lưu ma trận nhầm lẫn (Tổng thể) ---
        print("  Đang tạo và lưu ma trận nhầm lẫn tổng thể...")
        model_results_dir = os.path.join(RESULTS_DIR, f'{vec_name}_{model_name}')
        os.makedirs(model_results_dir, exist_ok=True)

        # Tính toán confusion matrix tổng thể cho tất cả các lớp
        cm = confusion_matrix(y_test, y_pred, labels=label_names) # Sử dụng label_names để đảm bảo thứ tự các lớp

        plt.figure(figsize=(10, 8)) # Tăng kích thước biểu đồ cho nhiều lớp
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=label_names, yticklabels=label_names)
        plt.title(f'Confusion Matrix ({vec_name} + {model_name})')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        confusion_matrix_path = os.path.join(model_results_dir, 'confusion_matrix_total.png') # Đổi tên file
        plt.savefig(confusion_matrix_path)
        print(f"    Đã lưu ma trận nhầm lẫn tổng thể tại: {confusion_matrix_path}")
        plt.close()

print("\n--- Huấn luyện và đánh giá tất cả mô hình hoàn tất ---")

# --- Lưu bảng độ chính xác tổng thể ---
print("\nĐang lưu bảng độ chính xác tổng thể...")
overall_acc_df = pd.DataFrame(overall_accuracy_results).T
overall_acc_csv_path = os.path.join(RESULTS_DIR, 'overall_accuracy_summary.csv')
overall_acc_df.to_csv(overall_acc_csv_path)
print(f"Đã lưu bảng độ chính xác tổng thể tại: {overall_acc_csv_path}")

# --- Lưu ảnh độ chính xác tổng thể (từ bảng) ---
print("Đang tạo và lưu ảnh tổng hợp độ chính xác...")

plt.figure(figsize=(10, 6))
sns.heatmap(overall_acc_df, annot=True, fmt='.4f', cmap='viridis')
plt.title('Overall Accuracy by Vectorizer and Model (Multiclass)') # Cập nhật tiêu đề
plt.xlabel('Model')
plt.ylabel('Vectorizer')
overall_summary_plot_path = os.path.join(RESULTS_DIR, 'overall_accuracy_heatmap.png')
plt.savefig(overall_summary_plot_path)
print(f"Đã lưu ảnh tổng hợp độ chính xác tại: {overall_summary_plot_path}")
plt.close()

print("\nQuá trình hoàn tất. Kết quả chi tiết được lưu trong các thư mục con trong Results/baseline_transfer_multiclass, và kết quả tổng hợp ở cấp cao nhất.") 