import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Tải dữ liệu hoa Iris
iris = load_iris()
X = iris.data  # 4 đặc trưng (sepal length, sepal width, petal length, petal width)
y = iris.target  # Nhãn (0: Setosa, 1: Versicolor, 2: Virginica)
feature_names = iris.feature_names
target_names = iris.target_names

# Chuyển nhãn thành dạng one-hot vector
encoder = OneHotEncoder(sparse_output=False)
y_encoded = encoder.fit_transform(y.reshape(-1, 1))

# Chia tập dữ liệu thành tập huấn luyện và tập kiểm tra (80% - 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Lưu bộ chuẩn hóa để sử dụng sau này
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(encoder, 'encoder.pkl')

# Hiển thị phân bố dữ liệu
def visualize_data():
    # Tạo DataFrame cho dễ phân tích
    import pandas as pd
    df = pd.DataFrame(data=iris.data, columns=feature_names)
    df['species'] = [target_names[i] for i in iris.target]
    
    # Tạo scatter plot để phân tích đặc trưng
    plt.figure(figsize=(12, 10))
    
    # Biểu đồ phân tán với 2 đặc trưng cánh hoa
    plt.subplot(2, 2, 1)
    sns.scatterplot(data=df, x='petal length (cm)', y='petal width (cm)', 
                    hue='species', palette='viridis', s=100)
    plt.title('Phân bố Chiều dài và Chiều rộng cánh hoa')
    
    # Biểu đồ phân tán với 2 đặc trưng đài hoa
    plt.subplot(2, 2, 2)
    sns.scatterplot(data=df, x='sepal length (cm)', y='sepal width (cm)', 
                    hue='species', palette='viridis', s=100)
    plt.title('Phân bố Chiều dài và Chiều rộng đài hoa')
    
    # Biểu đồ violin của các đặc trưng
    plt.subplot(2, 2, 3)
    sns.violinplot(x='species', y='petal length (cm)', data=df, palette='viridis')
    plt.title('Phân bố chiều dài cánh hoa theo loài')
    
    # Hiển thị ma trận tương quan
    plt.subplot(2, 2, 4)
    corr = df.iloc[:, :-1].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title('Ma trận tương quan giữa các đặc trưng')
    
    plt.tight_layout()
    plt.savefig('iris_data_analysis.png')
    plt.close()

# Xây dựng mô hình ANN với các kỹ thuật hiện đại
def build_and_train_model():
    # Xây dựng mô hình ANN
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(4,)),  # Lớp ẩn 1
        keras.layers.Dropout(0.3),  # Thêm Dropout để giảm overfitting
        keras.layers.Dense(32, activation='relu'),  # Lớp ẩn 2
        keras.layers.Dropout(0.2),
        keras.layers.Dense(16, activation='relu'),  # Thêm một lớp ẩn nữa
        keras.layers.Dense(3, activation='softmax')  # Lớp đầu ra
    ])

    # Compile mô hình
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    # Thêm callbacks
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=20, restore_best_weights=True)
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=10, min_lr=0.00001)

    # Huấn luyện mô hình
    history = model.fit(
        X_train, y_train, 
        epochs=150, 
        batch_size=16, 
        validation_data=(X_test, y_test),
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    return model, history

def evaluate_model(model, history):
    # Vẽ quá trình huấn luyện
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='train accuracy')
    plt.plot(history.history['val_accuracy'], label='validation accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Độ chính xác trong quá trình huấn luyện')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss trong quá trình huấn luyện')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()
    
    # Đánh giá mô hình
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f'Độ chính xác trên tập kiểm tra: {acc * 100:.2f}%')
    
    # Dự đoán và tạo báo cáo phân loại chi tiết
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # In báo cáo phân loại chi tiết
    print("\n--- BÁO CÁO PHÂN LOẠI CHI TIẾT ---")
    print(classification_report(y_true, y_pred_classes, target_names=target_names))
    
    # Vẽ ma trận nhầm lẫn
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred_classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names,
                yticklabels=target_names)
    plt.xlabel('Dự đoán')
    plt.ylabel('Thực tế')
    plt.title('Ma trận nhầm lẫn')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

# HÀM DỰ ĐOÁN LOÀI HOA IRIS
def predict_iris():
    print('\nNhập thông tin về bông hoa Iris:')
    sepal_length = float(input('Chiều dài đài hoa (cm): '))
    sepal_width = float(input('Chiều rộng đài hoa (cm): '))
    petal_length = float(input('Chiều dài cánh hoa (cm): '))
    petal_width = float(input('Chiều rộng cánh hoa (cm): '))
    
    # Tải mô hình và bộ chuẩn hóa
    model = keras.models.load_model('iris_ann_model.keras')
    scaler = joblib.load('scaler.pkl')
    
    # Tiền xử lý dữ liệu nhập vào
    sample = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    sample = scaler.transform(sample)  # Chuẩn hóa dữ liệu
    
    # Dự đoán
    prediction = model.predict(sample)
    predicted_class = np.argmax(prediction)
    
    print(f'Loài hoa dự đoán: {target_names[predicted_class]}')
    print(f'Xác suất các loài:')
    for i, species in enumerate(target_names):
        print(f'- {species}: {prediction[0][i]*100:.2f}%')

# Hàm dự đoán nhiều mẫu (ví dụ thực tế)
def predict_multiple_samples():
    # Một số mẫu thử nghiệm
    samples = [
        [5.1, 3.5, 1.4, 0.2],  # Setosa
        [6.7, 3.1, 4.4, 1.4],  # Versicolor
        [6.3, 3.3, 6.0, 2.5],  # Virginica
        [5.5, 2.6, 4.4, 1.2],  # Mẫu phức tạp hơn
    ]
    
    # Tải mô hình và bộ chuẩn hóa
    model = keras.models.load_model('iris_ann_model.keras')
    scaler = joblib.load('scaler.pkl')
    
    # Chuẩn hóa dữ liệu
    samples = scaler.transform(samples)
    
    # Dự đoán
    predictions = model.predict(samples)
    predicted_classes = np.argmax(predictions, axis=1)
    
    print("\n--- DỰ ĐOÁN NHIỀU MẪU ---")
    for i, (sample, pred_class) in enumerate(zip(samples, predicted_classes)):
        print(f"\nMẫu {i+1}:")
        print(f"- Dự đoán: {target_names[pred_class]}")
        print(f"- Xác suất các loài:")
        for j, species in enumerate(target_names):
            print(f"  + {species}: {predictions[i][j]*100:.2f}%")

# Chạy chương trình chính
if __name__ == "__main__":
    # Phân tích và hiển thị dữ liệu
    print("Đang phân tích và hiển thị dữ liệu...")
    visualize_data()
    
    # Huấn luyện mô hình
    print("\nĐang huấn luyện mô hình ANN...")
    model, history = build_and_train_model()
    
    # Đánh giá mô hình
    print("\nĐánh giá mô hình")
    evaluate_model(model, history)
    
    # Lưu mô hình
    model.save('iris_ann_model.keras')
    print("\nĐã lưu mô hình vào file 'iris_ann_model.keras'")
    
    # Dự đoán với nhiều mẫu (ví dụ thực tế)
    predict_multiple_samples()
    
    # Cho phép người dùng nhập mẫu mới để dự đoán
    while True:
        choice = input("\nBạn có muốn nhập thông tin để dự đoán loài hoa không? (y/n): ")
        if choice.lower() != 'y':
            break
        predict_iris()
