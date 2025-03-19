import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.datasets import make_classification, fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# 1. Gaussian Naive Bayes với dữ liệu liên tục
# Tạo dữ liệu mẫu
X, y = make_classification(n_samples=300, n_features=2, n_informative=2, 
                          n_redundant=0, n_clusters_per_class=1, random_state=42)

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Huấn luyện Gaussian Naive Bayes
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred = gnb.predict(X_test)
y_pred_prob = gnb.predict_proba(X_test)

# Đánh giá mô hình
print("Gaussian Naive Bayes:")
print(f"Độ chính xác: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))

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

# Trực quan hóa xác suất dự đoán
def plot_proba_contour(model, X, y):
    # Tạo lưới điểm
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
    # Dự đoán xác suất cho từng điểm trong lưới
    Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)
    
    # Vẽ đường đồng mức xác suất
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdBu_r)
    plt.colorbar(label='Xác suất thuộc lớp 1')
    
    # Vẽ đường biên quyết định (P=0.5)
    plt.contour(xx, yy, Z, levels=[0.5], colors='k', linestyles='-')
    
    # Vẽ các điểm dữ liệu
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', s=50, cmap=plt.cm.RdBu_r)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("Phân bố xác suất dự đoán của Gaussian Naive Bayes")
    plt.xlabel("Đặc trưng 1")
    plt.ylabel("Đặc trưng 2")
    
    return plt

# Vẽ đường biên quyết định và phân bố xác suất
plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
plot_decision_boundary(gnb, X, y, "Đường biên quyết định của Gaussian Naive Bayes")

plt.subplot(1, 2, 2)
plot_proba_contour(gnb, X, y)

plt.tight_layout()
plt.show()

# 2. Multinomial Naive Bayes với dữ liệu văn bản
# Sử dụng một tập dữ liệu nhỏ từ 20 Newsgroups
categories = ['alt.atheism', 'sci.space', 'comp.graphics', 'rec.sport.hockey']
newsgroups = fetch_20newsgroups(subset='train', categories=categories, remove=('headers', 'footers', 'quotes'))

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train_text, X_test_text, y_train_text, y_test_text = train_test_split(
    newsgroups.data, newsgroups.target, test_size=0.3, random_state=42)

# Trích xuất đặc trưng - Bag of Words
count_vect = CountVectorizer(stop_words='english', max_features=1000)
X_train_counts = count_vect.fit_transform(X_train_text)
X_test_counts = count_vect.transform(X_test_text)

# Trích xuất đặc trưng - TF-IDF
tfidf_vect = TfidfVectorizer(stop_words='english', max_features=1000)
X_train_tfidf = tfidf_vect.fit_transform(X_train_text)
X_test_tfidf = tfidf_vect.transform(X_test_text)

# Huấn luyện và đánh giá Multinomial Naive Bayes với CountVectorizer
mnb_count = MultinomialNB()
mnb_count.fit(X_train_counts, y_train_text)
y_pred_count = mnb_count.predict(X_test_counts)

print("\nMultinomial Naive Bayes với CountVectorizer:")
print(f"Độ chính xác: {accuracy_score(y_test_text, y_pred_count):.4f}")
print(classification_report(y_test_text, y_pred_count, target_names=categories))

# Huấn luyện và đánh giá Multinomial Naive Bayes với TF-IDF
mnb_tfidf = MultinomialNB()
mnb_tfidf.fit(X_train_tfidf, y_train_text)
y_pred_tfidf = mnb_tfidf.predict(X_test_tfidf)

print("\nMultinomial Naive Bayes với TF-IDF:")
print(f"Độ chính xác: {accuracy_score(y_test_text, y_pred_tfidf):.4f}")
print(classification_report(y_test_text, y_pred_tfidf, target_names=categories))

# Hiển thị ma trận nhầm lẫn
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
cm_count = confusion_matrix(y_test_text, y_pred_count)
sns.heatmap(cm_count, annot=True, fmt='d', cmap='Blues', xticklabels=categories, yticklabels=categories)
plt.title("Ma trận nhầm lẫn - CountVectorizer")
plt.xlabel("Dự đoán")
plt.ylabel("Thực tế")

plt.subplot(1, 2, 2)
cm_tfidf = confusion_matrix(y_test_text, y_pred_tfidf)
sns.heatmap(cm_tfidf, annot=True, fmt='d', cmap='Blues', xticklabels=categories, yticklabels=categories)
plt.title("Ma trận nhầm lẫn - TF-IDF")
plt.xlabel("Dự đoán")
plt.ylabel("Thực tế")

plt.tight_layout()
plt.show()

# Hiển thị các từ quan trọng nhất cho mỗi danh mục
def plot_top_features(vectorizer, classifier, class_names, n=10):
    feature_names = np.array(vectorizer.get_feature_names_out())
    plt.figure(figsize=(15, 10))
    
    for i, class_label in enumerate(class_names):
        # Lấy các hệ số đặc trưng cho lớp hiện tại
        top_features = classifier.feature_log_prob_[i, :]
        # Sắp xếp và lấy n đặc trưng hàng đầu
        top_n_idx = np.argsort(top_features)[-n:]
        
        plt.subplot(2, 2, i+1)
        plt.barh(range(n), top_features[top_n_idx])
        plt.yticks(range(n), feature_names[top_n_idx])
        plt.title(f"Top {n} từ trong danh mục '{class_label}'")
        plt.tight_layout()
    
    plt.tight_layout()
    plt.show()

plot_top_features(tfidf_vect, mnb_tfidf, categories)

# 3. Bernoulli Naive Bayes (phù hợp cho dữ liệu nhị phân)
# Tạo dữ liệu nhị phân từ dữ liệu văn bản (có từ hoặc không)
bin_vect = CountVectorizer(stop_words='english', max_features=1000, binary=True)
X_train_bin = bin_vect.fit_transform(X_train_text)
X_test_bin = bin_vect.transform(X_test_text)

# Huấn luyện và đánh giá Bernoulli Naive Bayes
bnb = BernoulliNB()
bnb.fit(X_train_bin, y_train_text)
y_pred_bin = bnb.predict(X_test_bin)

print("\nBernoulli Naive Bayes với biểu diễn nhị phân:")
print(f"Độ chính xác: {accuracy_score(y_test_text, y_pred_bin):.4f}")
print(classification_report(y_test_text, y_pred_bin, target_names=categories))

# So sánh hiệu suất của 3 kiểu Naive Bayes trên dữ liệu văn bản
models = ['MultinomialNB + Count', 'MultinomialNB + TF-IDF', 'BernoulliNB + Binary']
scores = [
    accuracy_score(y_test_text, y_pred_count),
    accuracy_score(y_test_text, y_pred_tfidf),
    accuracy_score(y_test_text, y_pred_bin)
]

plt.figure(figsize=(10, 6))
plt.bar(models, scores)
plt.ylim(0, 1.0)
plt.title('So sánh độ chính xác của các biến thể Naive Bayes trên dữ liệu văn bản')
plt.ylabel('Độ chính xác')
for i, v in enumerate(scores):
    plt.text(i, v + 0.01, f'{v:.4f}', ha='center')
plt.tight_layout()
plt.show()