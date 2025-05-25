import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# Đường dẫn dữ liệu
data_path = 'Dataset/dataset_binary.csv'
results_dir = 'Results/baseline_binary'
os.makedirs(results_dir, exist_ok=True)

# Đọc dữ liệu
df = pd.read_csv(data_path)
X = df['text'].astype(str)
y = df['label']

# Tách train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Các vectorizer
vectorizers = {
    'BOW': CountVectorizer(max_features=300),
    'TFIDF': TfidfVectorizer(max_features=300)
}

# Các mô hình
models = {
    'NaiveBayes': MultinomialNB(),
    'DecisionTree': DecisionTreeClassifier(random_state=42),
    'RandomForest': RandomForestClassifier(n_estimators=10, random_state=42),
    'LogisticRegression': LogisticRegression(max_iter=200, random_state=42),
    'AdaBoost': AdaBoostClassifier(random_state=42)
}

# Lưu kết quả
accuracy_table = {}
for vec_name, vectorizer in vectorizers.items():
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    accuracy_table[vec_name] = {}
    # Lưu vectorizer
    with open(os.path.join(results_dir, f'{vec_name}_vectorizer.pkl'), 'wb') as f:
        pickle.dump(vectorizer, f)
    for model_name, model in models.items():
        model.fit(X_train_vec, y_train)
        y_pred = model.predict(X_test_vec)
        acc = accuracy_score(y_test, y_pred)
        accuracy_table[vec_name][model_name] = acc
        # Lưu model
        with open(os.path.join(results_dir, f'{vec_name}_{model_name}_model.pkl'), 'wb') as f:
            pickle.dump(model, f)

# Lưu bảng kết quả
acc_df = pd.DataFrame(accuracy_table).T
acc_df.to_csv(os.path.join(results_dir, 'accuracy_summary.csv'))
print('Đã lưu kết quả baseline vào', results_dir) 