# Web Attack Detection System

A comprehensive machine learning research project for detecting and classifying web attacks using various deep learning and traditional machine learning approaches.

**Author:** Nguyễn Vũ Thành  
**Research Focus:** Machine Learning, Cybersecurity, Web Attack Detection

## 🎯 Project Overview

This project implements and compares multiple machine learning models for web attack detection, featuring both binary classification (Normal vs Attack) and multi-class classification (CAPEC-based attack categorization). The research explores traditional machine learning methods, deep learning approaches, and advanced transfer learning techniques.

## 🔬 Research Objectives

- **Binary Classification**: Distinguish between normal web traffic and attack patterns
- **Multi-class Classification**: Classify different types of attacks based on CAPEC (Common Attack Pattern Enumeration and Classification)
- **Transfer Learning**: Evaluate model performance across different datasets and attack types
- **Model Comparison**: Comprehensive analysis of various ML/DL approaches for cybersecurity applications

## 📊 Datasets

Due to file size limitations, datasets are hosted externally:

**📁 [Download Datasets](https://drive.google.com/drive/folders/1xP9yRUVrYaK5DKr-BjDPE8-0NXMM_ABe?dmr=1&ec=wgc-drive-hero-goto)**

### Dataset Description:

- `dataset_binary.csv` - Binary classification dataset (Normal/Attack)
- `dataset_capec_transfer.csv` - Multi-class CAPEC attack types for transfer learning
- `dataset_capec_test.csv` - Test dataset for CAPEC classification
- `dataset_attack_get_7000.csv` - 7,000 attack samples
- `dataset_normal_get_1000.csv` - 1,000 normal traffic samples

### Attack Types Covered:

- SQL Injection
- Cross-Site Scripting (XSS)
- Path Traversal
- Command Injection
- LDAP Injection
- And various other web attack patterns

## 🧠 Machine Learning Models

### 1. Baseline Models (`baseline_models.py`)

**Traditional ML approaches for binary classification:**

- Naive Bayes
- Decision Tree
- Random Forest
- Logistic Regression
- AdaBoost

**Feature Extraction:**

- Bag of Words (BOW)
- TF-IDF Vectorization

### 2. Deep Learning Models

#### BERT Models

- **`bert_model.py`**: BERT for binary classification
- **`bert_transfer_model.py`**: BERT with transfer learning for multi-class classification

#### LSTM Models

- **`lstm_model.py`**: Bidirectional LSTM for binary classification
- **`lstm_transfer_models.py`**: LSTM with transfer learning capabilities

#### GAN-based Transfer Learning

- **`gan_transfer_model.py`**: Generative Adversarial Network for attack detection and novel attack identification

### 3. Transfer Learning Models

- **`baseline_transfer.py`**: Traditional ML models applied to multi-class CAPEC classification
- Cross-dataset evaluation and model adaptation techniques

## 🏗️ Project Structure

```
Web_attack_detection/
├── Models/                          # ML/DL model implementations
│   ├── baseline_models.py          # Traditional ML models
│   ├── baseline_transfer.py        # Transfer learning with traditional ML
│   ├── bert_model.py              # BERT implementation
│   ├── bert_transfer_model.py     # BERT transfer learning
│   ├── lstm_model.py              # LSTM implementation
│   ├── lstm_transfer_models.py    # LSTM transfer learning
│   └── gan_transfer_model.py      # GAN-based transfer learning
├── Results/                        # Model outputs and evaluations
│   ├── baseline_binary/           # Baseline model results
│   ├── baseline_transfer/         # Transfer learning results
│   ├── bert_binary/              # BERT model results
│   ├── bert_transfer/            # BERT transfer results
│   ├── lstm_binary/              # LSTM model results
│   ├── lstm_transfer/            # LSTM transfer results
│   └── gan_capec_transfer/       # GAN model results
└── Dataset/                       # Training and test datasets (external)
```

## 📈 Key Features

### Advanced Analytics

- **Comprehensive Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score
- **Confusion Matrix Analysis**: Detailed classification performance visualization
- **Cross-Model Comparison**: Systematic comparison across all implemented models
- **Transfer Learning Assessment**: Model adaptation capabilities evaluation

### Novel Approaches

- **GAN-based Anomaly Detection**: Using adversarial training for novel attack detection
- **Cross-Dataset Transfer**: Evaluating model generalization across different datasets
- **Multi-Modal Feature Extraction**: BOW, TF-IDF, and deep embeddings

### Production-Ready Components

- **Model Serialization**: All trained models saved for deployment
- **Automated Pipeline**: End-to-end training and evaluation automation
- **Scalable Architecture**: Modular design for easy extension

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy scikit-learn tensorflow transformers matplotlib seaborn
```

### Running the Models

1. **Download datasets** from the provided Google Drive link
2. **Extract datasets** to the `Dataset/` directory
3. **Run individual models**:

```bash
# Traditional ML models
python Models/baseline_models.py

# Deep learning models
python Models/bert_model.py
python Models/lstm_model.py

# Transfer learning
python Models/bert_transfer_model.py
python Models/lstm_transfer_models.py
python Models/gan_transfer_model.py
```

## 📊 Results and Performance

### Model Performance Summary

All experimental results are systematically organized in the `Results/` directory:

- **Accuracy Summaries**: CSV files with detailed performance metrics
- **Confusion Matrices**: Visual classification performance analysis
- **Model Comparisons**: Comparative analysis across different approaches
- **Transfer Learning Evaluations**: Cross-dataset performance assessments

### Key Findings

- **BERT models** achieve superior performance in text-based attack detection
- **LSTM models** demonstrate strong sequential pattern recognition
- **GAN-based approaches** show promise for novel attack detection
- **Transfer learning** enables effective cross-domain attack classification

## 🔍 Research Applications

This research contributes to:

- **Cybersecurity Enhancement**: Real-time web attack detection systems
- **ML in Security**: Advanced machine learning applications in cybersecurity
- **Transfer Learning**: Cross-domain model adaptation techniques
- **Anomaly Detection**: Novel attack pattern identification

## 📧 Contact

**Nguyễn Vũ Thành**  
_Machine Learning Researcher_

For questions about this research or potential collaborations, please feel free to reach out.

## 📝 License

This project is developed for research and educational purposes. Please cite appropriately if used in academic work.

---

_This project demonstrates advanced machine learning techniques applied to cybersecurity challenges, showcasing expertise in both traditional ML and cutting-edge deep learning approaches._
