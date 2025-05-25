import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import pickle
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Define constants
DATASET_PATH = "Dataset/dataset_capec_transfer.csv"
RESULTS_DIR = "Results/bert_transfer"
MODEL_SAVE_PATH = os.path.join(RESULTS_DIR, "bert_transfer_model.pth")
TOKENIZER_SAVE_PATH = os.path.join(RESULTS_DIR, "tokenizer.pkl")
LABEL_ENCODER_SAVE_PATH = os.path.join(RESULTS_DIR, "label_encoder.pkl")

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class CAPECTransferDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class BERTTransferModel:
    def __init__(self, model_name='bert-base-uncased'):
        self.model_name = model_name
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.label_encoder = None
        self.num_classes = 0
        self.model = None
        self.results_dir = RESULTS_DIR

    def load_data(self):
        print(f"Loading data from {DATASET_PATH}")
        
        chunks = []
        chunksize = 10000  # Define chunk size
        try:
            for chunk in pd.read_csv(DATASET_PATH, chunksize=chunksize, engine='python'):
                chunks.append(chunk)
            df = pd.concat(chunks, ignore_index=True)
        except Exception as e:
            print(f"Error reading CSV file in chunks: {e}")
            # Fallback or raise error
            raise e # Re-raise the exception after printing

        texts = df['text'].values
        labels = df['label'].values

        # Encode labels
        self.label_encoder = LabelEncoder()
        encoded_labels = self.label_encoder.fit_transform(labels)
        self.num_classes = len(self.label_encoder.classes_)
        print(f"Found {self.num_classes} unique labels.")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(texts, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels)

        return X_train, X_test, y_train, y_test

    def create_datasets(self, X_train, X_test, y_train, y_test, max_len=128):
        train_dataset = CAPECTransferDataset(X_train, y_train, self.tokenizer, max_len)
        test_dataset = CAPECTransferDataset(X_test, y_test, self.tokenizer, max_len)
        return train_dataset, test_dataset

    def build_model(self, num_classes):
        # Load pre-trained BERT model for sequence classification
        # The last layer will be automatically adjusted for the number of classes
        print(f"Loading pre-trained BERT model: {self.model_name}")
        self.model = BertForSequenceClassification.from_pretrained(self.model_name, num_labels=num_classes)
        self.model.to(device)
        print("BERT model built successfully.")

    def train(self, train_dataloader, epochs=3, learning_rate=2e-5):
        print("Training BERT model...")
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            print(f'Epoch {epoch + 1}/{epochs}')
            print('----------')

            self.model.train()
            total_loss = 0
            progress_bar = tqdm(train_dataloader, desc='Training')

            for batch in progress_bar:
                optimizer.zero_grad()
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss
                total_loss += loss.item()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item()/len(batch))})

            avg_train_loss = total_loss / len(train_dataloader)
            print(f'Average training loss: {avg_train_loss:.3f}')

        print("Training finished.")

    def evaluate(self, test_dataloader):
        print("Evaluating BERT model...")
        self.model.eval()
        predictions = []
        true_labels = []

        with torch.no_grad():
            for batch in test_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                logits = outputs.logits
                _, predicted = torch.max(logits, dim=1)

                predictions.extend(predicted.cpu().tolist())
                true_labels.extend(labels.cpu().tolist())

        accuracy = accuracy_score(true_labels, predictions)
        # Precision, Recall, F1 for multi-class require averaging. Use 'weighted' for imbalance.
        precision = precision_score(true_labels, predictions, average='weighted', zero_division=0)
        recall = recall_score(true_labels, predictions, average='weighted', zero_division=0)
        f1 = f1_score(true_labels, predictions, average='weighted', zero_division=0)
        cm = confusion_matrix(true_labels, predictions)

        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Test Precision: {precision:.4f}")
        print(f"Test Recall: {recall:.4f}")
        print(f"Test F1 Score: {f1:.4f}")

        return accuracy, precision, recall, f1, cm

    def save_results(self, accuracy, precision, recall, f1, cm):
        os.makedirs(self.results_dir, exist_ok=True)

        # Save metrics
        results_path = os.path.join(self.results_dir, "evaluation_results.pkl")
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm.tolist(), # Save as list for pickle
            'labels': self.label_encoder.classes_.tolist()
        }
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
        print(f"Evaluation results saved to {results_path}")

        # Save model state dictionary
        torch.save(self.model.state_dict(), MODEL_SAVE_PATH)
        print(f"Model saved to {MODEL_SAVE_PATH}")

        # Save tokenizer and label encoder
        with open(TOKENIZER_SAVE_PATH, 'wb') as f:
            pickle.dump(self.tokenizer, f)
        with open(LABEL_ENCODER_SAVE_PATH, 'wb') as f:
            pickle.dump(self.label_encoder, f)

        print("Tokenizer and Label Encoder saved.")

        # Generate and save confusion matrix plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.label_encoder.classes_, yticklabels=self.label_encoder.classes_)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()
        cm_plot_path = os.path.join(self.results_dir, "confusion_matrix.png")
        plt.savefig(cm_plot_path)
        print(f"Confusion matrix plot saved to {cm_plot_path}")
        plt.close()


    def run(self):
        X_train, X_test, y_train, y_test = self.load_data()
        train_dataset, test_dataset = self.create_datasets(X_train, X_test, y_train, y_test)

        train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

        self.build_model(num_classes=self.num_classes)
        self.train(train_dataloader)
        accuracy, precision, recall, f1, cm = self.evaluate(test_dataloader)
        self.save_results(accuracy, precision, recall, f1, cm)

if __name__ == "__main__":
    # Example of how to run this script directly
    # In a real transfer learning scenario, you might load a pre-trained model here
    # For simplicity, this example builds and trains from scratch on the transfer dataset
    bert_transfer = BERTTransferModel()
    bert_transfer.run() 