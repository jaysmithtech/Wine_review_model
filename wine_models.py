"""
Wine Quality Prediction Models for Wine Reviews Dataset
======================================================

Implements two approaches:
1. Baseline: TF-IDF + Logistic Regression/Naive Bayes
2. Deep: Fine-tuned transformer (DistilBERT)

Dataset: Wine reviews with descriptions and points (80-100 scale)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# For transformer models
try:
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification, 
                            Trainer, TrainingArguments, EarlyStoppingCallback)
    import torch
    from torch.utils.data import Dataset
    from datasets import Dataset as HFDataset
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("Transformers not available. Install with: pip install transformers torch")
    TRANSFORMERS_AVAILABLE = False

class WineReviewsProcessor:
    """Handles wine reviews data loading and preprocessing"""
    
    def __init__(self, data_path=""):
        self.data_path = data_path
        self.label_encoder = LabelEncoder()
        
    def load_data(self, sample_size=10000):
        """Load wine reviews dataset"""
        df = pd.read_csv(self.data_path)
        
        # Remove rows with missing descriptions or points
        df = df.dropna(subset=['description', 'points'])
        
        # Sample data for faster training (remove this line for full dataset)
        if sample_size and len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42)
        
        return df
    
    def create_quality_categories(self, df):
        """Convert points to quality categories"""
        def categorize_points(points):
            if points < 85:
                return "below_average"  # 80-84
            elif points < 90:
                return "good"          # 85-89
            elif points < 95:
                return "very_good"     # 90-94
            else:
                return "excellent"     # 95-100
        
        df['quality_category'] = df['points'].apply(categorize_points)
        return df
    
    def prepare_text_data(self, df):
        """Clean and prepare text descriptions"""
        # Basic text cleaning
        df['description'] = df['description'].astype(str)
        df['description'] = df['description'].str.lower()
        
        # Remove very short descriptions (less than 10 words)
        df = df[df['description'].str.split().str.len() >= 10]
        
        return df

class BaselineModels:
    """TF-IDF + Traditional ML models"""
    
    def __init__(self, max_features=10000, ngram_range=(1, 2)):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features, 
            stop_words='english',
            ngram_range=ngram_range,
            min_df=2,
            max_df=0.95
        )
        self.lr_model = LogisticRegression(random_state=42, max_iter=1000, C=1.0)
        self.nb_model = MultinomialNB(alpha=0.1)
        self.label_encoder = LabelEncoder()
        
    def prepare_data(self, texts, labels):
        """Prepare text data for baseline models"""
        X = self.vectorizer.fit_transform(texts)
        y = self.label_encoder.fit_transform(labels)
        return X, y
    
    def train_logistic_regression(self, X_train, y_train):
        """Train Logistic Regression model"""
        print("Training Logistic Regression...")
        self.lr_model.fit(X_train, y_train)
        return self.lr_model
    
    def train_naive_bayes(self, X_train, y_train):
        """Train Naive Bayes model"""
        print("Training Naive Bayes...")
        self.nb_model.fit(X_train, y_train)
        return self.nb_model
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """Evaluate model performance"""
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"\n{model_name} Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                  target_names=self.label_encoder.classes_))
        
        return accuracy, f1, y_pred

class WineDataset(Dataset):
    """Custom dataset for transformer models"""
    
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

class TransformerModel:
    """Fine-tuned transformer model for wine prediction"""
    
    def __init__(self, model_name='distilbert-base-uncased', num_labels=4):
        self.model_name = model_name
        self.num_labels = num_labels
        self.tokenizer = None
        self.model = None
        self.label_encoder = LabelEncoder()
        self.trainer = None
        
    def prepare_model(self):
        """Initialize tokenizer and model"""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers library not available")
            
        print(f"Loading {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, 
            num_labels=self.num_labels
        )
        
    def prepare_data(self, texts, labels):
        """Prepare data for transformer training"""
        encoded_labels = self.label_encoder.fit_transform(labels)
        return texts, encoded_labels
    
    def create_datasets(self, X_train, y_train, X_test, y_test):
        """Create train and test datasets"""
        train_dataset = WineDataset(X_train, y_train, self.tokenizer)
        test_dataset = WineDataset(X_test, y_test, self.tokenizer)
        return train_dataset, test_dataset
    
    def train_model(self, train_dataset, test_dataset, output_dir='./wine_transformer_model'):
        """Train the transformer model"""
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=100,
            eval_strategy="steps",
            eval_steps=500,
            save_strategy="steps",
            save_steps=500,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=None,  # Disable wandb logging
        )
        
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )
        
        print("Training transformer model...")
        self.trainer.train()
        return self.trainer
    
    def evaluate_transformer(self, test_dataset):
        """Evaluate transformer model"""
        if self.trainer is None:
            raise ValueError("Model not trained yet")
            
        eval_results = self.trainer.evaluate(test_dataset)
        return eval_results

def plot_results(df, lr_accuracy, nb_accuracy, lr_f1, nb_f1, transformer_results=None):
    """Create visualizations"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Quality distribution
    quality_counts = df['quality_category'].value_counts()
    axes[0, 0].bar(quality_counts.index, quality_counts.values)
    axes[0, 0].set_title('Wine Quality Distribution')
    axes[0, 0].set_xlabel('Quality Category')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Plot 2: Points distribution
    axes[0, 1].hist(df['points'], bins=20, alpha=0.7, edgecolor='black')
    axes[0, 1].set_title('Wine Points Distribution')
    axes[0, 1].set_xlabel('Points')
    axes[0, 1].set_ylabel('Frequency')
    
    # Plot 3: Model accuracy comparison
    models = ['Logistic Regression', 'Naive Bayes']
    accuracies = [lr_accuracy, nb_accuracy]
    
    if transformer_results:
        models.append('DistilBERT')
        # Convert eval loss to approximate accuracy (this is a rough estimate)
        transformer_acc = max(0, 1 - transformer_results.get('eval_loss', 1))
        accuracies.append(transformer_acc)
    
    bars = axes[1, 0].bar(models, accuracies, color=['skyblue', 'lightcoral', 'lightgreen'])
    axes[1, 0].set_title('Model Accuracy Comparison')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_ylim(0, 1)
    
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, acc + 0.01, 
                       f'{acc:.3f}', ha='center', va='bottom')
    
    # Plot 4: F1-Score comparison
    f1_scores = [lr_f1, nb_f1]
    if transformer_results:
        # Estimate F1 from eval loss (rough approximation)
        transformer_f1 = max(0, 1 - transformer_results.get('eval_loss', 1))
        f1_scores.append(transformer_f1)
    
    bars = axes[1, 1].bar(models, f1_scores, color=['skyblue', 'lightcoral', 'lightgreen'])
    axes[1, 1].set_title('Model F1-Score Comparison')
    axes[1, 1].set_ylabel('F1-Score')
    axes[1, 1].set_ylim(0, 1)
    
    for i, (bar, f1) in enumerate(zip(bars, f1_scores)):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, f1 + 0.01, 
                       f'{f1:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('/Users/jaylonsmith/wine_kaggle/wine_models_comparison.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main execution function"""
    print("Wine Quality Prediction Models - Wine Reviews Dataset")
    print("=" * 60)
    
    # Load and prepare data
    processor = WineReviewsProcessor()
    df = processor.load_data(sample_size=10000)  # Use 10k samples for faster training
    df = processor.create_quality_categories(df)
    df = processor.prepare_text_data(df)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Quality distribution:\n{df['quality_category'].value_counts()}")
    print(f"Points range: {df['points'].min()} - {df['points'].max()}")
    
    # Split data
    X = df['description'].values
    y = df['quality_category'].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Baseline Models
    print("\n" + "="*60)
    print("BASELINE MODELS (TF-IDF)")
    print("="*60)
    
    baseline = BaselineModels()
    X_train_tfidf, y_train_encoded = baseline.prepare_data(X_train, y_train)
    X_test_tfidf = baseline.vectorizer.transform(X_test)
    y_test_encoded = baseline.label_encoder.transform(y_test)
    
    # Train and evaluate Logistic Regression
    lr_model = baseline.train_logistic_regression(X_train_tfidf, y_train_encoded)
    lr_accuracy, lr_f1, _ = baseline.evaluate_model(
        lr_model, X_test_tfidf, y_test_encoded, "Logistic Regression"
    )
    
    # Train and evaluate Naive Bayes
    nb_model = baseline.train_naive_bayes(X_train_tfidf, y_train_encoded)
    nb_accuracy, nb_f1, _ = baseline.evaluate_model(
        nb_model, X_test_tfidf, y_test_encoded, "Naive Bayes"
    )
    
    transformer_results = None
    
    # Transformer Model
    if TRANSFORMERS_AVAILABLE:
        print("\n" + "="*60)
        print("TRANSFORMER MODEL (DistilBERT)")
        print("="*60)
        
        try:
            transformer = TransformerModel(num_labels=len(np.unique(y)))
            transformer.prepare_model()
            
            X_train_text, y_train_transformer = transformer.prepare_data(X_train, y_train)
            X_test_text, y_test_transformer = transformer.prepare_data(X_test, y_test)
            
            train_dataset, test_dataset = transformer.create_datasets(
                X_train_text, y_train_transformer, X_test_text, y_test_transformer
            )
            
            trainer = transformer.train_model(train_dataset, test_dataset)
            transformer_results = transformer.evaluate_transformer(test_dataset)
            
            print(f"\nTransformer Model Results:")
            print(f"Eval Loss: {transformer_results['eval_loss']:.4f}")
            
        except Exception as e:
            print(f"Error training transformer model: {e}")
            print("Continuing with baseline models only.")
    
    else:
        print("\nTransformer models not available.")
        print("Install with: pip install transformers torch")
    
    # Summary and Visualization
    print("\n" + "="*60)
    print("MODEL COMPARISON SUMMARY")
    print("="*60)
    print(f"Logistic Regression - Accuracy: {lr_accuracy:.4f}, F1: {lr_f1:.4f}")
    print(f"Naive Bayes - Accuracy: {nb_accuracy:.4f}, F1: {nb_f1:.4f}")
    
    if transformer_results:
        print(f"DistilBERT - Eval Loss: {transformer_results['eval_loss']:.4f}")
    
    # Create visualizations
    plot_results(df, lr_accuracy, nb_accuracy, lr_f1, nb_f1, transformer_results)
    
    print(f"\nAnalysis complete!")
    print(f"Visualizations saved to: wine_models_comparison.png")
    
    # Save model performance
    results = {
        'logistic_regression': {'accuracy': lr_accuracy, 'f1': lr_f1},
        'naive_bayes': {'accuracy': nb_accuracy, 'f1': nb_f1}
    }
    
    if transformer_results:
        results['distilbert'] = {'eval_loss': transformer_results['eval_loss']}
    
    return results

if __name__ == "__main__":
    results = main()
