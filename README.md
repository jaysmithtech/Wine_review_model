# Wine Review Quality Prediction Models

A machine learning project that predicts wine quality ratings from textual wine reviews using both traditional and deep learning approaches.

## Overview

This project implements two different approaches to predict wine quality from wine review descriptions:

1. **Baseline Models**: TF-IDF vectorization with traditional ML classifiers
2. **Deep Learning Model**: Fine-tuned DistilBERT transformer

## Dataset

The project uses the **Wine Reviews Dataset** from Kaggle containing 150,000+ wine reviews with:
- Wine descriptions (text reviews)
- Quality points (80-100 scale)
- Wine metadata (country, variety, price, etc.)

Quality categories are created from the point scale:
- **Below Average**: 80-84 points
- **Good**: 85-89 points  
- **Very Good**: 90-94 points
- **Excellent**: 95-100 points

## Models Implemented

### Baseline Models (Traditional ML)
- **TF-IDF + Logistic Regression**: Text vectorization with logistic regression classifier
- **TF-IDF + Naive Bayes**: Text vectorization with multinomial naive bayes classifier

**Features:**
- 10,000 TF-IDF features with bigrams
- Stop word removal and text preprocessing
- Stratified train/test split

### Deep Learning Model
- **Fine-tuned DistilBERT**: Transformer model fine-tuned on wine descriptions

**Features:**
- Pre-trained DistilBERT tokenization
- 3 training epochs with early stopping
- Batch size optimization for performance

## Project Structure

```
wine_kaggle/
├── wine_models.py          # Main implementation with both model approaches
├── requirements.txt        # Python dependencies
├── datapath.txt           # Path to dataset (gitignored)
├── .gitignore            # Git ignore file
└── README.md             # This file
```

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/jaysmithtech/Wine_review_model.git
   cd Wine_review_model
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the dataset:**
   - Download the wine reviews dataset from Kaggle
   - Update the path in `datapath.txt` or modify the `data_path` in `wine_models.py`

## Usage

Run the complete model training and evaluation:

```bash
python wine_models.py
```

This will:
- Load and preprocess the wine reviews dataset
- Train both baseline models (Logistic Regression & Naive Bayes)
- Fine-tune the DistilBERT transformer model
- Generate performance comparisons and visualizations
- Save results to `wine_models_comparison.png`

## Model Performance

The models are evaluated using:
- **Accuracy**: Overall classification accuracy
- **F1-Score**: Weighted F1-score for multi-class classification
- **Classification Report**: Per-class precision, recall, and F1-score

Results include visualizations showing:
- Wine quality distribution in the dataset
- Wine points distribution histogram
- Model accuracy comparison
- Model F1-score comparison

## Key Features

- **Text Preprocessing**: Cleaning, tokenization, and feature extraction
- **Stratified Sampling**: Balanced train/test splits
- **Model Comparison**: Side-by-side evaluation of different approaches
- **Visualization**: Comprehensive plots and performance metrics
- **Scalability**: Configurable sample sizes for faster experimentation

## Dependencies

- pandas >= 1.3.0
- numpy >= 1.21.0
- scikit-learn >= 1.0.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- transformers >= 4.20.0
- torch >= 1.12.0
- datasets >= 2.0.0

## Results

The project demonstrates the effectiveness of both traditional ML and deep learning approaches for text-based wine quality prediction, with detailed performance metrics and visualizations to compare model effectiveness.

## Future Improvements

- Experiment with other transformer models (BERT, RoBERTa)
- Implement ensemble methods combining multiple approaches
- Add hyperparameter tuning and cross-validation
- Incorporate additional features (price, region, variety)
- Deploy models as a web application

## License

This project is open source and available under the MIT License.