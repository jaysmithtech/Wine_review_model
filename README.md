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
├── wine_models.py                    # ML models implementation (baseline + transformer)
├── wine_analysis_file_output.py     # RAG-based wine analysis system
├── wine_analysis_app.py              # Flask web app for wine analysis (optional)
├── requirements.txt                  # Python dependencies
├── datapath.txt                     # Path to dataset (gitignored)
├── .env                             # Environment variables (gitignored)
├── .gitignore                       # Git ignore file
├── wine_analysis_outputs/           # Generated analysis reports (markdown)
└── README.md                        # This file
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

### Machine Learning Models

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

### Wine Analysis System (RAG)

Generate comprehensive wine analysis reports:

```bash
python wine_analysis_file_output.py
```

This will:
- Load wine reviews and create vector embeddings
- Generate expert-level wine analysis using GPT-4
- Create detailed reports on wine characteristics, regions, and varieties
- Save all analyses as markdown files in `wine_analysis_outputs/`

**Generated Analysis Reports:**
- Dataset summary with statistics and top varieties/countries
- Highly rated wines characteristics (90+ points)
- Regional comparisons (Pinot Noir, Chardonnay, etc.)
- Price vs quality analysis
- Variety-specific flavor profiles
- Similar wine recommendations

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

### Core ML Dependencies
- pandas >= 1.3.0
- numpy >= 1.21.0
- scikit-learn >= 1.0.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- transformers >= 4.20.0
- torch >= 1.12.0
- datasets >= 2.0.0

### RAG Analysis Dependencies
- flask >= 2.0.0
- flask-cors >= 3.0.0
- python-dotenv >= 0.19.0
- langchain >= 0.1.0
- langchain-openai >= 0.1.0
- langchain-community >= 0.0.20
- faiss-cpu >= 1.7.0
- openai >= 1.0.0

## Results

### Machine Learning Models
The project demonstrates the effectiveness of both traditional ML and deep learning approaches for text-based wine quality prediction, with detailed performance metrics and visualizations to compare model effectiveness.

### Wine Analysis System
The RAG-powered analysis system generates expert-level insights about wine characteristics, regional differences, and quality patterns. Using OpenAI's GPT-4 with semantic search through wine descriptions, it produces comprehensive markdown reports covering:

- **Dataset insights**: Statistical analysis of 150k+ wine reviews
- **Quality analysis**: What makes wines highly rated (90+ points)
- **Regional comparisons**: How terroir affects wine characteristics
- **Variety profiles**: Typical flavor patterns for different grape varieties
- **Price analysis**: Relationship between cost and quality
- **Recommendations**: Similar wine suggestions based on descriptions

## Future Improvements

- Experiment with other transformer models (BERT, RoBERTa)
- Implement ensemble methods combining multiple approaches
- Add hyperparameter tuning and cross-validation
- Incorporate additional features (price, region, variety)
- Deploy models as a web application

## License

This project is open source and available under the MIT License.