# CS5228 Project

A machine learning project for resale car price prediction, utilizing both traditional ML approaches and deep learning with text features.

## Project Structure

### Data Directory
- `data/processed/` 
  - Processed dataset files (without GPT scores)
  - BERT embeddings for text features
- `data/with_text_features_train.csv` - Training data with GPT text scores
- `data/with_text_features_test.csv` - Test data with GPT text scores
- `data/predictions.csv` - Model predictions

### Text Feature Processing
- `gpt_requests.py` - Extract text feature scores using OpenAI API
- `bert.ipynb` - Generate BERT embeddings for text features
- `prompt.txt` - Prompt template for OpenAI API

### Data Processing & Models
- `Project EDA.ipynb` - Exploratory Data Analysis
- `preprocessing.ipynb` - Basic data preprocessing
- `preprocessing_gpt.ipynb` - Process data with GPT text scores

### Model Implementation
- `ml_models.ipynb` - Traditional ML models (without BERT)
- `ml_models_bert.ipynb` - ML models with BERT embeddings
- `nn_models.ipynb` - Neural Networks (without BERT)
- `nn_models_bert.ipynb` - Neural Networks with BERT embeddings

## Getting Started

1. Review the EDA notebook to understand the data
2. Run preprocessing steps:
   - Basic preprocessing in preprocessing.ipynb
   - Generate text scores using gpt_requests.py
   - Process data with text scores in preprocessing_text.ipynb
   - Generate BERT embeddings using bert.ipynb
3. Train models using either:
   - Traditional ML: ml_models.ipynb or ml_models_bert.ipynb
   - Neural Networks: nn_models.ipynb or nn_models_bert.ipynb
4. Generate predictions using the best performing model

## Requirements
- Python 3.x
- Required packages in requirements.txt
- OpenAI API key (for GPT text feature generation)