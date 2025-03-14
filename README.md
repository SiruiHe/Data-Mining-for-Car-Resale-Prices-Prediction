# Data Mining for Car Resale Prices Prediction

A data mining project for resale car price prediction, utilizing both traditional ML approaches and neural networks.

## Project Structure

### Data Directory (`data/`)
- `processed/` - Preprocessed datasets with different settings
  - Generated by preprocessing.ipynb with various configurations
- Prediction files:
  - `ml_valid.csv`, `ml_test.csv` - Predictions from tree-based models
  - `nn_valid.csv`, `nn_test.csv` - Predictions from neural networks
  - `full_ml_valid.csv`, `full_ml_test.csv` - Predictions using full feature set
  - `combined_test_predictions.csv`, `fullml_combined_test_predictions.csv` - Final ensemble predictions

### Core Processing & Models
- `preprocessing.ipynb` - Data preprocessing pipeline
  - Configurable parameters for:
    - Normalization
    - Outlier removal
    - Log/Root transformations
  - Outputs processed datasets to `data/processed/`

- `linear_regression.ipynb` - Linear regression analysis
  - Comparison of different feature transformations:
    - Original features
    - Log-transformed features
    - Root-transformed features
  - Performance evaluation across different transformations

- `ml_models.ipynb` - Tree-based models implementation
  - Multiple models:
    - Random Forest
    - XGBoost
    - LightGBM
  - Configurable features:
    - Optional GPT Score integration
    - BERT embeddings comparison
      - Different dimension reduction methods
      - Configurable embedding dimensions
  
- `nn_models.ipynb` - Neural Network implementation
  - Neural Network models
  - Different architectures exploration

- `combine.ipynb` - Model ensemble implementation
  - Weighted averaging of tree-based models and neural networks
  - Generates final prediction files

### Text Feature Processing
- `bert.ipynb` - Generate BERT embeddings for text features
- `gpt_requests.py` - Extract text feature scores using OpenAI API
- `prompt.txt` - Prompt template for OpenAI API

## Getting Started

1. Review the EDA notebook (`Project EDA.ipynb`)
2. Run preprocessing pipeline:
   ```
   preprocessing.ipynb
   ├── Configure preprocessing parameters
   │   ├── normalization
   │   ├── outlier_removal
   │   └── feature_transformations
   └── Generate processed datasets in data/processed/
   ```
3. [Optional] Text features are already generated and stored:
   - GPT scores generated using `gpt_requests.py`
   - BERT embeddings generated using `bert.ipynb`
4. Train models:
   - Tree-based models: `ml_models.ipynb`
   - Neural Networks: `nn_models.ipynb`
   - Linear Regression: `linear_regression.ipynb`
5. Generate ensemble predictions using `combine.ipynb`

## Requirements
- Python 3.x
- Required packages in requirements.txt
- OpenAI API key (for GPT text feature generation)
