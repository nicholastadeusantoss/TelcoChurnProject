# Importing data loading and cleaning function
from src.data.load_and_clean_data import load_and_clean_data

# Importing model training and tuning function
from src.models.train_models import train_and_tune_models

def main():
    # Loading and preprocessing the dataset
    print("Loading and preprocessing data...")
    X, y = load_and_clean_data('src/data/telco_churn.csv')

    # Training and tuning machine learning models
    print("Training and tuning models...")
    best_models, X_test, y_test = train_and_tune_models(X, y)

# Entry point of the script
if __name__ == "__main__":
    main()
