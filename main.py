from src.data.load_and_clean_data import load_and_clean_data
from src.models.train_models import train_and_tune_models

def main():
    print("Carregando e pré-processando dados...")
    X, y = load_and_clean_data('src/data/telco_churn.csv')

    print("Treinando e ajustando modelos...")
    best_models, X_test, y_test = train_and_tune_models(X, y)

if __name__ == "__main__":
    main()
