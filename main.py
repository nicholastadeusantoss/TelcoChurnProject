from src.data.preprocessing import load_and_clean_data
from src.models.train_models import train_and_tune_models
from src.visualization.plots import plot_confusion_matrix, plot_roc_curve

def main():
    # Caminho do dataset
    data_path = 'src/data/WA_Fn-UseC_-Telco-Customer-Churn.csv'
    
    print("Carregando e pré-processando dados...")
    X, y = load_and_clean_data(data_path)
    
    print("Treinando e ajustando modelos...")
    best_models, X_test, y_test = train_and_tune_models(X, y)
    
    print("Gerando visualizações...")
    model_names = list(best_models.keys())
    models = list(best_models.values())
    
    for name, model in best_models.items():
        plot_confusion_matrix(model, X_test, y_test, name)
    
    plot_roc_curve(models, model_names, X_test, y_test)

if __name__ == '__main__':
    main()