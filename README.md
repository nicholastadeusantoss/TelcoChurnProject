# Telco Customer Churn Prediction

Este projeto é uma análise e modelagem preditiva para prever o churn (cancelamento) de clientes da operadora de telecomunicações Telco.

## Estrutura do Projeto

telco-churn-project/
│
├── data/ # Dados brutos (dataset original)
├── src/
│ ├── data/ # Código para pré-processamento
│ ├── models/ # Código para treinamento de modelos
│ └── visualization/ # Código para gráficos e análises visuais
├── main.py # Script principal para executar pipeline completo
├── requirements.txt # Dependências do projeto
└── README.md # Este arquivo

## Como usar

1. Clone o repositório:
    git clone <https://github.com/nicholastadeusantoss/TelcoChurnProject>
    cd telco-churn-project

2. Instale as dependências:
    pip install -r requirements.txt

3. Baixe o dataset e coloque dentro da pasta `src/data/` (ou configure o caminho no código).
    https://www.kaggle.com/datasets/blastchar/telco-customer-churn

4. Execute o script principal:
    ## Funcionalidades

    - Pré-processamento dos dados
    - Treinamento de três modelos: Logistic Regression, Random Forest e XGBoost
    - Ajuste de hiperparâmetros via RandomizedSearchCV
    - Avaliação dos modelos (métricas, matrizes de confusão, curva ROC)
    - Explicabilidade com SHAP para o modelo XGBoost
    - Salvamento dos modelos treinados para produção

    ## Requisitos

    - Python 3.8+
    - Bibliotecas listadas no `requirements.txt`

    ## Autor

    - Nicholas Tadeu - nicholastadeusantoss@gmail.com
