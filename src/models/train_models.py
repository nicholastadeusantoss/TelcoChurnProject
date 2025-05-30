import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
import joblib
import os

def train_and_tune_models(X, y):
    # Scale features to improve convergence of logistic regression
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the dataset into training and testing sets with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Define hyperparameter grid for Logistic Regression with 'liblinear' solver
    param_dist_lr_liblinear = {
        'penalty': ['l1', 'l2'],
        'C': np.logspace(-4, 4, 20),
        'solver': ['liblinear'],
        'class_weight': [None, 'balanced']
    }
    
    # Perform randomized search for Logistic Regression (liblinear)
    random_search_lr_liblinear = RandomizedSearchCV(
        LogisticRegression(max_iter=5000, random_state=42),
        param_distributions=param_dist_lr_liblinear,
        n_iter=10,
        scoring='roc_auc',
        cv=5,
        random_state=42,
        n_jobs=-1,
        error_score='raise'
    )
    random_search_lr_liblinear.fit(X_train, y_train)
    
    # Define hyperparameter grid for Logistic Regression with 'saga' solver
    param_dist_lr_saga = [
        {
            'penalty': ['l1', 'l2'],
            'C': np.logspace(-4, 4, 20),
            'solver': ['saga'],
            'class_weight': [None, 'balanced']
        },
        {
            'penalty': ['elasticnet'],
            'C': np.logspace(-4, 4, 20),
            'solver': ['saga'],
            'class_weight': [None, 'balanced'],
            'l1_ratio': [0, 0.5, 1]
        }
    ]
    
    # Perform randomized search for Logistic Regression (saga)
    random_search_lr_saga = RandomizedSearchCV(
        LogisticRegression(max_iter=5000, random_state=42),
        param_distributions=param_dist_lr_saga,
        n_iter=10,
        scoring='roc_auc',
        cv=5,
        random_state=42,
        n_jobs=-1,
        error_score='raise'
    )
    random_search_lr_saga.fit(X_train, y_train)
    
    # Select the best logistic regression model
    if random_search_lr_liblinear.best_score_ > random_search_lr_saga.best_score_:
        best_lr = random_search_lr_liblinear.best_estimator_
    else:
        best_lr = random_search_lr_saga.best_estimator_
    
    # Define hyperparameter grid for Random Forest
    param_dist_rf = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'class_weight': [None, 'balanced']
    }
    
    # Perform randomized search for Random Forest
    random_search_rf = RandomizedSearchCV(
        RandomForestClassifier(random_state=42),
        param_distributions=param_dist_rf,
        n_iter=20,
        scoring='roc_auc',
        cv=5,
        random_state=42,
        n_jobs=-1,
        error_score='raise'
    )
    random_search_rf.fit(X_train, y_train)
    
    # Define hyperparameter grid for XGBoost
    param_dist_xgb = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.3],
        'subsample': [0.5, 0.8, 1],
        'colsample_bytree': [0.5, 0.8, 1],
        'scale_pos_weight': [1, 2, 5]
    }
    
    # Perform randomized search for XGBoost
    random_search_xgb = RandomizedSearchCV(
        XGBClassifier(eval_metric='logloss', random_state=42),
        param_distributions=param_dist_xgb,
        n_iter=20,
        scoring='roc_auc',
        cv=5,
        random_state=42,
        n_jobs=-1,
        error_score='raise'
    )
    random_search_xgb.fit(X_train, y_train)
    
    # Store the best models from each algorithm
    best_models = {
        "Logistic Regression": best_lr,
        "Random Forest": random_search_rf.best_estimator_,
        "XGBoost": random_search_xgb.best_estimator_
    }
    
    # Evaluate all models on the test set
    for name, model in best_models.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        print(f"\n📊 {name}")
        print(classification_report(y_test, y_pred))
        print(f"AUC-ROC: {roc_auc_score(y_test, y_proba):.3f}")
    
    # Create models directory if it does not exist
    os.makedirs('models', exist_ok=True)
    
    # Save each model to a file
    for name, model in best_models.items():
        filename = f"models/{name.lower().replace(' ', '_')}_model.pkl"
        joblib.dump(model, filename)
        print(f"Model {name} saved to: {filename}")
    
    return best_models, X_test, y_test