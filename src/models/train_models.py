from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib

def train_and_tune_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Logistic Regression hyperparams
    param_dist_lr = {
        'penalty': ['l1', 'l2', 'elasticnet', None],
        'C': np.logspace(-4, 4, 20),
        'solver': ['liblinear', 'saga'],
        'class_weight': [None, 'balanced']
    }
    
    random_search_lr = RandomizedSearchCV(
        LogisticRegression(max_iter=1000),
        param_distributions=param_dist_lr,
        n_iter=20,
        scoring='roc_auc',
        cv=5,
        random_state=42,
        n_jobs=-1
    )
    random_search_lr.fit(X_train, y_train)
    
    # Random Forest hyperparams
    param_dist_rf = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'class_weight': [None, 'balanced']
    }
    random_search_rf = RandomizedSearchCV(
        RandomForestClassifier(random_state=42),
        param_distributions=param_dist_rf,
        n_iter=20,
        scoring='roc_auc',
        cv=5,
        random_state=42,
        n_jobs=-1
    )
    random_search_rf.fit(X_train, y_train)
    
    # XGBoost hyperparams
    param_dist_xgb = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.3],
        'subsample': [0.5, 0.8, 1],
        'colsample_bytree': [0.5, 0.8, 1],
        'scale_pos_weight': [1, 2, 5]
    }
    random_search_xgb = RandomizedSearchCV(
        XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        param_distributions=param_dist_xgb,
        n_iter=20,
        scoring='roc_auc',
        cv=5,
        random_state=42,
        n_jobs=-1
    )
    random_search_xgb.fit(X_train, y_train)
    
    best_models = {
        "Logistic Regression": random_search_lr.best_estimator_,
        "Random Forest": random_search_rf.best_estimator_,
        "XGBoost": random_search_xgb.best_estimator_
    }
    
    for name, model in best_models.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        print(f"\n📊 {name}")
        print(classification_report(y_test, y_pred))
        print(f"AUC-ROC: {roc_auc_score(y_test, y_proba):.3f}")
    
    # Salvar modelos
    for name, model in best_models.items():
        filename = f"models/{name.lower().replace(' ', '_')}_model.pkl"
        joblib.dump(model, filename)
        print(f"Modelo {name} salvo em: {filename}")
    
    return best_models, X_test, y_test