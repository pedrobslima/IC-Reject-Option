import json
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import optuna
#import logging
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC#, SVR # kernels: 'linear', 'poly' e 'rbf'
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier#, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score

# =================================================

with open('config.json', 'r') as f:
    CONFIG = json.load(f)

#with open('config.json', 'w') as f:
#    CONFIG['SEED'] = randint(0, 4294967295)
#    json.dump(CONFIG, f)
#    print(CONFIG['SEED'])

experiment_name = f"{CONFIG['VERSION']}_{CONFIG['SEED']}"
mlflow.set_experiment(experiment_name=experiment_name)

NUM_TRIALS=100

# ==================================================

scaler = StandardScaler()
X, y = make_moons(n_samples=CONFIG['TWO_MOONS']['N_SAMPLES'], 
                  noise=CONFIG['TWO_MOONS']['NOISE'], 
                  random_state=CONFIG['SEED'])
df = pd.DataFrame(dict(x0=X[:,0], x1=X[:,1], label=y))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=CONFIG['SEED'])
X_train_norm = scaler.fit_transform(X_train)
X_test_norm = scaler.transform(X_test)

# ==================================================

def log_model_to_mlflow(model, model_name, hyperparams, X_train, y_train, X_test, y_test):
    """
    Treina, avalia e registra um modelo no MLflow.
    
    Parameters:
    -----------
    model : estimator
        Modelo do scikit-learn ou compatível
    model_name : str
        Nome do modelo para registro
    hyperparams : dict
        Dicionário com os hiperparâmetros do modelo
    X_train, y_train : array-like
        Dados de treino
    X_test, y_test : array-like
        Dados de teste
    """
    with mlflow.start_run(run_name=model_name):
        # Treinar modelo
        model.fit(X_train, y_train)
        
        # Fazer predições
        y_pred = model.predict(X_test)
        
        # Calcular probabilidades para AUROC (se disponível)
        try:
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)
                # Para classificação binária
                if y_proba.shape[1] == 2:
                    auroc = roc_auc_score(y_test, y_proba[:, 1])
                # Para classificação multiclasse
                else:
                    auroc = roc_auc_score(y_test, y_proba)
            elif hasattr(model, 'decision_function'):
                y_scores = model.decision_function(X_test)
                if len(np.unique(y_test)) == 2:
                    auroc = roc_auc_score(y_test, y_scores)
                else:
                    auroc = roc_auc_score(y_test, y_scores)
            else:
                auroc = None
        except Exception as e:
            #print(f"Aviso: Não foi possível calcular AUROC para {model_name}: {e}")
            auroc = None
        
        # Calcular métricas
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # Registrar hiperparâmetros
        mlflow.log_params(hyperparams)
        
        # Registrar métricas
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        if auroc is not None:
            mlflow.log_metric("auroc", auroc)
        
        # Registrar modelo
        mlflow.sklearn.log_model(model, name=model_name)
        
        #print(f"\n{'='*60}")
        #print(f"Modelo: {model_name}")
        #print(f"{'='*60}")
        #print(f"Acurácia:  {accuracy:.4f}")
        #print(f"Precisão:  {precision:.4f}")
        #print(f"Recall:    {recall:.4f}")
        #print(f"F1-Score:  {f1:.4f}")
        #if auroc is not None:
            #print(f"AUROC:     {auroc:.4f}")
        #print(f"{'='*60}\n")
        
        return model
    
def load_model_by_name(experiment_name, run_name):
    """
    Carrega um modelo específico do MLflow pelo nome da run.
    
    Parameters:
    -----------
    experiment_name : str
        Nome do experimento MLflow
    run_name : str
        Nome da run (ex: "Random_Forest", "XGBoost")
    
    Returns:
    --------
    model : estimator
        Modelo carregado
    run_info : dict
        Informações da run (hiperparâmetros e métricas)
    """
    # Buscar experimento
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experimento '{experiment_name}' não encontrado!")
    
    # Buscar runs do experimento
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'"
    )
    
    if runs.empty:
        raise ValueError(f"Run '{run_name}' não encontrada no experimento '{experiment_name}'!")
    
    # Pegar a run mais recente se houver múltiplas
    run = runs.iloc[0]
    run_id = run.run_id
    print(run_id)
    # Carregar modelo
    model_uri = f"runs:/{run_id}/{run_name}"
    model = mlflow.sklearn.load_model(model_uri)
    
    # Extrair informações da run
    run_info = {
        'run_id': run_id,
        'run_name': run_name,
        'start_time': run.start_time,
        'params': {k.replace('params.', ''): v for k, v in run.items() if k.startswith('params.')},
        'metrics': {k.replace('metrics.', ''): v for k, v in run.items() if k.startswith('metrics.')}
    }
    
    #print(f"✓ Modelo '{run_name}' carregado com sucesso!")
    #print(f"  Run ID: {run_id}")
    ##print(f"  Métricas: F1={run_info['metrics'].get('f1_score', 'N/A'):.4f}")
    
    return model, run_info

# ======================================================

# 1. Define an objective function to be maximized.
def dtree_objective(trial:optuna.trial._trial.Trial):
    
    # 2. Suggest values for the hyperparameters using a trial object.
    max_depth = trial.suggest_int('max_depth', 5, 100, log=True)
    criterion = trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss'])
    min_samples_split = trial.suggest_int('min_samples_split', 2, 60)
    min_samples_leaf = trial.suggest_int('min_samples_leaf',1, 30)
    
    clf = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion,
                                 min_samples_split=min_samples_split,
                                 min_samples_leaf=min_samples_leaf).fit(X_train, y_train)

    score = cross_val_score(clf, X_train, y_train, scoring='f1', n_jobs=-1, cv=10).mean()
    
    return score

# 3. Create a study object and optimize the objective function.
try:
    _ = load_model_by_name(experiment_name=experiment_name, run_name='Decision_Tree')
except ValueError:
    dtree_study = optuna.create_study(direction='maximize')
    dtree_study.optimize(dtree_objective, n_trials=NUM_TRIALS)
    #print(dtree_study.best_trial)

    dtree_params = dtree_study.best_params
    dtree_model = DecisionTreeClassifier(**dtree_params, random_state=CONFIG['SEED'])
    dtree_trained = log_model_to_mlflow(
        dtree_model, "Decision_Tree", dtree_params, 
        X_train, y_train, X_test, y_test
    )

def sgd_objective(trial):
    loss = trial.suggest_categorical('loss', ['log_loss', 'modified_huber'])
    penalty = trial.suggest_categorical('penalty', ['l2', 'l1', 'elasticnet'])
    alpha = trial.suggest_float('alpha', 1e-5, 1e-1, log=True)
    learning_rate = trial.suggest_categorical('learning_rate', ['constant', 'optimal', 'invscaling', 'adaptive'])
    eta0 = trial.suggest_float('eta0', 1e-5, 1e-1, log=True)
    max_iter = trial.suggest_int('max_iter', 500, 2000)
    
    clf = SGDClassifier(
        loss=loss, penalty=penalty, alpha=alpha, 
        learning_rate=learning_rate, eta0=eta0, max_iter=max_iter,
        random_state=CONFIG['SEED']
    ).fit(X_train_norm, y_train)
    
    score = cross_val_score(clf, X_train_norm, y_train, scoring='f1', n_jobs=-1, cv=10)
    return score.mean()

try:
    _ = load_model_by_name(experiment_name=experiment_name, run_name='SGD')
except ValueError:
    sgd_study = optuna.create_study(direction='maximize')
    sgd_study.optimize(sgd_objective, n_trials=NUM_TRIALS)
    #print("SGD Best Trial:", sgd_study.best_trial)
    sgd_params = sgd_study.best_params
    sgd_model = SGDClassifier(**sgd_params, random_state=CONFIG['SEED'])
    sgd_trained = log_model_to_mlflow(
        sgd_model, "SGD", sgd_params,
        X_train_norm, y_train, X_test_norm, y_test
    )

def logreg_objective(trial):
    #penalty = trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet'])
    C = trial.suggest_float('C', 1e-3, 100, log=True)
    solver = trial.suggest_categorical('solver', ['lbfgs', 'liblinear', 'saga'])
    max_iter = trial.suggest_int('max_iter', 100, 1000)
    
    # Ajusta solver baseado no penalty
    params = {
        'penalty': 'l2', 'C': C, 'solver': solver, 
        'max_iter': max_iter, 'random_state': CONFIG['SEED']
    }
    
    clf = LogisticRegression(**params).fit(X_train_norm, y_train)
    score = cross_val_score(clf, X_train_norm, y_train, scoring='f1', n_jobs=-1, cv=10)
    return score.mean()

try:
    _ = load_model_by_name(experiment_name=experiment_name, run_name='Logistic_Regression')
except ValueError:
    logreg_study = optuna.create_study(direction='maximize')
    logreg_study.optimize(logreg_objective, n_trials=NUM_TRIALS)
    #print("Logistic Regression Best Trial:", logreg_study.best_trial)
    logreg_params = logreg_study.best_params
    logreg_model = LogisticRegression(**logreg_params, random_state=CONFIG['SEED'])
    logreg_trained = log_model_to_mlflow(
        logreg_model, "Logistic_Regression", logreg_params,
        X_train, y_train, X_test_norm, y_test
    )

def knn_objective(trial):
    n_neighbors = trial.suggest_int('n_neighbors', 1, 50)
    weights = trial.suggest_categorical('weights', ['uniform', 'distance'])
    metric = trial.suggest_categorical('metric', ['euclidean', 'manhattan', 'minkowski'])
    p = trial.suggest_int('p', 1, 5) if metric == 'minkowski' else 2
    
    clf = KNeighborsClassifier(
        n_neighbors=n_neighbors, weights=weights, 
        metric=metric, p=p
    ).fit(X_train_norm, y_train)
    
    score = cross_val_score(clf, X_train_norm, y_train, scoring='f1', n_jobs=-1, cv=10)
    return score.mean()

try:
    _ = load_model_by_name(experiment_name=experiment_name, run_name='KNN')
except ValueError:
    knn_study = optuna.create_study(direction='maximize')
    knn_study.optimize(knn_objective, n_trials=NUM_TRIALS)
    #print("KNN Best Trial:", knn_study.best_trial)
    knn_params = knn_study.best_params
    knn_model = KNeighborsClassifier(**knn_params)
    knn_trained = log_model_to_mlflow(
        knn_model, "KNN", knn_params,
        X_train_norm, y_train, X_test_norm, y_test
    )

def svm_linear_objective(trial):
    C = trial.suggest_float('C', 1e-3, 100, log=True)
    max_iter = trial.suggest_int('max_iter', 500, 3000)
    
    clf = SVC(
        kernel='linear', C=C, max_iter=max_iter, random_state=CONFIG['SEED']
    ).fit(X_train_norm, y_train)
    
    score = cross_val_score(clf, X_train_norm, y_train, scoring='f1', n_jobs=-1, cv=10)
    return score.mean()

try:
    _ = load_model_by_name(experiment_name=experiment_name, run_name='SVM_Linear')
except ValueError:
    svm_linear_study = optuna.create_study(direction='maximize')
    svm_linear_study.optimize(svm_linear_objective, n_trials=NUM_TRIALS)
    #print("SVM Linear Best Trial:", svm_linear_study.best_trial)
    svm_linear_params = svm_linear_study.best_params
    svm_linear_model = SVC(kernel='linear', **svm_linear_params, random_state=CONFIG['SEED'], probability=True)
    svm_linear_trained = log_model_to_mlflow(
        svm_linear_model, "SVM_Linear", svm_linear_params,
        X_train, y_train, X_test_norm, y_test
    )

def svm_poly_objective(trial):
    C = trial.suggest_float('C', 1e-3, 100, log=True)
    degree = trial.suggest_int('degree', 2, 5)
    gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])
    coef0 = trial.suggest_float('coef0', 0, 10)
    max_iter = trial.suggest_int('max_iter', 500, 3000)
    
    clf = SVC(
        kernel='poly', C=C, degree=degree, gamma=gamma, 
        coef0=coef0, max_iter=max_iter, random_state=CONFIG['SEED']
    ).fit(X_train_norm, y_train)
    
    score = cross_val_score(clf, X_train_norm, y_train, scoring='f1', n_jobs=-1, cv=10)
    return score.mean()

try:
    _ = load_model_by_name(experiment_name=experiment_name, run_name='SVM_Polynomial')
except ValueError:
    svm_poly_study = optuna.create_study(direction='maximize')
    svm_poly_study.optimize(svm_poly_objective, n_trials=NUM_TRIALS)
    #print("SVM Poly Best Trial:", svm_poly_study.best_trial)
    svm_poly_params = svm_poly_study.best_params
    svm_poly_model = SVC(kernel='poly', **svm_poly_params, random_state=CONFIG['SEED'], probability=True)
    svm_poly_trained = log_model_to_mlflow(
        svm_poly_model, "SVM_Polynomial", svm_poly_params,
        X_train_norm, y_train, X_test_norm, y_test
    )

def svm_rbf_objective(trial):
    C = trial.suggest_float('C', 1e-3, 100, log=True)
    gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])
    max_iter = trial.suggest_int('max_iter', 500, 3000)
    
    clf = SVC(
        kernel='rbf', C=C, gamma=gamma, max_iter=max_iter, random_state=CONFIG['SEED']
    ).fit(X_train_norm, y_train)
    
    score = cross_val_score(clf, X_train_norm, y_train, scoring='f1', n_jobs=-1, cv=10)
    return score.mean()

try:
    _ = load_model_by_name(experiment_name=experiment_name, run_name='SVM_RBF')
except ValueError:
    svm_rbf_study = optuna.create_study(direction='maximize')
    svm_rbf_study.optimize(svm_rbf_objective, n_trials=NUM_TRIALS)
    #print("SVM RBF Best Trial:", svm_rbf_study.best_trial)
    svm_rbf_params = svm_rbf_study.best_params
    svm_rbf_model = SVC(kernel='rbf', **svm_rbf_params, random_state=CONFIG['SEED'], probability=True)
    svm_rbf_trained = log_model_to_mlflow(
        svm_rbf_model, "SVM_RBF", svm_rbf_params,
        X_train_norm, y_train, X_test_norm, y_test
    )

def mlp_objective(trial):
    
    hidden_layer_sizes = trial.suggest_int('hidden_layer_sizes', 50, 500, step=5)
    activation = trial.suggest_categorical('activation', ['relu', 'tanh', 'logistic'])
    solver = trial.suggest_categorical('solver', ['adam', 'sgd'])
    alpha = trial.suggest_float('alpha', 1e-5, 1e-1, log=True)
    learning_rate = trial.suggest_categorical('learning_rate', ['constant', 'adaptive'])
    learning_rate_init = trial.suggest_float('learning_rate_init', 1e-4, 1e-2, log=True)
    #max_iter = trial.suggest_int('max_iter', 200, 1000)
    
    clf = MLPClassifier(max_iter=10000, early_stopping=True, 
        n_iter_no_change=20, shuffle=True,
        hidden_layer_sizes=(hidden_layer_sizes,), activation=activation,
        solver=solver, alpha=alpha, learning_rate=learning_rate,
        learning_rate_init=learning_rate_init,
        random_state=CONFIG['SEED']
    ).fit(X_train_norm, y_train)
    
    score = cross_val_score(clf, X_train_norm, y_train, scoring='f1', n_jobs=-1, cv=10)
    return score.mean()

try:
    _ = load_model_by_name(experiment_name=experiment_name, run_name='MLP')
except ValueError:
    mlp_study = optuna.create_study(direction='maximize')
    mlp_study.optimize(mlp_objective, n_trials=NUM_TRIALS)
    #print("MLP Best Trial:", mlp_study.best_trial)
    mlp_params = mlp_study.best_params
    mlp_params['hidden_layer_sizes'] = (mlp_params['hidden_layer_sizes'],)
    mlp_model = MLPClassifier(**mlp_params, random_state=CONFIG['SEED'])
    mlp_trained = log_model_to_mlflow(
        mlp_model, "MLP", mlp_params,
        X_train_norm, y_train, X_test_norm, y_test
    )

def rf_objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 50, 500, log=True)
    max_depth = trial.suggest_int('max_depth', 5, 100, log=True)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 60)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 30)
    #max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
    criterion = trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss'])
    
    clf = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, criterion=criterion,
        min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
        #max_features=max_features, 
        random_state=CONFIG['SEED'], n_jobs=-1
    ).fit(X_train, y_train)
    
    score = cross_val_score(clf, X_train, y_train, scoring='f1', n_jobs=-1, cv=10)
    return score.mean()

try:
    _ = load_model_by_name(experiment_name=experiment_name, run_name='Random_Forest')
except ValueError:
    rf_study = optuna.create_study(direction='maximize')
    rf_study.optimize(rf_objective, n_trials=NUM_TRIALS)
    #print("Random Forest Best Trial:", rf_study.best_trial)
    rf_params = rf_study.best_params
    rf_model = RandomForestClassifier(**rf_params, random_state=CONFIG['SEED'], n_jobs=-1)
    rf_trained = log_model_to_mlflow(
        rf_model, "Random_Forest", rf_params,
        X_train, y_train, X_test, y_test
    )

def gb_objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 50, 500, log=True)
    learning_rate = trial.suggest_float('learning_rate', 1e-3, 1, log=True)
    max_depth = trial.suggest_int('max_depth', 3, 20)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 60)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 30)
    subsample = trial.suggest_float('subsample', 0.5, 1.0)
    max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
    
    clf = GradientBoostingClassifier(
        n_estimators=n_estimators, learning_rate=learning_rate,
        max_depth=max_depth, min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf, subsample=subsample,
        max_features=max_features, random_state=CONFIG['SEED']
    ).fit(X_train, y_train)
    
    score = cross_val_score(clf, X_train, y_train, scoring='f1', n_jobs=-1, cv=10)
    return score.mean()

try:
    _ = load_model_by_name(experiment_name=experiment_name, run_name='Gradient_Boosting')
except ValueError:
    gb_study = optuna.create_study(direction='maximize')
    gb_study.optimize(gb_objective, n_trials=NUM_TRIALS)
    #print("Gradient Boosting Best Trial:", gb_study.best_trial)
    gb_params = gb_study.best_params
    gb_model = GradientBoostingClassifier(**gb_params, random_state=CONFIG['SEED'])
    gb_trained = log_model_to_mlflow(
        gb_model, "Gradient_Boosting", gb_params,
        X_train, y_train, X_test, y_test
    )

def ada_objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 50, 500, log=True)
    learning_rate = trial.suggest_float('learning_rate', 1e-3, 2, log=True)
    #algorithm = trial.suggest_categorical('algorithm', ['SAMME', 'SAMME.R'])
    
    clf = AdaBoostClassifier(
        n_estimators=n_estimators, learning_rate=learning_rate,
        #algorithm=algorithm, 
        random_state=CONFIG['SEED']
    ).fit(X_train, y_train)
    
    score = cross_val_score(clf, X_train, y_train, scoring='f1', n_jobs=-1, cv=10)
    return score.mean()

try:
    _ = load_model_by_name(experiment_name=experiment_name, run_name='AdaBoost')
except ValueError:
    ada_study = optuna.create_study(direction='maximize')
    ada_study.optimize(ada_objective, n_trials=NUM_TRIALS)
    #print("AdaBoost Best Trial:", ada_study.best_trial)
    ada_params = ada_study.best_params
    ada_model = AdaBoostClassifier(**ada_params, random_state=CONFIG['SEED'])
    ada_trained = log_model_to_mlflow(
        ada_model, "AdaBoost", ada_params,
        X_train, y_train, X_test, y_test
    )

def xgb_objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 50, 500, log=True)
    learning_rate = trial.suggest_float('learning_rate', 1e-3, 1, log=True)
    max_depth = trial.suggest_int('max_depth', 3, 20)
    min_child_weight = trial.suggest_int('min_child_weight', 1, 10)
    gamma = trial.suggest_float('gamma', 0, 5)
    #subsample = trial.suggest_float('subsample', 0.5, 1.0)
    #colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 1.0)
    reg_alpha = trial.suggest_float('reg_alpha', 1e-5, 100, log=True)
    reg_lambda = trial.suggest_float('reg_lambda', 1e-5, 100, log=True)
    
    clf = XGBClassifier(
        n_estimators=n_estimators, learning_rate=learning_rate,
        max_depth=max_depth, min_child_weight=min_child_weight,
        gamma=gamma, #1subsample=subsample, colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha, reg_lambda=reg_lambda, 
        random_state=CONFIG['SEED'], n_jobs=-1, eval_metric='logloss'
    ).fit(X_train, y_train)
    
    score = cross_val_score(clf, X_train, y_train, scoring='f1', n_jobs=1, cv=10)
    return score.mean()

try:
    _ = load_model_by_name(experiment_name=experiment_name, run_name='XGBoost')
except ValueError:
    xgb_study = optuna.create_study(direction='maximize')
    xgb_study.optimize(xgb_objective, n_trials=NUM_TRIALS)
    #print("XGBoost Best Trial:", xgb_study.best_trial)
    xgb_params = xgb_study.best_params
    xgb_model = XGBClassifier(**xgb_params, random_state=CONFIG['SEED'], n_jobs=-1, eval_metric='logloss')
    xgb_trained = log_model_to_mlflow(
        xgb_model, "XGBoost", xgb_params,
        X_train, y_train, X_test, y_test
    )