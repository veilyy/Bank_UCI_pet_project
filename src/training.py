# Модуль обучения модели

import pandas as pd
import numpy as np

#Функция обучения baseline логистической регрессии
def train_logreg(X_train, y_train, **params):
    from sklearn.linear_model import LogisticRegression
    
    default_params = {
       'class_weight': 'balanced',
        'random_state': 42,
        'max_iter': 1000,
        'solver': 'lbfgs'
    }
    default_params.update(params)
    
    model = LogisticRegression(**default_params)
    model.fit(X_train, y_train)
    
    return model



def predict_with_model(model, X, threshold=0.5):
    
    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    
    return {
        'predictions': predictions.tolist(),
        'probabilities': probabilities.tolist(),
        'threshold': threshold
    }


def get_feature_importances(model, X_train, feature_names=None):
    #Функция для проверки важностей фичей, возвращает датафрейм с отсортированными по важности фичами
    # Если имена фичей не переданы, создаем
    if feature_names is None:
        feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
    
    result = pd.DataFrame({
        'feature': feature_names
    })
    
    # Для логистической регрессии
    if hasattr(model, 'coef_'):
        coef = model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
        result['coefficient'] = coef
        result['abs_coefficient'] = np.abs(coef)
    
    # Для деревьев
    elif hasattr(model, 'feature_importances_'):
        result['importance'] = model.feature_importances_
    
    # Сортируем по важности
    if 'abs_coefficient' in result.columns:
        result = result.sort_values('abs_coefficient', ascending=False)
    elif 'importance' in result.columns:
        result = result.sort_values('importance', ascending=False)
    
    # Ранжируем
    result['rank'] = range(1, len(result) + 1)
    
    return result    



def evaluate_model(model, predicts, y_test):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    accuracy = accuracy_score(y_test, predicts['predictions'])
    precision = precision_score(y_test, predicts['predictions'], zero_division=0)
    recall = recall_score(y_test, predicts['predictions'], zero_division=0)
    f1 = f1_score(y_test, predicts['predictions'], zero_division=0)
    roc_auc = roc_auc_score(y_test, predicts['probabilities'])
    
    metrics = {
        'model': type(model).__name__,
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'roc_auc' : float(roc_auc)
    }

    print(f"\nПредсказано 1: {sum(predicts['predictions'])} из {len(predicts['predictions'])}")
    print(f"Истинных 1:    {sum(y_test)} из {len(y_test)}")
    
    return metrics    

