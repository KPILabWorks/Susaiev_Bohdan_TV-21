from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np

# Функція для реалізації RFE з оцінкою результатів
def select_features_with_rfe(X, y, n_features=5, estimator=None):
    """
    Виконує вибір ознак за допомогою RFE
    
    Parameters:
    X : DataFrame з ознаками
    y : цільова змінна
    n_features : кількість ознак для вибору
    estimator : базова модель (за замовчуванням RandomForest)
    
    Returns:
    selected_features : список обраних ознак
    rfe : навчений RFE об'єкт
    """
    
    # RandomForest якщо модель не вказана
    if estimator is None:
        estimator = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Ініціалізація RFE
    rfe = RFE(estimator=estimator, n_features_to_select=n_features)
    
    # Навчання моделі
    rfe.fit(X, y)
    
    # Отримання обраних ознак
    selected_features = X.columns[rfe.support_].tolist()
    
    # Оцінка важливості за допомогою крос-валідації
    X_selected = X[selected_features]
    scores = cross_val_score(estimator, X_selected, y, cv=5)
    
    print(f"Обрано {len(selected_features)} ознак:")
    print(selected_features)
    print(f"\nСередня точність при крос-валідації: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
    
    # Детальний рейтинг всіх ознак
    feature_ranking = pd.DataFrame({
        'Feature': X.columns,
        'Ranking': rfe.ranking_
    }).sort_values('Ranking')
    
    print("\nРейтинг усіх ознак (1 - обрані):")
    print(feature_ranking)
    
    return selected_features, rfe

# Приклад використання
if __name__ == "__main__":
    # Створення тестових даних
    np.random.seed(42)
    X = pd.DataFrame(np.random.rand(100, 10), columns=[f'feature_{i}' for i in range(10)])
    y = np.random.randint(0, 2, 100)
    
    # Виконання RFE
    selected_features, rfe = select_features_with_rfe(X, y, n_features=5)
    
    # Використання обраних ознак
    X_selected = X[selected_features]