import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
from scipy import stats

# Генерація синтетичних даних для демонстрації
np.random.seed(42)

# Кількість спостережень
n_samples = 100

# Погодні умови: температура, вологість, швидкість вітру
temperature = np.random.normal(15, 10, n_samples)  # температура в градусах Цельсія
humidity = np.random.normal(60, 15, n_samples)     # вологість у відсотках
wind_speed = np.random.normal(5, 3, n_samples)     # швидкість вітру в м/с

# Створення залежності енергоспоживання від погоди з шумом
# Більш низька температура -> більше споживання (опалення)
# Більш висока вологість -> більше споживання (кондиціонери)
# Більш високий вітер -> більше споживання (вентиляція)
energy_consumption = (
    200 - 3 * temperature +  # негативна кореляція з температурою
    0.5 * humidity +         # помірна позитивна кореляція з вологістю
    2 * wind_speed +         # позитивна кореляція з вітром
    np.random.normal(0, 15, n_samples)  # випадковий шум
)

# Збирання даних у DataFrame
data = pd.DataFrame({
    'temperature': temperature,
    'humidity': humidity,
    'wind_speed': wind_speed,
    'energy_consumption': energy_consumption
})

# Виведення перших рядків даних
print("Перші рядки сгенерованих даних:")
print(data.head())

# Описова статистика
print("\nОписова статистика:")
print(data.describe())

# Матриця кореляцій
print("\nМатриця кореляцій:")
correlation_matrix = data.corr()
print(correlation_matrix)

# Візуалізація матриці кореляцій
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Матриця кореляцій між змінними')
plt.tight_layout()
plt.savefig('correlation_matrix.png')
plt.show() 

# Підготовка даних для моделювання
X = data[['temperature', 'humidity', 'wind_speed']]
y = data['energy_consumption']

# Стандартизація ознак
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Розділення на тренувальну та тестову вибірки
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 1. Байєсівська регресія
bayesian_model = BayesianRidge(max_iter=300, tol=1e-6, alpha_1=1e-6, alpha_2=1e-6, 
                              lambda_1=1e-6, lambda_2=1e-6, compute_score=True)
bayesian_model.fit(X_train, y_train)

# Отримання прогнозів
y_pred_bayesian = bayesian_model.predict(X_test)
y_pred_train_bayesian = bayesian_model.predict(X_train)

# Оцінка якості моделі
mse_bayesian = mean_squared_error(y_test, y_pred_bayesian)
r2_bayesian = r2_score(y_test, y_pred_bayesian)

print("\nБайєсівська регресія:")
print(f"Коефіцієнти: {bayesian_model.coef_}")
print(f"Перехоплення (intercept): {bayesian_model.intercept_}")
print(f"Середньоквадратична похибка (MSE): {mse_bayesian}")
print(f"Коефіцієнт детермінації (R²): {r2_bayesian}")

# Отримання стандартних відхилень для коефіцієнтів
sigma_bayesian = np.sqrt(np.diag(bayesian_model.sigma_))
print(f"Стандартні відхилення коефіцієнтів: {sigma_bayesian}")

# 2. Класична лінійна регресія
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Отримання прогнозів
y_pred_linear = linear_model.predict(X_test)
y_pred_train_linear = linear_model.predict(X_train)

# Оцінка якості моделі
mse_linear = mean_squared_error(y_test, y_pred_linear)
r2_linear = r2_score(y_test, y_pred_linear)

print("\nКласична лінійна регресія:")
print(f"Коефіцієнти: {linear_model.coef_}")
print(f"Перехоплення (intercept): {linear_model.intercept_}")
print(f"Середньоквадратична похибка (MSE): {mse_linear}")
print(f"Коефіцієнт детермінації (R²): {r2_linear}")

# Порівняння результатів моделей
print("\nПорівняння моделей:")
print(f"Різниця в MSE: {mse_linear - mse_bayesian}")
print(f"Різниця в R²: {r2_bayesian - r2_linear}")

# Байєсівський аналіз достовірності та невизначеності параметрів

feature_names = ['Температура', 'Вологість', 'Швидкість вітру']
ci_lower = bayesian_model.coef_ - 1.96 * sigma_bayesian
ci_upper = bayesian_model.coef_ + 1.96 * sigma_bayesian

print("\nДовірчі інтервали для коефіцієнтів байєсівської регресії (95%):")
for i, feature in enumerate(feature_names):
    print(f"{feature}: {bayesian_model.coef_[i]:.4f} ± {1.96 * sigma_bayesian[i]:.4f} [{ci_lower[i]:.4f}, {ci_upper[i]:.4f}]")

# Візуалізація результатів

# 1. Порівняння прогнозів байєсівської та лінійної регресії
plt.figure(figsize=(12, 6))

# Прогнози на тестовій вибірці
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_bayesian, color='blue', alpha=0.7, label='Байєсівська регресія')
plt.scatter(y_test, y_pred_linear, color='red', alpha=0.5, label='Лінійна регресія')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Фактичне споживання')
plt.ylabel('Прогнозоване споживання')
plt.title('Порівняння прогнозів на тестовій вибірці')
plt.legend()

# Прогнози на тренувальній вибірці
plt.subplot(1, 2, 2)
plt.scatter(y_train, y_pred_train_bayesian, color='blue', alpha=0.7, label='Байєсівська регресія')
plt.scatter(y_train, y_pred_train_linear, color='red', alpha=0.5, label='Лінійна регресія')
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=2)
plt.xlabel('Фактичне споживання')
plt.ylabel('Прогнозоване споживання')
plt.title('Порівняння прогнозів на тренувальній вибірці')
plt.legend()

plt.tight_layout()
plt.savefig('prediction_comparison.png')
plt.show() 

# 2. Візуалізація залежностей енергоспоживання від окремих факторів
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Залежність від температури
axes[0].scatter(data['temperature'], data['energy_consumption'], alpha=0.6)
temp_range = np.linspace(data['temperature'].min(), data['temperature'].max(), 100).reshape(-1, 1)
temp_scaled = scaler.transform(np.hstack([
    temp_range, 
    np.ones_like(temp_range) * np.mean(data['humidity']), 
    np.ones_like(temp_range) * np.mean(data['wind_speed'])
]))
axes[0].plot(temp_range, bayesian_model.predict(temp_scaled), 'r-', linewidth=2, label='Байєсівська модель')
axes[0].set_xlabel('Температура (°C)')
axes[0].set_ylabel('Енергоспоживання')
axes[0].set_title('Залежність споживання від температури')
axes[0].legend()

# Залежність від вологості
axes[1].scatter(data['humidity'], data['energy_consumption'], alpha=0.6)
# Лінія регресії
humidity_range = np.linspace(data['humidity'].min(), data['humidity'].max(), 100).reshape(-1, 1)
humidity_scaled = scaler.transform(np.hstack([
    np.ones_like(humidity_range) * np.mean(data['temperature']),
    humidity_range,
    np.ones_like(humidity_range) * np.mean(data['wind_speed'])
]))
axes[1].plot(humidity_range, bayesian_model.predict(humidity_scaled), 'r-', linewidth=2, label='Байєсівська модель')
axes[1].set_xlabel('Вологість (%)')
axes[1].set_ylabel('Енергоспоживання')
axes[1].set_title('Залежність споживання від вологості')
axes[1].legend()

# Залежність від швидкості вітру
axes[2].scatter(data['wind_speed'], data['energy_consumption'], alpha=0.6)
wind_range = np.linspace(data['wind_speed'].min(), data['wind_speed'].max(), 100).reshape(-1, 1)
wind_scaled = scaler.transform(np.hstack([
    np.ones_like(wind_range) * np.mean(data['temperature']),
    np.ones_like(wind_range) * np.mean(data['humidity']),
    wind_range
]))
axes[2].plot(wind_range, bayesian_model.predict(wind_scaled), 'r-', linewidth=2, label='Байєсівська модель')
axes[2].set_xlabel('Швидкість вітру (м/с)')
axes[2].set_ylabel('Енергоспоживання')
axes[2].set_title('Залежність споживання від швидкості вітру')
axes[2].legend()

plt.tight_layout()
plt.savefig('weather_influence_plots.png')
plt.show()

# 3. Візуалізація невизначеності байєсівських коефіцієнтів
plt.figure(figsize=(8, 6))
plt.bar(feature_names, bayesian_model.coef_, yerr=sigma_bayesian, capsize=5, color='skyblue', edgecolor='black')
plt.title('Невизначеність байєсівських коефіцієнтів')
plt.xlabel('Фактори')
plt.ylabel('Коефіцієнти')
plt.tight_layout()
plt.savefig('coefficients_uncertainty.png')
plt.show() 

print("\nВИСНОВКИ:")
print("1. Байєсівська регресія надає додаткову інформацію про невизначеність через довірчі інтервали.")
print("2. Обидві моделі мають схожу точність.")
print(f"3. Найбільший вплив на споживання має температура ({bayesian_model.coef_[0]:.4f}).")
print(f"4. Найменший вплив має вологість ({bayesian_model.coef_[1]:.4f}).")
print("5. Байєсівський підхід корисний для аналізу ризиків.")
print("6. Байєсівська регресія підходить при малих даних або апріорній інформації.")
