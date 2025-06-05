import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Налаштування для графіків
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

class LightAnalyzer:
    def __init__(self):
        self.data = None
    
    def generate_sample_data(self):
        """Генерація прикладних даних освітленості"""
        print("Генерація даних освітленості...")
        
        # Створюємо дані на 24 години
        hours = 24
        points_per_hour = 60  # кожну хвилину
        total_points = hours * points_per_hour
        
        # Базовий час
        start_time = datetime(2024, 6, 1, 0, 0, 0)
        timestamps = [start_time + timedelta(minutes=i) for i in range(total_points)]
        
        # Генерація реалістичних даних освітленості
        light_data = []
        
        for i, timestamp in enumerate(timestamps):
            hour = timestamp.hour
            minute = timestamp.minute
            
            # Природне освітлення (денне світло)
            if 6 <= hour <= 18:  # день
                base_light = 300 + 400 * np.sin((hour - 6) * np.pi / 12)
                natural_light = max(0, base_light + np.random.normal(0, 50))
            else:  # ніч
                natural_light = np.random.uniform(0, 20)
            
            # Штучне освітлення (ввечері та вранці)
            if 7 <= hour <= 9 or 18 <= hour <= 23:  # активні години
                artificial_light = np.random.uniform(150, 300)
            elif 23 <= hour or hour <= 6:  # ніч
                artificial_light = np.random.uniform(0, 50)
            else:  # день
                artificial_light = np.random.uniform(50, 150)
            
            # Комбіноване освітлення
            if 7 <= hour <= 9 or 17 <= hour <= 19:
                combined_light = natural_light + artificial_light
            else:
                combined_light = max(natural_light, artificial_light)
            
            # Додаємо шум
            final_light = max(0, combined_light + np.random.normal(0, 20))
            
            light_data.append({
                'timestamp': timestamp,
                'hour': hour,
                'minute': minute,
                'light_level': final_light,
                'period': self.get_period(hour)
            })
        
        self.data = pd.DataFrame(light_data)
        print(f"Згенеровано {len(self.data)} записів")
        return True
    
    def get_period(self, hour):
        """Визначення періоду дня"""
        if 6 <= hour < 12:
            return 'Ранок'
        elif 12 <= hour < 18:
            return 'День'
        elif 18 <= hour < 22:
            return 'Вечір'
        else:
            return 'Ніч'
    
    def load_data(self, file_path):
        """Завантаження даних з CSV"""
        try:
            self.data = pd.read_csv(file_path)
            # Перетворення timestamp якщо потрібно
            if 'timestamp' in self.data.columns:
                self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
            
            # Додаємо години та періоди якщо їх немає
            if 'hour' not in self.data.columns and 'timestamp' in self.data.columns:
                self.data['hour'] = self.data['timestamp'].dt.hour
                self.data['period'] = self.data['hour'].apply(self.get_period)
            
            print(f"Дані завантажено: {self.data.shape}")
            print(f"Колонки: {list(self.data.columns)}")
            return True
        except Exception as e:
            print(f"Помилка завантаження: {e}")
            return False
    
    def basic_statistics(self):
        """Базова статистика"""
        print("\n=== БАЗОВА СТАТИСТИКА ===")
        
        stats = self.data['light_level'].describe()
        print(f"\nЗагальна статистика освітленості:")
        print(f"Середнє значення: {stats['mean']:.2f} lux")
        print(f"Мінімум: {stats['min']:.2f} lux")
        print(f"Максимум: {stats['max']:.2f} lux")
        print(f"Стандартне відхилення: {stats['std']:.2f} lux")
        
        # Статистика по періодах
        if 'period' in self.data.columns:
            print(f"\nСтатистика по періодах дня:")
            period_stats = self.data.groupby('period')['light_level'].agg(['mean', 'min', 'max', 'std'])
            print(period_stats.round(2))
    
    def visualize_data(self):
        """Створення графіків"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Зміна освітленості протягом дня
        axes[0, 0].plot(self.data['hour'], self.data['light_level'], 'o', alpha=0.6, markersize=2)
        axes[0, 0].set_title('Рівень освітленості протягом доби')
        axes[0, 0].set_xlabel('Година дня')
        axes[0, 0].set_ylabel('Освітленість (lux)')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_xlim(0, 23)
        
        # Додаємо середні значення по годинах
        hourly_mean = self.data.groupby('hour')['light_level'].mean()
        axes[0, 0].plot(hourly_mean.index, hourly_mean.values, 'r-', linewidth=3, label='Середнє')
        axes[0, 0].legend()
        
        # 2. Розподіл освітленості
        axes[0, 1].hist(self.data['light_level'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 1].set_title('Розподіл рівня освітленості')
        axes[0, 1].set_xlabel('Освітленість (lux)')
        axes[0, 1].set_ylabel('Частота')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Додаємо вертикальні лінії для середнього та медіани
        mean_light = self.data['light_level'].mean()
        median_light = self.data['light_level'].median()
        axes[0, 1].axvline(mean_light, color='red', linestyle='--', label=f'Середнє: {mean_light:.1f}')
        axes[0, 1].axvline(median_light, color='green', linestyle='--', label=f'Медіана: {median_light:.1f}')
        axes[0, 1].legend()
        
        # 3. Boxplot по періодах дня
        if 'period' in self.data.columns:
            period_order = ['Ніч', 'Ранок', 'День', 'Вечір']
            period_data = [self.data[self.data['period'] == period]['light_level'].values 
                          for period in period_order if period in self.data['period'].values]
            period_labels = [period for period in period_order if period in self.data['period'].values]
            
            box_plot = axes[1, 0].boxplot(period_data, labels=period_labels, patch_artist=True)
            axes[1, 0].set_title('Розподіл освітленості по періодах дня')
            axes[1, 0].set_ylabel('Освітленість (lux)')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Розфарбовуємо boxplot
            colors = ['darkblue', 'orange', 'gold', 'purple']
            for patch, color in zip(box_plot['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
        
        # 4. Часовий ряд (якщо є timestamp)
        if 'timestamp' in self.data.columns:
            # Беремо кожен 10-й запис для кращої читабельності
            sample_data = self.data[::10]
            axes[1, 1].plot(sample_data['timestamp'], sample_data['light_level'], '-', alpha=0.8)
            axes[1, 1].set_title('Часовий ряд освітленості')
            axes[1, 1].set_xlabel('Час')
            axes[1, 1].set_ylabel('Освітленість (lux)')
            axes[1, 1].grid(True, alpha=0.3)
            
            # Форматування осі часу
            axes[1, 1].tick_params(axis='x', rotation=45)
        else:
            # Альтернативний графік - середні значення по годинах
            hourly_stats = self.data.groupby('hour')['light_level'].agg(['mean', 'std'])
            axes[1, 1].errorbar(hourly_stats.index, hourly_stats['mean'], 
                               yerr=hourly_stats['std'], fmt='o-', capsize=5)
            axes[1, 1].set_title('Середня освітленість по годинах')
            axes[1, 1].set_xlabel('Година')
            axes[1, 1].set_ylabel('Освітленість (lux)')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def analyze_patterns(self):
        """Аналіз патернів освітленості"""
        print("\n=== АНАЛІЗ ПАТЕРНІВ ===")
        
        # Найяскравіші та найтемніші моменти
        max_light = self.data.loc[self.data['light_level'].idxmax()]
        min_light = self.data.loc[self.data['light_level'].idxmin()]
        
        print(f"Найвищий рівень освітленості:")
        print(f"  Значення: {max_light['light_level']:.2f} lux")
        if 'hour' in max_light:
            print(f"  Час: {max_light['hour']}:00")
        if 'period' in max_light:
            print(f"  Період: {max_light['period']}")
        
        print(f"\nНайнижчий рівень освітленості:")
        print(f"  Значення: {min_light['light_level']:.2f} lux")
        if 'hour' in min_light:
            print(f"  Час: {min_light['hour']}:00")
        if 'period' in min_light:
            print(f"  Період: {min_light['period']}")
        
        # Аналіз періодів
        if 'period' in self.data.columns:
            print(f"\nСередня освітленість по періодах:")
            for period in ['Ніч', 'Ранок', 'День', 'Вечір']:
                if period in self.data['period'].values:
                    avg_light = self.data[self.data['period'] == period]['light_level'].mean()
                    print(f"  {period}: {avg_light:.2f} lux")
        
        # Класифікація рівнів освітленості
        def classify_light(lux):
            if lux < 50:
                return 'Дуже темно'
            elif lux < 150:
                return 'Темно'
            elif lux < 300:
                return 'Помірно'
            elif lux < 500:
                return 'Світло'
            else:
                return 'Дуже світло'
        
        self.data['light_category'] = self.data['light_level'].apply(classify_light)
        
        print(f"\nРозподіл за категоріями освітленості:")
        category_counts = self.data['light_category'].value_counts()
        for category, count in category_counts.items():
            percentage = (count / len(self.data)) * 100
            print(f"  {category}: {count} записів ({percentage:.1f}%)")
    
    def create_recommendations(self):
        """Створення рекомендацій"""
        print("\n=== РЕКОМЕНДАЦІЇ ===")
        
        avg_light = self.data['light_level'].mean()
        min_light = self.data['light_level'].min()
        max_light = self.data['light_level'].max()
        
        print("На основі аналізу освітленості:")
        
        if avg_light < 100:
            print("• Середній рівень освітленості низький - рекомендується покращити освітлення")
        elif avg_light > 500:
            print("• Високий рівень освітленості - можливо варто розглянути затемнення")
        else:
            print("• Середній рівень освітленості в нормі")
        
        if min_light > 0:
            print(f"• Мінімальна освітленість: {min_light:.1f} lux - достатньо для орієнтації")
        else:
            print("• Є періоди повної темряви")
        
        if max_light > 1000:
            print(f"• Максимальна освітленість висока ({max_light:.1f} lux) - може засліплювати")
        
        # Рекомендації по періодах
        if 'period' in self.data.columns:
            day_light = self.data[self.data['period'] == 'День']['light_level'].mean()
            night_light = self.data[self.data['period'] == 'Ніч']['light_level'].mean()
            
            if day_light < 300:
                print("• Денне освітлення низьке - рекомендується використовувати природне світло")
            
            if night_light > 100:
                print("• Нічне освітлення високе - може заважати сну")
    
    def export_results(self, filename="light_analysis_results.csv"):
        """Експорт результатів"""
        if self.data is not None:
            # Додаємо категорії освітленості якщо їх немає
            if 'light_category' not in self.data.columns:
                def classify_light(lux):
                    if lux < 50: return 'Дуже темно'
                    elif lux < 150: return 'Темно'
                    elif lux < 300: return 'Помірно'
                    elif lux < 500: return 'Світло'
                    else: return 'Дуже світло'
                
                self.data['light_category'] = self.data['light_level'].apply(classify_light)
            
            self.data.to_csv(filename, index=False)
            print(f"\nРезультати збережено в файл: {filename}")
    
    def run_analysis(self, file_path=None):
        """Запуск повного аналізу"""
        print("=== АНАЛІЗ РІВНЯ ОСВІТЛЕНОСТІ ===\n")
        
        # Завантаження або генерація даних
        if file_path and self.load_data(file_path):
            print("Дані завантажено з файлу")
        else:
            print("Використовуємо згенеровані дані для демонстрації")
            self.generate_sample_data()
        
        # Базова статистика
        self.basic_statistics()
        
        # Візуалізація
        self.visualize_data()
        
        # Аналіз патернів
        self.analyze_patterns()
        
        # Рекомендації
        self.create_recommendations()
        
        # Експорт результатів
        self.export_results()
        
        print("\n=== ВИСНОВКИ ===")
        print("Аналіз освітленості завершено!")
        print("Основні результати:")
        avg_light = self.data['light_level'].mean()
        print(f"• Середня освітленість: {avg_light:.2f} lux")
        print(f"• Діапазон: {self.data['light_level'].min():.1f} - {self.data['light_level'].max():.1f} lux")
        
        if 'period' in self.data.columns:
            best_period = self.data.groupby('period')['light_level'].mean().idxmax()
            print(f"• Найкращий період для освітлення: {best_period}")

# Інструкції по збору реальних даних
def print_instructions():
    """Інструкції по збору даних"""
    print("\n" + "="*60)
    print("📱 ІНСТРУКЦІЇ ПО ЗБОРУ ДАНИХ З ДАТЧИКА ОСВІТЛЕНОСТІ")
    print("="*60)
    print("""
1. ВСТАНОВІТЬ ДОДАТОК:
   • Phyphox (безкоштовний, Android/iOS)
   • Sensor Kinetics (Android/iOS)
   • Arduino Science Journal (Android)

2. НАЛАШТУВАННЯ:
   • Знайдіть розділ "Light" або "Illuminance"
   • Встановіть частоту 1-10 вимірювань на секунду
   • Підготуйте телефон (очистіть датчик)

3. ЗБІР ДАНИХ (протягом доби):
   • Ранок (6:00-12:00): біля вікна, з природним світлом
   • День (12:00-18:00): різні кімнати, різне освітлення
   • Вечір (18:00-22:00): штучне освітлення
   • Ніч (22:00-6:00): мінімальне освітлення

4. ЕКСПОРТ:
   • Збережіть як CSV файл
   • Назвіть файл, наприклад: "light_data.csv"

5. ФОРМАТ CSV:
   timestamp,light_level
   2024-06-01 08:00:00,245.5
   2024-06-01 08:01:00,251.2
   ...

6. ВИКОРИСТАННЯ:
   analyzer = LightAnalyzer()
   analyzer.run_analysis("light_data.csv")
""")
    print("="*60)

# Запуск програми
if __name__ == "__main__":
    print_instructions()
    
    analyzer = LightAnalyzer()
    
    # Для використання з реальними даними розкоментуйте:
    # analyzer.run_analysis("your_light_data.csv")
    
    # Для демонстрації:
    analyzer.run_analysis()