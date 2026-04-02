import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns


def filter_iqr(data, column, multiplier=1.5):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    cleaned_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return cleaned_data

df = pd.read_csv('laptop_price.csv', encoding='latin-1', skiprows=range(1, 300), nrows=100)
df_clean = filter_iqr(df, 'Price_euros')

# гистограмма
plt.figure(figsize=(8, 5))
plt.hist(df_clean['Price_euros'], bins=30, edgecolor='black', alpha=0.7, color='skyblue')
plt.xlabel('Цена (евро)')
plt.ylabel('Частота')
plt.title('Распределение цен на ноутбуки')
plt.grid(True, alpha=0.3)
plt.show()


# цдф
sorted_data = np.sort(df_clean['Price_euros'])
cdf = np.arange(1, len(sorted_data)+1) / len(sorted_data)

plt.figure(figsize=(8, 5))
plt.step(sorted_data, cdf, where='post', color='red', linewidth=2, label='CDF')
plt.xlabel('Цена (евро)')
plt.ylabel('Кумулятивная вероятность')
plt.title('Кумулятивная функция распределения (CDF)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()


# усы
plt.figure(figsize=(6,4))
sns.boxplot(x=df_clean['Price_euros'], color='lightgreen')
plt.xlabel('Цена (евро)')
plt.title('Boxplot цен на ноутбуки')
plt.grid(True, alpha=0.3, axis='x')
plt.show()


mean_val = df_clean['Price_euros'].mean()
median_val = df_clean['Price_euros'].median()
std_dev = df_clean['Price_euros'].std()
skewness = df_clean['Price_euros'].skew()
kurtosis = df_clean['Price_euros'].kurt()

print(f'Среднее: {mean_val:.2f}')
print(f'Медиана: {median_val:.2f}')
print(f'Стандартное отклонение: {std_dev:.2f}')
print(f'Коэффициент асимметрии: {skewness:.2f}')
print(f'Эксцесс: {kurtosis:.2f}')



# тест шапиро
shapiro_stat, shapiro_p = stats.shapiro(df_clean['Price_euros'])
print(f'Тест Шапиро-Уилка: статистика = {shapiro_stat:.4f}, p-value = {shapiro_p:.4f}')


if shapiro_p > 0.05:
    print('Данные в норме')
else:
    print('Данные не в норме')

    # тест колмогорова
    shape, loc, scale = stats.lognorm.fit(df_clean['Price_euros'], floc=0)
    ks_stat, ks_p = stats.kstest(df_clean['Price_euros'], 'lognorm', args=(shape, loc, scale))
    print(f'статистика = {ks_stat:.4f}, p-value = {ks_p:.4f}')

    a, loc, scale = stats.gamma.fit(df_clean['Price_euros'])
    ks_stat_gamma, ks_p_gamma = stats.kstest(df_clean['Price_euros'], 'gamma', args=(a, loc, scale))
    print(f'статистика = {ks_stat_gamma:.4f}, p-value = {ks_p_gamma:.4f}')

