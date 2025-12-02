import platform
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from IPython.display import display
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_samples,
    silhouette_score,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ЭТАП 1: ПОСТАНОВКА ЗАДАЧИ И ДАННЫЕ

PALETTE = "Set2"
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 12

RANDOM_STATE = 42


print("\nЭТАП 1: ПОСТАНОВКА ЗАДАЧИ И ДАННЫЕ\n")

try:
    url = "https://drive.google.com/uc?id=1dvVgFSH22J7okTKYD8sHzJvJJ9MRZzkN&export=download"
    df = pd.read_csv(url, delimiter=";")
    print("Файл успешно загружен")
except Exception as e:
    print(f"Ошибка загрузки файла: {e}")
    exit(1)

COLUMN_TRANSLATOR = {
    "age": "Возраст",
    "job": "Профессия",
    "marital": "Семейное положение",
    "education": "Образование",
    "default": "Кредитный дефолт",
    "housing": "Ипотека",
    "loan": "Потребительский кредит",
    "contact": "Тип контакта",
    "month": "Месяц",
    "day_of_week": "День недели",
    "duration": "Длительность звонка (сек)",
    "campaign": "Кол-во контактов в кампании",
    "pdays": "Дней с прошлого контакта",
    "previous": "Кол-во прошлых контактов",
    "poutcome": "Результат прошлой кампании",
    "emp.var.rate": "Изм. уровня занятости",
    "cons.price.idx": "Индекс потреб. цен",
    "cons.conf.idx": "Индекс потреб. доверия",
    "euribor3m": "Ставка Euribor 3M",
    "nr.employed": "Число занятых",
    "y": "Согласие на вклад",
}

print("\nОПИСАНИЕ ДАТАСЕТА:")
print(f"Размерность: {df.shape[0]} строк × {df.shape[1]} столбцов")

print("\nПервые 5 строк датасета:")
display(df.head())

print("\nИнформация о типах данных:")
df.info()

# Удаление дубликатов
df.drop_duplicates(inplace=True)
print(f"Дубликаты удалены. Размерность: {df.shape}")

# Обработка 'unknown' значений
object_cols = df.select_dtypes(include=["object"]).columns
for col in object_cols:
    if "unknown" in df[col].unique():
        mode_value = df[col].mode()[0]
        if mode_value == "unknown":
            second_mode = df[col].value_counts().index[1]
            df[col] = df[col].replace("unknown", second_mode)
        else:
            df[col] = df[col].replace("unknown", mode_value)
print("Значения 'unknown' заменены на моду")

# будем использовать только релевантные признаки клиента
clustering_features = [
    "age",
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "campaign",
    "pdays",
    "previous",
    "poutcome",
]

df_cluster = df[clustering_features].copy()
print(f"\nВыбраны признаки для кластеризации: {len(clustering_features)} шт.")
print(f"  {clustering_features}")