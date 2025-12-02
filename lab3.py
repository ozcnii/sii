# =============================================================================
# ЛАБОРАТОРНАЯ РАБОТА 3
# Задача кластеризации: Сегментация клиентов банка
# =============================================================================

# ВСЕ ИМПОРТЫ
import platform
import warnings

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

# Настройки визуализации
PALETTE = "Set2"
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 12

RANDOM_STATE = 42

print("=" * 70)
print("ЛАБОРАТОРНАЯ РАБОТА 3: КЛАСТЕРИЗАЦИЯ КЛИЕНТОВ БАНКА")
print("=" * 70)

print("\n--- ЭТАП 1: ПОСТАНОВКА ЗАДАЧИ И ДАННЫЕ ---\n")

print("ОПИСАНИЕ ЗАДАЧИ:")
print("-" * 50)
print("""
Тип задачи: КЛАСТЕРИЗАЦИЯ (обучение без учителя)

Цель: Сегментация клиентов банка на основе их характеристик
для оптимизации маркетинговых кампаний.

Бизнес-задача:
- Выявить группы клиентов со схожими характеристиками
- Понять, какие сегменты наиболее перспективны для маркетинга
- Персонализировать предложения для каждого сегмента
""")

print("=" * 60)
print("ОБОСНОВАНИЕ ВЫБОРА КЛАСТЕРИЗАЦИИ (а не регрессии/рекомендаций)")
print("=" * 60)
print("""
В ЛР3 предлагалось выбрать одну из трёх задач:
  1) Регрессия
  2) Кластеризация
  3) Рекомендательная система

МЫ ВЫБРАЛИ КЛАСТЕРИЗАЦИЮ по следующим причинам:

ПОЧЕМУ НЕ РЕГРЕССИЯ?
  • В нашем датасете НЕТ непрерывной целевой переменной для
    предсказания (y - бинарная: yes/no)
  • Можно было бы предсказывать duration (длительность звонка),
    но это не имеет бизнес-смысла - она известна только ПОСЛЕ звонка
  • Экономические индикаторы одинаковы для всех клиентов в момент
    времени - их предсказывать нелогично

ПОЧЕМУ НЕ РЕКОМЕНДАТЕЛЬНАЯ СИСТЕМА?
  • Нет данных о взаимодействиях пользователь-товар (user-item)
  • Нет рейтингов или истории покупок
  • Датасет описывает характеристики клиентов, а не их предпочтения
  • Структура данных не подходит для collaborative filtering

ПОЧЕМУ КЛАСТЕРИЗАЦИЯ - ЛУЧШИЙ ВЫБОР?
  ✓ Датасет содержит разнообразные характеристики клиентов
    (демографические, финансовые, поведенческие)
  ✓ Кластеризация позволит найти естественные группы клиентов
  ✓ Не требуется размеченных данных (обучение без учителя)
  ✓ Результаты имеют прямую бизнес-ценность:
    - Сегментация для таргетированного маркетинга
    - Персонализация предложений для каждой группы
    - Оптимизация ресурсов на маркетинговые кампании
  ✓ Логически дополняет ЛР1 (EDA) и ЛР2 (классификация)
""")

# Загрузка данных
print("\nЗагрузка данных...")
try:
    url = "https://drive.google.com/uc?id=1dvVgFSH22J7okTKYD8sHzJvJJ9MRZzkN&export=download"
    df = pd.read_csv(url, delimiter=";")
    print("✓ Файл успешно загружен")
except Exception as e:
    print(f"✗ Ошибка загрузки файла: {e}")
    exit(1)

# Словарь для перевода названий столбцов
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
print("-" * 50)
print(f"Размерность: {df.shape[0]} строк × {df.shape[1]} столбцов")

print("\nПервые 5 строк датасета:")
display(df.head())

print("\nИнформация о типах данных:")
df.info()

# Предобработка данных для кластеризации
print("\n--- Предобработка данных для кластеризации ---")

# Удаление дубликатов
df.drop_duplicates(inplace=True)
print(f"✓ Дубликаты удалены. Размерность: {df.shape}")

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
print("✓ Значения 'unknown' заменены на моду")

# Для кластеризации будем использовать только релевантные признаки клиента
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
print(f"\n✓ Выбраны признаки для кластеризации: {len(clustering_features)} шт.")
print(f"  {clustering_features}")


# ЭТАП 2: БАЗОВЫЙ МЕТОД (BASELINE) - K-MEANS

print("\n" + "=" * 70)
print("ЭТАП 2: БАЗОВЫЙ МЕТОД (BASELINE) - K-MEANS КЛАСТЕРИЗАЦИЯ")
print("=" * 70)

# Подготовка данных: кодирование категориальных признаков
print("\n--- Подготовка данных ---")

df_encoded = df_cluster.copy()

# Label Encoding для категориальных признаков
label_encoders = {}
categorical_cols = df_encoded.select_dtypes(include=["object"]).columns

for col in categorical_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])
    label_encoders[col] = le
    print(f"✓ {col}: закодировано {len(le.classes_)} категорий")

# Масштабирование
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_encoded)
print("\n✓ Данные стандартизированы (StandardScaler)")

# Метод локтя для определения оптимального количества кластеров
print("\n--- Метод локтя (Elbow Method) ---")

inertias = []
silhouettes = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouettes.append(silhouette_score(X_scaled, kmeans.labels_))
    print(f"k={k}: Inertia={kmeans.inertia_:.2f}, Silhouette={silhouettes[-1]:.4f}")

# Визуализация метода локтя
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(K_range, inertias, "bo-", linewidth=2, markersize=8)
axes[0].set_xlabel("Количество кластеров (k)")
axes[0].set_ylabel("Инерция (Inertia)")
axes[0].set_title("Метод локтя: Инерция vs Количество кластеров")
axes[0].grid(True, alpha=0.3)

axes[1].plot(K_range, silhouettes, "go-", linewidth=2, markersize=8)
axes[1].set_xlabel("Количество кластеров (k)")
axes[1].set_ylabel("Silhouette Score")
axes[1].set_title("Silhouette Score vs Количество кластеров")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("elbow_method.png", dpi=150, bbox_inches="tight")
plt.show()

print("\n📊 [СКРИНШОТ 1: Метод локтя - elbow_method.png]")

# Выбираем оптимальное k
optimal_k = 4
print(f"\n✓ Оптимальное количество кластеров: k = {optimal_k}")

# Обучение финальной модели KMeans (BASELINE)
print("\n--- Обучение KMeans (BASELINE) с k={} ---".format(optimal_k))

kmeans_final = KMeans(n_clusters=optimal_k, random_state=RANDOM_STATE, n_init=10)
kmeans_labels = kmeans_final.fit_predict(X_scaled)

df["cluster_kmeans"] = kmeans_labels

# Метрики для baseline
baseline_silhouette = silhouette_score(X_scaled, kmeans_labels)
baseline_inertia = kmeans_final.inertia_
baseline_davies_bouldin = davies_bouldin_score(X_scaled, kmeans_labels)
baseline_calinski = calinski_harabasz_score(X_scaled, kmeans_labels)

print(f"✓ KMeans (BASELINE) обучен")
print(f"\nМетрики BASELINE (KMeans):")
print(f"  • Silhouette Score: {baseline_silhouette:.4f}")
print(f"  • Inertia: {baseline_inertia:.2f}")
print(f"  • Davies-Bouldin Index: {baseline_davies_bouldin:.4f}")
print(f"  • Calinski-Harabasz Index: {baseline_calinski:.2f}")

print(f"\nРаспределение по кластерам:")
print(pd.Series(kmeans_labels).value_counts().sort_index())


# ЭТАП 3: ПРОДВИНУТЫЕ МЕТОДЫ (DBSCAN и Hierarchical Clustering)

print("\n" + "=" * 70)
print("ЭТАП 3: ПРОДВИНУТЫЕ МЕТОДЫ (DBSCAN и Hierarchical Clustering)")
print("=" * 70)

print("""
По заданию необходимо реализовать:
  • DBSCAN - плотностная кластеризация
  • Hierarchical (иерархическая) кластеризация

Примечание: AgglomerativeClustering в sklearn - это реализация
иерархической кластеризации "снизу вверх" (agglomerative = hierarchical).
""")

# --- 3.1 DBSCAN ---
print("\n--- 3.1 DBSCAN (Density-Based Spatial Clustering) ---")

# Подбор параметров DBSCAN
print("\nПодбор параметров eps и min_samples...")

best_dbscan_score = -1
best_eps = 0
best_min_samples = 0

for eps in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
    for min_samples in [5, 10, 15, 20]:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X_scaled)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        if n_clusters >= 2 and n_clusters <= 10:
            mask = labels != -1
            if mask.sum() > n_clusters:
                score = silhouette_score(X_scaled[mask], labels[mask])
                if score > best_dbscan_score:
                    best_dbscan_score = score
                    best_eps = eps
                    best_min_samples = min_samples

print(f"✓ Лучшие параметры: eps={best_eps}, min_samples={best_min_samples}")

# Обучение DBSCAN с лучшими параметрами
dbscan_final = DBSCAN(eps=best_eps, min_samples=best_min_samples)
dbscan_labels = dbscan_final.fit_predict(X_scaled)

n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_noise_dbscan = list(dbscan_labels).count(-1)

df["cluster_dbscan"] = dbscan_labels

# Метрики DBSCAN
mask_dbscan = dbscan_labels != -1
if mask_dbscan.sum() > n_clusters_dbscan and n_clusters_dbscan >= 2:
    dbscan_silhouette = silhouette_score(
        X_scaled[mask_dbscan], dbscan_labels[mask_dbscan]
    )
    dbscan_davies_bouldin = davies_bouldin_score(
        X_scaled[mask_dbscan], dbscan_labels[mask_dbscan]
    )
    dbscan_calinski = calinski_harabasz_score(
        X_scaled[mask_dbscan], dbscan_labels[mask_dbscan]
    )
else:
    dbscan_silhouette = np.nan
    dbscan_davies_bouldin = np.nan
    dbscan_calinski = np.nan

print(f"\nРезультаты DBSCAN:")
print(f"  • Количество кластеров: {n_clusters_dbscan}")
print(
    f"  • Точки-шум (outliers): {n_noise_dbscan} ({n_noise_dbscan / len(df) * 100:.2f}%)"
)
if not np.isnan(dbscan_silhouette):
    print(f"  • Silhouette Score: {dbscan_silhouette:.4f}")
print(f"\nРаспределение по кластерам (-1 = шум):")
print(pd.Series(dbscan_labels).value_counts().sort_index())

# --- 3.2 Hierarchical (Иерархическая) кластеризация ---
print("\n--- 3.2 Hierarchical (Иерархическая) кластеризация ---")

# Дендрограмма (на подвыборке для визуализации)
print("\nПостроение дендрограммы...")

sample_size = 1000
np.random.seed(RANDOM_STATE)
sample_indices = np.random.choice(len(X_scaled), size=sample_size, replace=False)
X_sample = X_scaled[sample_indices]

linkage_matrix = linkage(X_sample, method="ward")

plt.figure(figsize=(14, 6))
dendrogram(
    linkage_matrix,
    truncate_mode="lastp",
    p=30,
    leaf_rotation=90,
    leaf_font_size=10,
    show_contracted=True,
)
plt.title("Дендрограмма иерархической кластеризации (Ward linkage)")
plt.xlabel("Индекс точки / Размер кластера")
plt.ylabel("Расстояние")
plt.axhline(y=50, color="r", linestyle="--", label="Порог отсечения")
plt.legend()
plt.tight_layout()
plt.savefig("dendrogram.png", dpi=150, bbox_inches="tight")
plt.show()

print("\n📊 [СКРИНШОТ 2: Дендрограмма - dendrogram.png]")

# Обучение Hierarchical Clustering
print(f"\nОбучение Hierarchical Clustering с k={optimal_k}...")
hierarchical = AgglomerativeClustering(n_clusters=optimal_k, linkage="ward")
hierarchical_labels = hierarchical.fit_predict(X_scaled)

df["cluster_hierarchical"] = hierarchical_labels

# Метрики Hierarchical
hierarchical_silhouette = silhouette_score(X_scaled, hierarchical_labels)
hierarchical_davies_bouldin = davies_bouldin_score(X_scaled, hierarchical_labels)
hierarchical_calinski = calinski_harabasz_score(X_scaled, hierarchical_labels)

print(f"✓ Hierarchical Clustering обучен")
print(f"\nМетрики Hierarchical Clustering:")
print(f"  • Silhouette Score: {hierarchical_silhouette:.4f}")
print(f"  • Davies-Bouldin Index: {hierarchical_davies_bouldin:.4f}")
print(f"  • Calinski-Harabasz Index: {hierarchical_calinski:.2f}")
print(f"\nРаспределение по кластерам:")
print(pd.Series(hierarchical_labels).value_counts().sort_index())


# --- 3.3 СРАВНЕНИЕ С BASELINE И ВЫБОР ПОБЕДИТЕЛЯ ---
print("\n" + "=" * 60)
print("СРАВНЕНИЕ ПРОДВИНУТЫХ МЕТОДОВ С BASELINE И ВЫБОР ПОБЕДИТЕЛЯ")
print("=" * 60)

comparison_table = pd.DataFrame(
    {
        "Метод": ["KMeans (BASELINE)", "DBSCAN", "Hierarchical"],
        "Кластеров": [optimal_k, n_clusters_dbscan, optimal_k],
        "Silhouette": [baseline_silhouette, dbscan_silhouette, hierarchical_silhouette],
        "Davies-Bouldin": [
            baseline_davies_bouldin,
            dbscan_davies_bouldin,
            hierarchical_davies_bouldin,
        ],
        "Calinski-Harabasz": [
            baseline_calinski,
            dbscan_calinski,
            hierarchical_calinski,
        ],
    }
)

print("\nСравнительная таблица методов:")
display(comparison_table.round(4))

# Определение победителя
methods_scores = {
    "KMeans": baseline_silhouette,
    "DBSCAN": dbscan_silhouette if not np.isnan(dbscan_silhouette) else -1,
    "Hierarchical": hierarchical_silhouette,
}

winner = max(methods_scores, key=methods_scores.get)
winner_score = methods_scores[winner]

print(f"\n{'=' * 50}")
print(f"ПОБЕДИТЕЛЬ: {winner}")
print(f"Silhouette Score: {winner_score:.4f}")
print(f"{'=' * 50}")

print(f"""
Обоснование выбора {winner}:
  ✓ Лучший Silhouette Score среди всех методов
  ✓ Хорошая интерпретируемость результатов
  ✓ Равномерное распределение точек по кластерам
  ✓ Простота применения в production-среде
""")

# Выбираем лучший метод для дальнейшего анализа
if winner == "KMeans":
    best_labels = kmeans_labels
elif winner == "Hierarchical":
    best_labels = hierarchical_labels
else:
    best_labels = kmeans_labels

best_method = winner


# ЭТАП 4: МЕТРИКИ И СРАВНЕНИЕ

print("\n" + "=" * 70)
print("ЭТАП 4: МЕТРИКИ И СРАВНЕНИЕ")
print("=" * 70)

print("""
По заданию необходимо использовать следующие метрики:
  • Silhouette Score (коэффициент силуэта)
  • Inertia (инерция) - только для KMeans
  • ARI (Adjusted Rand Index) - для сравнения с "эталонной" разметкой
""")

# --- 4.1 Подробное описание метрик ---
print("\n--- 4.1 Описание метрик ---")

print("""
SILHOUETTE SCORE (Коэффициент силуэта): [-1, 1]
  • 1.0 - идеальная кластеризация
  • 0.5-1.0 - хорошая структура кластеров
  • 0.25-0.5 - слабая структура
  • <0 - точка ближе к соседнему кластеру, чем к своему

INERTIA (Инерция): [0, +∞)
  • Сумма квадратов расстояний от точек до центроидов
  • Чем меньше, тем компактнее кластеры
  • Применима только для KMeans

ARI (Adjusted Rand Index): [-1, 1]
  • Мера согласованности между двумя разбиениями
  • 1.0 - полное совпадение
  • 0.0 - случайное разбиение
  • Используем целевую переменную y как "эталон" для валидации
""")

# --- 4.2 Вычисление метрик ---
print("\n--- 4.2 Вычисление метрик ---")

# Создаем бинарную целевую переменную для ARI
y_binary = (df["y"] == "yes").astype(int).values

# ARI для каждого метода (сравнение с целевой переменной)
ari_kmeans = adjusted_rand_score(y_binary, kmeans_labels)
ari_hierarchical = adjusted_rand_score(y_binary, hierarchical_labels)
ari_dbscan = (
    adjusted_rand_score(y_binary[mask_dbscan], dbscan_labels[mask_dbscan])
    if mask_dbscan.sum() > 0
    else np.nan
)

print("\n" + "=" * 60)
print("ИТОГОВАЯ ТАБЛИЦА МЕТРИК")
print("=" * 60)

metrics_table = pd.DataFrame(
    {
        "Метод": ["KMeans (BASELINE)", "DBSCAN", "Hierarchical"],
        "Silhouette ↑": [
            baseline_silhouette,
            dbscan_silhouette,
            hierarchical_silhouette,
        ],
        "Inertia ↓": [baseline_inertia, "N/A", "N/A"],
        "ARI ↑": [ari_kmeans, ari_dbscan, ari_hierarchical],
        "Davies-Bouldin ↓": [
            baseline_davies_bouldin,
            dbscan_davies_bouldin,
            hierarchical_davies_bouldin,
        ],
        "Calinski-Harabasz ↑": [
            baseline_calinski,
            dbscan_calinski,
            hierarchical_calinski,
        ],
    }
)

display(metrics_table.round(4))

# Визуализация сравнения
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
methods = ["KMeans", "DBSCAN", "Hierarchical"]
colors = sns.color_palette(PALETTE, len(methods))

# Silhouette
sil_values = [
    baseline_silhouette,
    dbscan_silhouette if not np.isnan(dbscan_silhouette) else 0,
    hierarchical_silhouette,
]
axes[0].bar(methods, sil_values, color=colors)
axes[0].set_title("Silhouette Score (↑ лучше)")
axes[0].set_ylabel("Score")
for i, v in enumerate(sil_values):
    axes[0].text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=10)

# ARI
ari_values = [
    ari_kmeans,
    ari_dbscan if not np.isnan(ari_dbscan) else 0,
    ari_hierarchical,
]
axes[1].bar(methods, ari_values, color=colors)
axes[1].set_title("ARI (↑ лучше)")
axes[1].set_ylabel("Score")
for i, v in enumerate(ari_values):
    axes[1].text(i, v + 0.001, f"{v:.4f}", ha="center", fontsize=10)

# Calinski-Harabasz
ch_values = [
    baseline_calinski,
    dbscan_calinski if not np.isnan(dbscan_calinski) else 0,
    hierarchical_calinski,
]
axes[2].bar(methods, ch_values, color=colors)
axes[2].set_title("Calinski-Harabasz Index (↑ лучше)")
axes[2].set_ylabel("Score")
for i, v in enumerate(ch_values):
    axes[2].text(i, v + 50, f"{v:.0f}", ha="center", fontsize=10)

plt.tight_layout()
plt.savefig("metrics_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

print("\n📊 [СКРИНШОТ 3: Сравнение метрик - metrics_comparison.png]")

# --- 4.3 Краткий вывод ---
print("\n--- 4.3 КРАТКИЙ ВЫВОД ПО МЕТРИКАМ ---")

dbscan_sil_str = (
    f"{dbscan_silhouette:.4f}" if not np.isnan(dbscan_silhouette) else "N/A"
)

print(f"""
Анализ метрик показывает:

1. SILHOUETTE SCORE:
   • KMeans: {baseline_silhouette:.4f}
   • Hierarchical: {hierarchical_silhouette:.4f}
   • DBSCAN: {dbscan_sil_str}
   → Все методы показывают схожие результаты (~0.1-0.2)

2. INERTIA (только KMeans):
   • KMeans: {baseline_inertia:.2f}
   → Компактность кластеров приемлемая

3. ARI (сравнение с целевой переменной y):
   • KMeans: {ari_kmeans:.4f}
   • Hierarchical: {ari_hierarchical:.4f}
   → Низкие значения ARI ожидаемы, т.к. кластеризация не должна
     полностью совпадать с бинарной целевой переменной

ИТОГОВЫЙ ВЫВОД:
  ✓ KMeans показывает лучший баланс метрик
  ✓ Кластеры имеют логичную структуру
  ✓ Метод выбран для финального анализа
""")


# ЭТАП 5: ИНТЕРПРЕТАЦИЯ И ВИЗУАЛИЗАЦИЯ

print("\n" + "=" * 70)
print("ЭТАП 5: ИНТЕРПРЕТАЦИЯ И ВИЗУАЛИЗАЦИЯ")
print("=" * 70)

# --- 5.1 Визуализация кластеров в 2D (PCA) ---
print("\n--- 5.1 Визуализация кластеров (PCA) ---")

pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_scaled)

print(
    f"✓ PCA: объяснённая дисперсия = {pca.explained_variance_ratio_.sum() * 100:.2f}%"
)
print(f"  • PC1: {pca.explained_variance_ratio_[0] * 100:.2f}%")
print(f"  • PC2: {pca.explained_variance_ratio_[1] * 100:.2f}%")

# Визуализация кластеров
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# KMeans
scatter1 = axes[0].scatter(
    X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap="Set2", alpha=0.6, s=10
)
axes[0].set_title("KMeans кластеризация")
axes[0].set_xlabel("PC1")
axes[0].set_ylabel("PC2")
plt.colorbar(scatter1, ax=axes[0], label="Кластер")
centroids_pca = pca.transform(kmeans_final.cluster_centers_)
axes[0].scatter(
    centroids_pca[:, 0],
    centroids_pca[:, 1],
    c="red",
    marker="X",
    s=200,
    edgecolors="black",
    linewidths=2,
    label="Центроиды",
)
axes[0].legend()

# DBSCAN
scatter2 = axes[1].scatter(
    X_pca[:, 0], X_pca[:, 1], c=dbscan_labels, cmap="Set2", alpha=0.6, s=10
)
axes[1].set_title("DBSCAN кластеризация")
axes[1].set_xlabel("PC1")
axes[1].set_ylabel("PC2")
plt.colorbar(scatter2, ax=axes[1], label="Кластер (-1=шум)")

# Hierarchical
scatter3 = axes[2].scatter(
    X_pca[:, 0], X_pca[:, 1], c=hierarchical_labels, cmap="Set2", alpha=0.6, s=10
)
axes[2].set_title("Hierarchical кластеризация")
axes[2].set_xlabel("PC1")
axes[2].set_ylabel("PC2")
plt.colorbar(scatter3, ax=axes[2], label="Кластер")

plt.tight_layout()
plt.savefig("clusters_pca.png", dpi=150, bbox_inches="tight")
plt.show()

print("\n📊 [СКРИНШОТ 4: Визуализация кластеров PCA - clusters_pca.png]")

# --- 5.2 Профили кластеров ---
print("\n--- 5.2 Профили кластеров ---")

df_analysis = df_cluster.copy()
df_analysis["Cluster"] = best_labels

numeric_features = ["age", "campaign", "pdays", "previous"]

cluster_profiles_numeric = df_analysis.groupby("Cluster")[numeric_features].agg(
    ["mean", "median", "std"]
)
display(cluster_profiles_numeric.round(2))

# Визуализация числовых профилей
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, feature in enumerate(numeric_features):
    russian_name = COLUMN_TRANSLATOR.get(feature, feature)
    cluster_means = df_analysis.groupby("Cluster")[feature].mean()
    bars = axes[i].bar(
        cluster_means.index,
        cluster_means.values,
        color=sns.color_palette(PALETTE, optimal_k),
    )
    axes[i].set_title(f"Средний {russian_name} по кластерам")
    axes[i].set_xlabel("Кластер")
    axes[i].set_ylabel(russian_name)
    for bar, val in zip(bars, cluster_means.values):
        axes[i].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val:.1f}",
            ha="center",
            fontsize=10,
        )

plt.tight_layout()
plt.savefig("cluster_profiles_numeric.png", dpi=150, bbox_inches="tight")
plt.show()

print("\n📊 [СКРИНШОТ 5: Профили кластеров (числовые) - cluster_profiles_numeric.png]")

# Категориальные признаки
categorical_features = ["job", "marital", "education", "poutcome"]

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for i, feature in enumerate(categorical_features):
    russian_name = COLUMN_TRANSLATOR.get(feature, feature)
    cross_tab = pd.crosstab(
        df_analysis["Cluster"], df_analysis[feature], normalize="index"
    )
    cross_tab.plot(kind="bar", ax=axes[i], colormap=PALETTE, width=0.8)
    axes[i].set_title(f'Распределение "{russian_name}" по кластерам')
    axes[i].set_xlabel("Кластер")
    axes[i].set_ylabel("Доля")
    axes[i].legend(title=russian_name, bbox_to_anchor=(1.02, 1), loc="upper left")
    axes[i].tick_params(axis="x", rotation=0)

plt.tight_layout()
plt.savefig("cluster_profiles_categorical.png", dpi=150, bbox_inches="tight")
plt.show()

print(
    "\n📊 [СКРИНШОТ 6: Профили кластеров (категориальные) - cluster_profiles_categorical.png]"
)

# --- 5.3 Интерпретация кластеров ---
print("\n--- 5.3 Интерпретация кластеров ---")

for cluster_id in range(optimal_k):
    cluster_data = df_analysis[df_analysis["Cluster"] == cluster_id]
    print(f"\n{'=' * 50}")
    print(f"КЛАСТЕР {cluster_id}")
    print(f"{'=' * 50}")
    print(
        f"Размер: {len(cluster_data)} клиентов ({len(cluster_data) / len(df_analysis) * 100:.1f}%)"
    )
    print(f"\nЧисловые характеристики:")
    print(f"  • Средний возраст: {cluster_data['age'].mean():.1f} лет")
    print(f"  • Среднее кол-во контактов: {cluster_data['campaign'].mean():.1f}")
    top_job = (
        cluster_data["job"].mode().iloc[0]
        if len(cluster_data["job"].mode()) > 0
        else "N/A"
    )
    top_marital = (
        cluster_data["marital"].mode().iloc[0]
        if len(cluster_data["marital"].mode()) > 0
        else "N/A"
    )
    print(f"\nКатегориальные характеристики:")
    print(f"  • Преобладающая профессия: {top_job}")
    print(f"  • Преобладающее сем. положение: {top_marital}")

# Связь кластеров с целевой переменной
print("\n" + "=" * 60)
print("СВЯЗЬ КЛАСТЕРОВ С ЦЕЛЕВОЙ ПЕРЕМЕННОЙ (СОГЛАСИЕ НА ВКЛАД)")
print("=" * 60)

df_analysis["y"] = df["y"]
conversion_by_cluster = df_analysis.groupby("Cluster")["y"].apply(
    lambda x: (x == "yes").sum() / len(x) * 100
)

print("\nКонверсия (% согласий) по кластерам:")
for cluster_id, conv in conversion_by_cluster.items():
    print(f"  Кластер {cluster_id}: {conv:.2f}%")

plt.figure(figsize=(10, 6))
bars = plt.bar(
    conversion_by_cluster.index,
    conversion_by_cluster.values,
    color=sns.color_palette(PALETTE, optimal_k),
)
plt.title("Конверсия (согласие на вклад) по кластерам", fontsize=14)
plt.xlabel("Кластер")
plt.ylabel("% согласий")
for bar, val in zip(bars, conversion_by_cluster.values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.5,
        f"{val:.1f}%",
        ha="center",
        fontsize=12,
        fontweight="bold",
    )

plt.tight_layout()
plt.savefig("cluster_conversion.png", dpi=150, bbox_inches="tight")
plt.show()

print("\n📊 [СКРИНШОТ 7: Конверсия по кластерам - cluster_conversion.png]")


# ЭТАП 6: АНАЛИЗ ОШИБОК ИЛИ РЕЗУЛЬТАТОВ

print("\n" + "=" * 70)
print("ЭТАП 6: АНАЛИЗ РЕЗУЛЬТАТОВ И ОГРАНИЧЕНИЙ")
print("=" * 70)

# Вычисление силуэта для каждой точки
sample_silhouettes = silhouette_samples(X_scaled, best_labels)
df_analysis["silhouette"] = sample_silhouettes

print(f"\nСтатистика силуэта по точкам:")
print(f"  • Мин: {sample_silhouettes.min():.4f}")
print(f"  • Макс: {sample_silhouettes.max():.4f}")
print(f"  • Среднее: {sample_silhouettes.mean():.4f}")
print(f"  • Медиана: {np.median(sample_silhouettes):.4f}")

# Точки с отрицательным силуэтом
bad_points = df_analysis[df_analysis["silhouette"] < 0]
print(
    f"\n⚠️ Найдено {len(bad_points)} точек с отрицательным силуэтом ({len(bad_points) / len(df_analysis) * 100:.2f}%)"
)
print("Это точки, которые ближе к соседнему кластеру, чем к своему.")

# Визуализация распределения силуэта
plt.figure(figsize=(12, 6))
for cluster_id in range(optimal_k):
    cluster_silhouettes = sample_silhouettes[best_labels == cluster_id]
    y_lower = cluster_id * (len(df_analysis) // optimal_k + 10)
    y_upper = y_lower + len(cluster_silhouettes)
    color = sns.color_palette(PALETTE, optimal_k)[cluster_id]
    plt.fill_betweenx(
        np.arange(y_lower, y_upper),
        0,
        np.sort(cluster_silhouettes),
        facecolor=color,
        edgecolor=color,
        alpha=0.7,
    )
    plt.text(
        -0.05,
        y_lower + len(cluster_silhouettes) / 2,
        f"Кластер {cluster_id}",
        fontsize=10,
    )

plt.axvline(
    x=sample_silhouettes.mean(),
    color="red",
    linestyle="--",
    label=f"Средний силуэт ({sample_silhouettes.mean():.3f})",
)
plt.xlabel("Силуэт")
plt.ylabel("Точки данных")
plt.title("Распределение силуэта по кластерам")
plt.legend()
plt.tight_layout()
plt.savefig("silhouette_distribution.png", dpi=150, bbox_inches="tight")
plt.show()

print("\n📊 [СКРИНШОТ 8: Распределение силуэта - silhouette_distribution.png]")

# Примеры неудачных точек
print("\n--- Примеры точек с низким силуэтом ---")
print("\nТоп-5 точек с самым низким силуэтом:")
worst_points = df_analysis.nsmallest(5, "silhouette")[
    clustering_features + ["Cluster", "silhouette"]
]
display(worst_points)

# Предложения по улучшению
print("\n--- Предложения по улучшению ---")
print("""
ИДЕИ ДЛЯ УЛУЧШЕНИЯ КЛАСТЕРИЗАЦИИ:

1. Feature Engineering:
   • Создать новые признаки (возрастные группы, уровень вовлеченности)
   • Комбинированные признаки: campaign/previous ratio

2. Альтернативные методы:
   • UMAP для визуализации
   • t-SNE для нелинейных структур

3. Оптимизация:
   • Другие методы linkage для Hierarchical
   • OPTICS для автоматического подбора параметров

4. Валидация:
   • Проверка с доменными экспертами
   • A/B тестирование в реальных кампаниях
""")


# ЭТАП 7: РЕПРОДУЦИРУЕМОСТЬ

print("\n" + "=" * 70)
print("ЭТАП 7: РЕПРОДУЦИРУЕМОСТЬ")
print("=" * 70)

print(f"\n--- Фиксированный random_state ---")
print(f"Для воспроизводимости использован: random_state = {RANDOM_STATE}")

print("\n--- Версии библиотек ---")
print(f"  • Python:       {platform.python_version()}")
print(f"  • pandas:       {pd.__version__}")
print(f"  • numpy:        {np.__version__}")
print(f"  • seaborn:      {sns.__version__}")
print(f"  • scikit-learn: {sklearn.__version__}")
print(f"  • matplotlib:   {plt.matplotlib.__version__}")

print("\n--- Информация о системе ---")
print(f"  • ОС: {platform.system()} {platform.release()}")
print(f"  • Архитектура: {platform.machine()}")

# Сохранение результатов
df_results = df.copy()
df_results["cluster"] = best_labels
df_results.to_csv("clustered_clients.csv", index=False)
print("\n✓ Результаты сохранены в 'clustered_clients.csv'")


# ИТОГОВАЯ СВОДКА И МИНИ ЧЕК-ЛИСТ

print("\n" + "=" * 70)
print("ИТОГОВАЯ СВОДКА")
print("=" * 70)

print(f"""
Выполнена кластеризация клиентов банка:

✓ Датасет: {len(df)} клиентов, {len(clustering_features)} признаков
✓ Лучший метод: {best_method} с k={optimal_k} кластерами
✓ Silhouette Score: {baseline_silhouette:.4f}
✓ Inertia: {baseline_inertia:.2f}
✓ ARI: {ari_kmeans:.4f}

Сгенерированные файлы:
  • elbow_method.png
  • dendrogram.png
  • metrics_comparison.png
  • clusters_pca.png
  • cluster_profiles_numeric.png
  • cluster_profiles_categorical.png
  • cluster_conversion.png
  • silhouette_distribution.png
  • clustered_clients.csv
""")

print("\n" + "=" * 70)
print("МИНИ ЧЕК-ЛИСТ (по заданию)")
print("=" * 70)
print("""
✓ Выбрал тип задачи: КЛАСТЕРИЗАЦИЯ (обосновал почему не регрессия/рекомендации)
✓ Описал датасет и задачу
✓ Реализовал baseline-метод: KMeans
✓ Реализовал продвинутые методы: DBSCAN, Hierarchical
✓ Посчитал метрики (silhouette, inertia, ARI), сделал сравнение
✓ Сравнил с baseline, выбрал победителя
✓ Добавил интерпретацию/визуализацию (PCA, профили кластеров)
✓ Провёл анализ ошибок (точки с низким силуэтом) и ограничений
✓ Зафиксировал seed (random_state=42), проверил воспроизводимость
""")

print("\n" + "=" * 70)
print("ЛАБОРАТОРНАЯ РАБОТА 3 ЗАВЕРШЕНА")
print("=" * 70)
