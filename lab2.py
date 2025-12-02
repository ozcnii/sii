# ЭТАП 1

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Загрузка данных
try:
    url = "https://drive.google.com/uc?id=1dvVgFSH22J7okTKYD8sHzJvJJ9MRZzkN&export=download"
    df = pd.read_csv(url, delimiter=';')
    print("Файл успешно загружен.")
except Exception as e:
    print(f"Ошибка загрузки файла: {e}")
    exit(1)

# 1. Удаление дубликатов
df.drop_duplicates(inplace=True)

# 2. Обработка 'unknown' значений (реализация стратегии "замена на моду")
object_cols_for_unknown = df.select_dtypes(include=['object']).columns
for col in object_cols_for_unknown:
    if 'unknown' in df[col].unique():
        mode_value = df[col].mode()[0]
        # Особый случай, если мода сама 'unknown'
        if mode_value == 'unknown':
            second_mode = df[col].value_counts().index[1]
            df[col] = df[col].replace('unknown', second_mode)
        else:
            df[col] = df[col].replace('unknown', mode_value)

# 3. Коррекция типов данных
object_cols = df.select_dtypes(include=['object']).columns
for col in object_cols:
    df[col] = df[col].astype('category')
    # Удаляем 'unknown' из списка возможных категорий, если он остался
    if 'unknown' in df[col].cat.categories:
        df[col] = df[col].cat.remove_unused_categories()

# 4. Преобразование целевой переменной в числовой формат
df['y'] = df['y'].apply(lambda x: 1 if x == 'yes' else 0).astype(int)

print("Очищенный датасет подготовлен.")

# Вывод первых 5 строк датасета
print("\nПервые 5 строк обработанного датасета:")
print("=" * 60)
print(df.head())
print("=" * 60)

# Итоговая размерность
print(f"\nИтоговая размерность: {df.shape}")


# Создание конвейера препроцессинга (Pipeline)
print("\n--- ЭТАП 1: Создание Pipeline для препроцессинга ---")

# Разделение на признаки (X) и цель (y)
X = df.drop('y', axis=1)
y = df['y']

# 1. Автоматическое определение списков признаков
numeric_features = X.select_dtypes(include=np.number).columns
categorical_features = X.select_dtypes(include=['category']).columns

print(f"Найдено {len(numeric_features)} числовых признаков.")
print(f"Найдено {len(categorical_features)} категориальных признаков.")

# 2. Создание единого препроцессора
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_features)
    ],
    remainder='passthrough'
)


# ЭТАП 2: Разбиение данных

from sklearn.model_selection import train_test_split
import pandas as pd


RANDOM_STATE = 42

print("\n--- ЭТАП 2: Разбиение данных на train/valid/test ---")

# 1. Первичное разделение на 80% (обучение+валидация) и 20% (тест)
# (пункт "Стратифицированный train/valid/test = 60/20/20")
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y,
    test_size=0.2,       # 20% данных откладываем для теста
    random_state=RANDOM_STATE,
    stratify=y           # Сохраняем пропорции классов
)

# 2. Вторичное разделение оставшихся 80% на 60% (обучение) и 20% (валидация)
# test_size=0.25, так как 0.25 * 80% = 20% от исходного датасета
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full,
    test_size=0.25,      # 25% от train_full (20% от общего)
    random_state=RANDOM_STATE,
    stratify=y_train_full # Сохраняем пропорции классов
)

# 3. Вывод размеров для проверки
print(f"Размер исходного датасета: {X.shape}")
print(f"Размер обучающей выборки (train): {X_train.shape} (~60%)")
print(f"Размер валидационной выборки (valid): {X_valid.shape} (~20%)")
print(f"Размер тестовой выборки (test): {X_test.shape} (~20%)")

# 4. Проверка стратификации (пункт "Для дисбалансных задач — проверьте стратификацию")
print("\nПроверка распределения классов в выборках:")
print(f"Обучающая выборка (train):\n{y_train.value_counts(normalize=True).round(4)}\n")
print(f"Валидационная выборка (valid):\n{y_valid.value_counts(normalize=True).round(4)}\n")
print(f"Тестовая выборка (test):\n{y_test.value_counts(normalize=True).round(4)}\n")

print("Как видно, пропорции классов во всех выборках практически идентичны.")
print("Тестовая выборка X_test и y_test отложена и не будет использоваться до финальной оценки.")

# ЭТАП 3: Обучение и оценка базовых моделей

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score


print("\n--- ЭТАП 3: Обучение и оценка базовых моделей ---")

# --- 3.1. Логистическая регрессия ---
print("\n--- 3.1. Логистическая регрессия ---")

# Создаем полный pipeline: сначала препроцессинг, потом модель
lr_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(
        random_state=RANDOM_STATE,
        max_iter=1000,
        class_weight='balanced' # Важно для борьбы с дисбалансом
    ))
])

# Обучаем ВЕСЬ pipeline на ТРЕНИРОВОЧНЫХ данных
lr_pipeline.fit(X_train, y_train)

# Делаем предсказания на ВАЛИДАЦИОННЫХ данных
y_pred_lr = lr_pipeline.predict(X_valid)
y_proba_lr = lr_pipeline.predict_proba(X_valid)[:, 1]

print("Отчет по Логистической регрессии на валидационной выборке:")
print(classification_report(y_valid, y_pred_lr))
print(f"ROC-AUC: {roc_auc_score(y_valid, y_proba_lr):.4f}")
print(f"PR-AUC: {average_precision_score(y_valid, y_proba_lr):.4f}")


# --- 3.2. Дерево решений с подбором max_depth ---
print("\n--- 3.2. Дерево решений с подбором max_depth ---")

# Создаем pipeline для дерева решений
dt_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(
        random_state=RANDOM_STATE,
        class_weight='balanced' # Важно для борьбы с дисбалансом
    ))
])

# Задаем сетку параметров для поиска
param_grid = {'classifier__max_depth': [3, 5, 7, 10, None]}

# Создаем объект GridSearchCV для поиска лучших параметров по f1-score
# cv=5 означает 5-кратную кросс-валидацию на тренировочных данных
grid_search_dt = GridSearchCV(dt_pipeline, param_grid, cv=5, scoring='f1', n_jobs=-1)
grid_search_dt.fit(X_train, y_train)

# Получаем лучшую модель после поиска
best_dt_pipeline = grid_search_dt.best_estimator_
print(f"Лучший параметр max_depth, найденный GridSearchCV: {grid_search_dt.best_params_['classifier__max_depth']}")

# Делаем предсказания на ВАЛИДАЦИОННЫХ данных
y_pred_dt = best_dt_pipeline.predict(X_valid)
y_proba_dt = best_dt_pipeline.predict_proba(X_valid)[:, 1]

print("\nОтчет по Дереву решений (с лучшим max_depth) на валидационной выборке:")
print(classification_report(y_valid, y_pred_dt))
print(f"ROC-AUC: {roc_auc_score(y_valid, y_proba_dt):.4f}")
print(f"PR-AUC: {average_precision_score(y_valid, y_proba_dt):.4f}")


# --- 3.3. Краткий вывод ---
print("\n--- Вывод по базовым моделям ---")
print("Логистическая регрессия выбрана как лучший baseline.")
print("Обоснование: Несмотря на сопоставимый f1-score, у логистической регрессии значительно выше обобщающие метрики ROC-AUC и PR-AUC,")
print("что указывает на ее лучшую разделительную способность в целом и делает ее более надежной отправной точкой.")

# ЭТАП 4: РАСШИРЕННЫЙ ПОИСК С БАЛАНСОМ ПРОИЗВОДИТЕЛЬНОСТИ

from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from lightgbm import LGBMClassifier
from scipy.stats import randint, uniform
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, f1_score
import pandas as pd
import numpy as np
import time
import warnings

warnings.filterwarnings('ignore')
print("\nЭТАП 4: Расширенный поиск гиперпараметров LightGBM")

# Метрики baseline
print("\nМетрики Baseline (Логистическая регрессия)")
y_pred_baseline = lr_pipeline.predict(X_valid)
y_proba_baseline = lr_pipeline.predict_proba(X_valid)[:, 1]
baseline_f1 = f1_score(y_valid, y_pred_baseline)
baseline_roc_auc = roc_auc_score(y_valid, y_proba_baseline)
baseline_pr_auc = average_precision_score(y_valid, y_proba_baseline)

print(f"F1: {baseline_f1:.4f}, ROC-AUC: {baseline_roc_auc:.4f}, PR-AUC: {baseline_pr_auc:.4f}")

# Создание pipeline
lgbm_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LGBMClassifier(
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbose=-1
    ))
])

scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
print(f"Scale_pos_weight: {scale_pos_weight:.2f}")

# Пространство параметров
param_dist = {
    'classifier__n_estimators': randint(100, 400),           # 300 вариантов
    'classifier__learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2], # 5 фиксированных
    'classifier__num_leaves': randint(20, 100),              # 80 вариантов
    'classifier__max_depth': randint(3, 12),                 # 9 вариантов
    'classifier__min_child_samples': randint(10, 50),        # 40 вариантов
    'classifier__subsample': [0.7, 0.8, 0.9, 1.0],          # 4 варианта
    'classifier__colsample_bytree': [0.7, 0.8, 0.9, 1.0],   # 4 варианта
    'classifier__reg_alpha': uniform(0, 2),                  # регуляризация
    'classifier__reg_lambda': uniform(0, 2),                 # регуляризация
    'classifier__scale_pos_weight': [scale_pos_weight, scale_pos_weight * 0.8, scale_pos_weight * 1.2]  # 3 варианта
}

random_search = RandomizedSearchCV(
    lgbm_pipeline,
    param_distributions=param_dist,
    n_iter=15,
    cv=3,
    scoring='f1',
    n_jobs=1,
    random_state=RANDOM_STATE,
    verbose=1,
    return_train_score=True
)

start_time = time.time()

random_search.fit(X_train, y_train)

search_time = time.time() - start_time
print(f"Поиск завершен за {search_time:.1f} секунд")

# Анализ результатов
print("\n" + "="*60)
print("Детальный анализ результатов поиска")
print("="*60)

cv_results = pd.DataFrame(random_search.cv_results_)

# Лучшие модели
top_models = cv_results.nlargest(5, 'mean_test_score')[
    ['rank_test_score', 'mean_test_score', 'std_test_score',
     'mean_train_score', 'param_classifier__n_estimators',
     'param_classifier__learning_rate', 'param_classifier__num_leaves',
     'param_classifier__max_depth', 'param_classifier__subsample']
]

print("ТОП-5 моделей:")
for i, row in top_models.iterrows():
    print(f"\n#{int(row['rank_test_score'])} | F1: {row['mean_test_score']:.4f} (±{row['std_test_score']:.4f})")
    print(f"   Параметры: n_est={row['param_classifier__n_estimators']}, "
          f"lr={row['param_classifier__learning_rate']}, "
          f"leaves={row['param_classifier__num_leaves']}, "
          f"depth={row['param_classifier__max_depth']}")

# Анализ переобучения
best_model_idx = cv_results['rank_test_score'].idxmin()
best_train_score = cv_results.loc[best_model_idx, 'mean_train_score']
best_test_score = cv_results.loc[best_model_idx, 'mean_test_score']
overfitting_gap = best_train_score - best_test_score

print(f"\nАнализ переобучения лучшей модели:")
print(f"   Train F1: {best_train_score:.4f}")
print(f"   Test F1:  {best_test_score:.4f}")
print(f"   Разрыв:   {overfitting_gap:.4f} ({overfitting_gap/best_test_score*100:.1f}%)")

if overfitting_gap > 0.1:
    print("Заметное переобучение!")
elif overfitting_gap > 0.05:
    print("Умеренное переобучение")
else:
    print("Хорошая сбалансированность")

# Лучшая модель
best_lgbm = random_search.best_estimator_
print(f"\nЛучшая модель: F1 = {random_search.best_score_:.4f}")

print("\nЛучшие параметры:")
best_params = random_search.best_params_
for param, value in best_params.items():
    clean_param = param.replace("classifier__", "")
    if isinstance(value, float):
        print(f"   {clean_param:<25}: {value:.4f}")
    else:
        print(f"   {clean_param:<25}: {value}")

print("Финальная оценка на валидационной выборке")

y_pred_final = best_lgbm.predict(X_valid)
y_proba_final = best_lgbm.predict_proba(X_valid)[:, 1]

final_f1 = f1_score(y_valid, y_pred_final)
final_roc_auc = roc_auc_score(y_valid, y_proba_final)
final_pr_auc = average_precision_score(y_valid, y_proba_final)

# Сравнение с предыдущими результатами
comparison = pd.DataFrame({
    'Metric': ['F1-Score', 'ROC-AUC', 'PR-AUC'],
    'Baseline (LR)': [baseline_f1, baseline_roc_auc, baseline_pr_auc],
    'LightGBM (простой)': [0.6166, 0.9444, 0.6556],  # из предыдущего запуска
    'LightGBM (расширенный)': [final_f1, final_roc_auc, final_pr_auc],
    'Improvement vs Baseline': [
        f"{(final_f1 - baseline_f1)/baseline_f1*100:+.1f}%",
        f"{(final_roc_auc - baseline_roc_auc)/baseline_roc_auc*100:+.1f}%",
        f"{(final_pr_auc - baseline_pr_auc)/baseline_pr_auc*100:+.1f}%"
    ]
})

print(comparison.to_string(index=False))

# Преимущества модели
print("\nВизуализация результатов:")

improvement_f1 = (final_f1 - 0.6166) / 0.6166 * 100
improvement_roc = (final_roc_auc - 0.9444) / 0.9444 * 100
improvement_pr = (final_pr_auc - 0.6556) / 0.6556 * 100

print(f"   F1:      {final_f1:.4f} ({improvement_f1:+.1f}% vs простой поиск)")
print(f"   ROC-AUC: {final_roc_auc:.4f} ({improvement_roc:+.1f}% vs простой поиск)")
print(f"   PR-AUC:  {final_pr_auc:.4f} ({improvement_pr:+.1f}% vs простой поиск)")

if final_f1 > 0.6166:
    print(f"\nРасширенный поиск дал улучшение")
else:
    print(f"\nРасширенный поиск не дал значительного улучшения.")

print("Отчет по классификации:")
print(classification_report(y_valid, y_pred_final))

# ЭТАП 5: Интерпретация и анализ ошибок

from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("\n--- ЭТАП 5: Финальная оценка, интерпретация и анализ ошибок ---")

# --- 5.1. Финальная оценка на ТЕСТОВОЙ выборке ---
print("\n--- 5.1. Финальная оценка на ТЕСТОВОЙ выборке ---")
y_pred_test = best_lgbm.predict(X_test)
y_proba_test = best_lgbm.predict_proba(X_test)[:, 1]

print("Финальный отчет по LightGBM на тестовой выборке:")
print(classification_report(y_test, y_pred_test))
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba_test):.4f}")
print(f"PR-AUC: {average_precision_score(y_test, y_proba_test):.4f}")

# --- 5.2. Важность признаков (Permutation Importance) ---
print("\n--- 5.2. Важность признаков (Permutation Importance) ---")

# ШАГ 1: Получаем обученные шаги из пайплайна
preprocessor_fitted = best_lgbm.named_steps['preprocessor']
model_fitted = best_lgbm.named_steps['classifier']

# ШАГ 2: Преобразуем тестовые данные с помощью обученного препроцессора
X_test_transformed = preprocessor_fitted.transform(X_test)

# ШАГ 3: Запускаем Permutation Importance на ОБРАБОТАННЫХ данных и ОБУЧЕННОЙ модели
perm_importance = permutation_importance(
    model_fitted, X_test_transformed, y_test, n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1
)

# ШАГ 4: Получаем правильные имена признаков из препроцессора
feature_names = preprocessor_fitted.get_feature_names_out()

# ШАГ 5: Создаем DataFrame (теперь длины точно совпадут)
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance_mean': perm_importance.importances_mean,
}).sort_values('importance_mean', ascending=False)

print("\nТоп-10 самых важных признаков для модели:")
print(importance_df.head(10).to_string())

# Визуализация
plt.figure(figsize=(10, 8))
sns.barplot(x='importance_mean', y='feature', data=importance_df.head(20), palette='viridis')
plt.title('Топ-20 самых важных признаков (Permutation Importance)', fontsize=16)
plt.xlabel('Среднее снижение метрики')
plt.ylabel('Признак')
plt.tight_layout()
plt.show()

# --- 5.3. Анализ ошибок ---
print("\n--- 5.3. Анализ ошибок ---")
results_df = X_test.copy()
results_df['true_label'] = y_test
results_df['predicted_label'] = y_pred_test
results_df['predicted_proba_1'] = y_proba_test

false_positives = results_df[(results_df['true_label'] == 0) & (results_df['predicted_label'] == 1)]
print(f"\nНайдено {len(false_positives)} ложноположительных ошибок (FP). Примеры:")
display(false_positives.head(3))

false_negatives = results_df[(results_df['true_label'] == 1) & (results_df['predicted_label'] == 0)]
print(f"\nНайдено {len(false_negatives)} ложноотрицательных ошибок (FN). Примеры:")
display(false_negatives.head(3))

print("\n--- Комментарии и шаги по улучшению ---")
print("Анализ ошибок показывает:")
print(" - FP (ложная тревога): Модель иногда предсказывает согласие для клиентов, с которыми был долгий разговор ('duration'), но которые в итоге отказались.")
print(" - FN (пропуск цели): Модель часто упускает клиентов, которые согласились, но имели нетипичные для 'успешных' клиентов параметры (например, короткий звонок).")
print("\nПредлагаемые шаги по улучшению:")
print(" 1. Feature Engineering: Создать новые признаки, которые могут лучше отражать сложные зависимости. Например, 'отношение длительности к числу контактов' (duration/campaign) или 'звонок после успешной прошлой кампании'.")
print(" 2. Изменение порога классификации: По умолчанию порог равен 0.5. Можно попробовать его снизить (например, до 0.4), чтобы уменьшить число дорогих для бизнеса FN (пропусков клиентов) за счет некоторого увеличения FP (лишних звонков).")

# ЭТАП 6: Репродуцируемость

import platform
import pandas as pd
import numpy as np
import seaborn as sns
import sklearn
import lightgbm

print("\n--- ЭТАП 6: Репродуцируемость ---")

# 1. Вывод зафиксированного random_state
print(f"\nДля обеспечения воспроизводимости результатов во всех алгоритмах,")
print(f"использующих случайность, был зафиксирован state: random_state = {RANDOM_STATE}")

# 2. Вывод версий ключевых библиотек
print("\nВерсии библиотек, использованных в проекте:")
print(f"  - Python:       {platform.python_version()}")
print(f"  - pandas:       {pd.__version__}")
print(f"  - numpy:        {np.__version__}")
print(f"  - seaborn:      {sns.__version__}")
print(f"  - scikit-learn: {sklearn.__version__}")
print(f"  - lightgbm:     {lightgbm.__version__}")

print("\nЛабораторная работа №2 завершена.")