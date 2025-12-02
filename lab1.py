# ЭТАП 1

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

PALETTE = "Set2"

COLUMN_TRANSLATOR = {
    'age': 'Возраст',
    'job': 'Профессия',
    'marital': 'Семейное положение',
    'education': 'Образование',
    'default': 'Кредитный дефолт',
    'housing': 'Ипотека',
    'loan': 'Потребительский кредит',
    'contact': 'Тип контакта',
    'month': 'Месяц',
    'day_of_week': 'День недели',
    'duration': 'Длительность звонка (сек)',
    'campaign': 'Кол-во контактов в кампании',
    'pdays': 'Дней с прошлого контакта',
    'previous': 'Кол-во прошлых контактов',
    'poutcome': 'Результат прошлой кампании',
    'emp.var.rate': 'Изм. уровня занятости',
    'cons.price.idx': 'Индекс потреб. цен',
    'cons.conf.idx': 'Индекс потреб. доверия',
    'euribor3m': 'Ставка Euribor 3M',
    'nr.employed': 'Число занятых',
    'y': 'Согласие на вклад'
}


VALUE_TRANSLATOR = {
    'job': {
        'admin.': 'Администратор', 'blue-collar': 'Рабочий', 'technician': 'Тех. специалист',
        'services': 'Сфера услуг', 'management': 'Менеджмент', 'retired': 'Пенсионер',
        'entrepreneur': 'Предприниматель', 'self-employed': 'Самозанятый',
        'housemaid': 'Домработница', 'unemployed': 'Безработный', 'student': 'Студент'
    },
    'marital': {
        'married': 'Женат/Замужем', 'single': 'Холост/Не замужем', 'divorced': 'В разводе'
    },
    'education': {
        'university.degree': 'Высшее образование', 'high.school': 'Старшая школа',
        'professional.course': 'Проф. курсы', 'basic.9y': '9 классов',
        'basic.4y': 'Начальная школа', 'basic.6y': '6 классов', 'illiterate': 'Неграмотный'
    },
    'poutcome': {
        'success': 'Успех', 'failure': 'Неудача', 'nonexistent': 'Не было'
    },
    'month': {
        'mar': 'Март', 'dec': 'Дек', 'sep': 'Сен', 'oct': 'Окт', 'apr': 'Апр',
        'aug': 'Авг', 'jun': 'Июнь', 'nov': 'Ноя', 'jul': 'Июль', 'may': 'Май'
    },
     'day_of_week': {
        'thu': 'Чт', 'tue': 'Вт', 'wed': 'Ср', 'fri': 'Пт', 'mon': 'Пн'
    },
}


sns.set(style="whitegrid")

try:
    url = "https://drive.google.com/uc?id=1dvVgFSH22J7okTKYD8sHzJvJJ9MRZzkN&export=download"
    df = pd.read_csv(url, delimiter=';')
    print("Файл успешно загружен")
except Exception as e:
    print(f"Ошибка загрузки файла: {e}")
    print("Программа завершена.")
    exit(1)


# 1. Первичный осмотр
print("\nПервичный осмотр")
print("\nПервые 5 строк:")
display(df.head())
print("\nПоследние 5 строк:")
display(df.tail())

print(f"\nРазмерность датасета (строки, столбцы): {df.shape}")
print("\nИсходные типы данных и информация:")
df.info()

# 2. Проверка качества данных
print("\nПроверка качества данных")
# Проверка на явные пропуски
if df.isnull().sum().sum() == 0:
    print("\nВ датасете нет стандартных пропущенных значений (NaN).")
else:
    print("\nКоличество стандартных пропущенных значений (NaN):")
    print(df.isnull().sum()[df.isnull().sum() > 0])

# Проверка на дубликаты
duplicate_rows = df.duplicated().sum()
print(f"\nКоличество полностью дублирующихся строк: {duplicate_rows}")

# Обнаружили 12 полностью дублирующихся строк. Это составляет < 0.03% от всех данных и, скорее всего, является ошибкой при сборе информации.
# Дубликаты удаляем, так как они не несут новой информации и могут незначительно исказить статистику
if duplicate_rows > 0:
    df.drop_duplicates(inplace=True)
    print(f"Дубликаты удалены. Новая размерность датасета: {df.shape}")

# Анализ и обработка "скрытых пропусков" ('unknown')
# В данных присутствуют значения 'unknown', которые являются скрытыми пропусками
# Поскольку удаление строк с этими значениями привело бы к потере значительной части данных,
# применяем стратегию импутации (замены) на самое частое значение (моду) для каждого столбца,
# что позволит сохранить все наблюдения, минимально искажая исходное распределение признаков

print("\nАнализ и обработка 'unknown'")
print("Значения 'unknown' являются скрытыми пропусками. Проанализируем их количество.")
# Выбираем столбцы типа 'object', так как 'unknown' - это строка
object_cols_for_unknown = df.select_dtypes(include=['object']).columns
for col in object_cols_for_unknown:
    if 'unknown' in df[col].unique():
        unknown_count = df[col].value_counts().get('unknown', 0)
        if unknown_count > 0:
            unknown_percent = (unknown_count / len(df)) * 100
            print(f"- Столбец '{col}': {unknown_count} значений 'unknown' ({unknown_percent:.2f}%)")

print("\nПрименяем стратегию: замена 'unknown' на самое частое значение (моду).")
for col in object_cols_for_unknown:
    if 'unknown' in df[col].unique():
        mode_value = df[col].mode()[0]
        if mode_value != 'unknown':
            df[col].replace('unknown', mode_value, inplace=True)
            print(f"В столбце '{col}' значения 'unknown' заменены на '{mode_value}'.")
        else: # Особый случай, если мода сама 'unknown'
            second_mode = df[col].value_counts().index[1]
            df[col].replace('unknown', second_mode, inplace=True)
            print(f"В столбце '{col}' значения 'unknown' заменены на вторую по частоте категорию '{second_mode}'.")


# Проверка и коррекция типов данных
print("\n--- 1.5. Коррекция типов данных ---")
# Столбцы с типом 'object' являются категориальными (содержат ограниченный набор значений), поэтому преобразовываем их в специальный тип 'category'
# Это значительно сокращает потребление памяти и может ускорить группировку и другие операции
print("Преобразуем столбцы типа 'object' в 'category' для оптимизации...")
object_cols = df.select_dtypes(include=['object']).columns
for col in object_cols:
    df[col] = df[col].astype('category')
    # После замены 'unknown' нужно удалить эту категорию из списка возможных
    if 'unknown' in df[col].cat.categories:
        df[col] = df[col].cat.remove_unused_categories()

print("\nПреобразование завершено. Финальная информация о датасете:")
df.info()

# ЭТАП 2

print("\n\n--- 2.1. Анализ числовых признаков ---")
numeric_cols = df.select_dtypes(include=np.number).columns
print("Описательная статистика:")
display(df[numeric_cols].describe())

print("\n--- Численный анализ формы распределения ---")
skewness = df[numeric_cols].skew()
kurtosis = df[numeric_cols].kurt()
interpretation_df = pd.DataFrame({'Skewness': skewness, 'Kurtosis': kurtosis})
interpretation_df['Интерпретация асимметрии'] = interpretation_df['Skewness'].apply(
    lambda s: 'Сильная правая (>1)' if s > 1 else ('Умеренная правая (0.5-1)' if s > 0.5 else ('Почти симметрично (-0.5-0.5)' if s > -0.5 else ('Умеренная левая (-1..-0.5)' if s > -1 else 'Сильная левая (<-1)')))
)
interpretation_df['Интерпретация куртозиса (хвостов)'] = interpretation_df['Kurtosis'].apply(
    lambda k: 'Очень тяжелые/длинные (>3)' if k > 3 else ('Тяжелые/длинные (1-3)' if k > 1 else ('Нормальные (-1-1)' if k > -1 else 'Легкие/короткие (<-1)'))
)
interpretation_df.rename(index=COLUMN_TRANSLATOR, inplace=True)
display(interpretation_df)

print("\n--- Визуализация распределений числовых признаков ---")
for col in numeric_cols:
    russian_col_name = COLUMN_TRANSLATOR.get(col, col)

    plt.figure(figsize=(15, 4))
    plt.suptitle(f'Анализ признака "{russian_col_name}"', fontsize=16)

    plt.subplot(1, 2, 1)
    sns.histplot(df[col], kde=True, bins=30, color=sns.color_palette(PALETTE, 1)[0])
    plt.title('Гистограмма и плотность')
    plt.xlabel(russian_col_name)
    plt.ylabel('Частота')

    plt.subplot(1, 2, 2)
    sns.boxplot(x=df[col], palette=PALETTE)
    plt.title('Boxplot для выявления выбросов')
    plt.xlabel(russian_col_name)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

# --- 2.2. Анализ категориальных признаков (пункт 4) ---
print("\n\n--- 2.2. Анализ категориальных признаков (после очистки) ---")
categorical_cols = df.select_dtypes(include=['category']).columns
for col in categorical_cols:
    russian_col_name = COLUMN_TRANSLATOR.get(col, col)

    print(f"\n--- Распределение признака '{russian_col_name}' ---")
    print(df[col].value_counts(normalize=True).mul(100).round(2).astype(str) + '%')

    plt.figure(figsize=(10, max(5, df[col].nunique() * 0.4)))

    plot_data = df[col].copy()
    if col in VALUE_TRANSLATOR:
        plot_data = plot_data.map(VALUE_TRANSLATOR[col])

    sns.countplot(y=plot_data, order=plot_data.value_counts().index, palette=PALETTE)
    plt.title(f'Распределение клиентов по признаку "{russian_col_name}"', fontsize=14)
    plt.xlabel('Количество клиентов')
    plt.ylabel('')
    plt.tight_layout()
    plt.show()


# ЭТАП 3

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu


print("--- 5. Анализ связи целевой переменной с категориальными признаками ---")

categorical_features_for_analysis = [col for col in df.select_dtypes(include=['category']).columns if col != 'y']

for feature in categorical_features_for_analysis:
    russian_feature_name = COLUMN_TRANSLATOR.get(feature, feature)

    prop_df = df.groupby(feature, observed=False)['y'].value_counts(normalize=True).unstack().fillna(0)
    prop_df = prop_df.sort_values(by='yes', ascending=False)

    # Создаем копию индекса для перевода
    translated_index = prop_df.index.to_series()
    if feature in VALUE_TRANSLATOR:
        translated_index = translated_index.map(VALUE_TRANSLATOR[feature])

    prop_df.index = translated_index
    prop_df.columns.name = None
    prop_df.index.name = None

    print(f"\nАнализ конверсии для признака: '{russian_feature_name}'")
    display((prop_df * 100).round(2))

    prop_df.rename(columns={'no': 'Отказался', 'yes': 'Согласился'}, inplace=True)
    prop_df.plot(kind='bar', stacked=True, figsize=(12, 6), colormap=PALETTE)

    plt.title(f'Доля согласий в разрезе признака "{russian_feature_name}"', fontsize=16)
    plt.xlabel(russian_feature_name, fontsize=12)
    plt.ylabel('Доля', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title=COLUMN_TRANSLATOR['y'])

    for i, percentage in enumerate(prop_df['Согласился']):
        if percentage > 0.05:
             # Небольшая коррекция координат текста, чтобы он оставался внутри правильной секции
             plt.text(i, prop_df['Отказался'].iloc[i] + percentage / 2, f'{percentage:.1%}', ha='center', va='center', color='white', fontsize=10, weight='bold')

    plt.show()

print("\nВыводы:")
print("\n1) Наиболее заинтересованными в банковских услугах являются студенты, пенсионеры и безработные. Это может быть связано с нехваткой денег.")
print("\n\n2) Одинокие люди пользуются банковскими услугами чаще. Возможно, они больше заинтересованны в построении своей карьеры и бизнеса.")
print("\nДругое предпололжение заключается в том, что обеспеченные люди чаще заводят семьи и имеют более стабильную жизнь.")
print("\n\n3) При звонке на мобильный телефон процент согласия выше почти в 3 раза")
print("\n\n4) Зависимость согласия от месяца, в который произошла связь с клиентом, довольно сильная и хаотичная")

print("\n\n--- 6. Сравнение распределений числовых признаков для групп 'yes' и 'no' ---")

numeric_cols = df.select_dtypes(include=np.number).columns

for feature in numeric_cols:
    russian_feature_name = COLUMN_TRANSLATOR.get(feature, feature)

    plt.figure(figsize=(10, 6))
    sns.boxplot(x='y', y=feature, data=df, hue='y', palette=PALETTE, legend=False)
    plt.title(f'Распределение "{russian_feature_name}" в зависимости от ответа клиента', fontsize=16)
    plt.xlabel(COLUMN_TRANSLATOR['y'], fontsize=12)
    plt.ylabel(russian_feature_name, fontsize=12)
    plt.xticks(ticks=[0, 1], labels=['Отказался', 'Согласился'])
    plt.show()

print("\nКомментарии:")
print("\n1) Люди, не заинтересованные в банковских услугах, довольно быстро заканчивают разговор.")
print("\n2) Чем больше компания связывается с клиентом, тем выше вероятность, что в результате человек откажется от услуг.")
print("\n3) Люди, с которыми связывались до этой маркетинговой кампании, и знакомы с предлагаемыми услугами, охотнее соглашаются.")

print("\nПроведение статистического теста (U-критерий Манна-Уитни)")

features_for_test = ['duration', 'campaign', 'previous', 'pdays', 'age']


for i in range(len(features_for_test)):
    for j in range(i + 1, len(features_for_test)):
        feature1 = features_for_test[i]
        feature2 = features_for_test[j]

        russian_feature1_name = COLUMN_TRANSLATOR.get(feature1, feature1)
        russian_feature2_name = COLUMN_TRANSLATOR.get(feature2, feature2)

        group1 = df[feature1]
        group2 = df[feature2]

        stat, p_value = mannwhitneyu(group1, group2, alternative='two-sided')

        print(f"\nРезультат теста '{russian_feature1_name}' и '{russian_feature2_name}':")

        print(f"\nU-статистика: {stat:.3f}")
        print(f"P-value: {p_value:.3f}")
        print(f"Размеры выборок: {len(group1):,} vs {len(group2):,}")
        print(f"Медиана '{russian_feature1_name}': {group1.median():.2f}")
        print(f"Медиана '{russian_feature2_name}': {group2.median():.2f}")
        print(f"Среднее '{russian_feature1_name}': {group1.mean():.2f}")
        print(f"Среднее '{russian_feature2_name}': {group2.mean():.2f}")

        if p_value < 0.05:
            print(f"Вывод: p-value < 0.05, распределения статистически значимо различаются.")
            print(f"Признаки '{russian_feature1_name}' и '{russian_feature2_name}' имеют разные распределения")
        else:
            print(f"Вывод: p-value >= 0.05, статистически значимых различий не обнаружено")
            print(f"Признаки '{russian_feature1_name}' и '{russian_feature2_name}' имеют схожие распределения.")

plt.figure(figsize=(14, 10))

df_renamed_numeric = df[numeric_cols].rename(columns=COLUMN_TRANSLATOR)
correlation_matrix = df_renamed_numeric.corr()

sns.heatmap(correlation_matrix, annot=True, cmap=PALETTE, fmt='.2f', linewidths=.5)

plt.title('Тепловая карта корреляций числовых признаков', fontsize=16)
plt.show()


print("\n\nВыводы:")
print("\nСвязь между внешними факторами (параметрами экономического контекста) не анализировалась, так как не связна с целью исследования - клиентами.")

print("\n\n1) Признаки 'pdays' и 'previous имеют сильную отрицательную корреляцию (-0.59). Это логично, так как чем чаще клиенту звонили,")
print("\n2)В остальном, числовые признаки почти не кореллируют, что предполагает их независимость.")

# ЭТАП 4:

import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
# Добавим display для красивого вывода в Jupyter
from IPython.display import display

df_prepared = df.copy()

print("--- 8. Работа с пропусками ---")
print("Стратегия: На начальном этапе анализа 'скрытые' пропуски (значения 'unknown')")
print("были заменены на самое частое значение (моду) в соответствующем столбце.")
print("Это позволило сохранить все строки данных, избежав потери информации.\n")


print("--- 9. Кодирование категорий ---")

print("Подход: Для преобразования категориальных признаков в числовой вид будет использован метод One-Hot Encoding.")
print("Этот метод создает новые бинарные столбцы (0/1) для каждой категории, что позволяет избежать")
print("неявного установления порядка между категориями (например, 'admin' не 'больше' чем 'student').")
print("Это является стандартным и наиболее надежным подходом для большинства моделей машинного обучения.\n")

print("--- 10. Масштабирование числовых признаков ---")
print("Подход: Для числовых признаков будет применена стандартизация (StandardScaler).")
print("Она приводит все признаки к единому масштабу (среднее=0, стандартное отклонение=1).")
print("Это необходимо для корректной работы моделей, чувствительных к масштабу, таких как логистическая регрессия или SVM.\n")


X = df_prepared.drop('y', axis=1)
y = df_prepared['y']

numeric_features = X.select_dtypes(include=np.number).columns
categorical_features = X.select_dtypes(include=['category']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)

X_transformed = preprocessor.fit_transform(X)

print("--- Демонстрация результата подготовки данных ---")
print(f"Размерность данных после One-Hot Encoding и масштабирования: {X_transformed.shape}")
print("Итог: Получен полностью числовой, чистый датасет, готовый для моделирования.")
print("Структура признаков сохранена в объекте 'preprocessor'.\n\n")

pd.set_option('display.max_columns', None)

print("--- Наглядный пример работы preprocessor'а ---")

print("\n1. ИСХОДНЫЕ ДАННЫЕ (первые 5 строк):")
# Переводим названия колонок для наглядности
display(X.head().rename(columns=COLUMN_TRANSLATOR))

df_transformed = pd.DataFrame(X_transformed, columns=preprocessor.get_feature_names_out(), index=X.index)

print("\n2. ДАННЫЕ ПОСЛЕ ПРЕОБРАЗОВАНИЯ (первые 5 строк):")
print("  - Числовые столбцы (с префиксом 'num__') теперь стандартизированы (значения распределены вокруг 0).")
print("  - Категориальные столбцы (с префиксом 'cat__') превратились в множество бинарных колонок (0 или 1).")
print("    Например, 'job' превратился в 'cat__job_admin.', 'cat__job_blue-collar' и т.д.")
display(df_transformed.head())

pd.reset_option('display.max_columns')

# ЭТАП 5

print("ЭТАП 5: АНАЛИЗ ВЫБРОСОВ ПО МЕТОДУ IQR")

for col in numeric_features:
    russian_col_name = COLUMN_TRANSLATOR.get(col, col)

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]

    print(f"\nАнализ '{russian_col_name}':")
    print(f"  Q1: {Q1:.2f}, Q3: {Q3:.2f}, IQR: {IQR:.2f}")
    print(f"  Границы нормальных значений: от {lower_bound:.2f} до {upper_bound:.2f}")

    percents = (len(outliers)/len(df))*100
    status = "😨" if percents > 5 else "🦐"
    if not outliers.empty:
        print(f"  {status} Найдено выбросов: {len(outliers)} ({percents:.2f}%)")
    else:
        # Статус для отсутствия выбросов
        status_ok = "✅"
        print(f"  {status_ok} Выбросов не найдено.")

print("\n\n11. Влияние выбросов. Обработка выбросов на примере 'Кол-во контактов в кампании'")

feature_to_cap = "campaign"
russian_feature_name = COLUMN_TRANSLATOR.get(feature_to_cap, feature_to_cap)

print(f"\nСтатистика '{russian_feature_name}' до обработки выбросов:")
display(df[feature_to_cap].describe())

plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
# Используем палитру
sns.boxplot(x=df[feature_to_cap], palette=PALETTE)
plt.title("До обработки", fontsize=14)
plt.xlabel(russian_feature_name)

Q1_camp = df[feature_to_cap].quantile(0.25)
Q3_camp = df[feature_to_cap].quantile(0.75)
IQR_camp = Q3_camp - Q1_camp
upper_bound_camp = Q3_camp + 1.5 * IQR_camp
print(f"\nСтратегия обработки: кэппинг. Верхняя граница: {upper_bound_camp:.2f}.")

df_capped = df.copy()

df_capped[feature_to_cap] = np.where(
    df_capped[feature_to_cap] > upper_bound_camp,
    upper_bound_camp, # Заменяем на границу
    df_capped[feature_to_cap] # Оставляем как есть
)

print(f"\nСтатистика '{russian_feature_name}' после обработки кэппингом:")
display(df_capped[feature_to_cap].describe())

plt.subplot(1, 2, 2)
# Используем палитру
sns.boxplot(x=df_capped[feature_to_cap], palette=PALETTE)
plt.title("После обработки", fontsize=14)
plt.xlabel(f"{russian_feature_name} (ограничено)")

plt.suptitle("Демонстрация влияния кэппинга на выбросы", fontsize=18)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# ЭТАП 6

import sys
import sklearn
import platform

RANDOM_STATE = 42
print(f"--- Воспроизводимость ---")
print(f"Для всех случайных процессов будет использован random_state = {RANDOM_STATE}\n")


print("--- Версии библиотек ---")
print(f"Python: {platform.python_version()}")
print(f"pandas: {pd.__version__}")
print(f"numpy: {np.__version__}")
print(f"seaborn: {sns.__version__}")
print(f"scikit-learn: {sklearn.__version__}")