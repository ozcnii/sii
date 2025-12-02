# ЭТАП 2: БАЗОВЫЙ МЕТОД (BASELINE) - K-MEANS

print("ЭТАП 2: БАЗОВЫЙ МЕТОД (BASELINE) - K-MEANS КЛАСТЕРИЗАЦИЯ")


df_encoded = df_cluster.copy()

label_encoders = {}
categorical_cols = df_encoded.select_dtypes(include=["object"]).columns

for col in categorical_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])
    label_encoders[col] = le
    print(f"{col}: закодировано {len(le.classes_)} категорий")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_encoded)
print("\nДанные стандартизированы (StandardScaler)")

# Метод локтя для определения оптимального количества кластеров
print("\nМетод локтя (Elbow Method)")

inertias = []
silhouettes = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouettes.append(silhouette_score(X_scaled, kmeans.labels_))
    print(f"k={k}: Inertia={kmeans.inertia_:.2f}, Silhouette={silhouettes[-1]:.4f}")

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

optimal_k = 4
print(f"\nОптимальное количество кластеров: k = {optimal_k}")

# Обучение финальной модели KMeans (BASELINE)
print("\nОбучение KMeans (BASELINE) с k={}".format(optimal_k))

kmeans_final = KMeans(n_clusters=optimal_k, random_state=RANDOM_STATE, n_init=10)
kmeans_labels = kmeans_final.fit_predict(X_scaled)

df["cluster_kmeans"] = kmeans_labels

# Метрики для baseline
baseline_silhouette = silhouette_score(X_scaled, kmeans_labels)
baseline_inertia = kmeans_final.inertia_
baseline_davies_bouldin = davies_bouldin_score(X_scaled, kmeans_labels)
baseline_calinski = calinski_harabasz_score(X_scaled, kmeans_labels)

print(f"KMeans (BASELINE) обучен")
print(f"\nМетрики BASELINE (KMeans):")
print(f"  - Silhouette Score: {baseline_silhouette:.4f}")
print(f"  - Inertia: {baseline_inertia:.2f}")
print(f"  - Davies-Bouldin Index: {baseline_davies_bouldin:.4f}")
print(f"  - Calinski-Harabasz Index: {baseline_calinski:.2f}")

print(f"\nРаспределение по кластерам:")
print(pd.Series(kmeans_labels).value_counts().sort_index())
