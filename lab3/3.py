# ЭТАП 3: ПРОДВИНУТЫЕ МЕТОДЫ (DBSCAN и Hierarchical Clustering)

print("\nЭТАП 3: ПРОДВИНУТЫЕ МЕТОДЫ (DBSCAN и Hierarchical Clustering)")

# AgglomerativeClustering в sklearn - это реализация
# иерархической кластеризации "снизу вверх" (agglomerative = hierarchical).

# 3.1 DBSCAN
print("\n3.1 DBSCAN (Density-Based Spatial Clustering)")

# Подбор параметров DBSCAN
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

print(f"Лучшие параметры: eps={best_eps}, min_samples={best_min_samples}")

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
print(f"  - Количество кластеров: {n_clusters_dbscan}")
print(
    f"  - Точки-шум (outliers): {n_noise_dbscan} ({n_noise_dbscan / len(df) * 100:.2f}%)"
)
if not np.isnan(dbscan_silhouette):
    print(f"  - Silhouette Score: {dbscan_silhouette:.4f}")
print(f"\nРаспределение по кластерам (-1 = шум):")
print(pd.Series(dbscan_labels).value_counts().sort_index())

# 3.2 Hierarchical (Иерархическая) кластеризация
print("\n3.2 Hierarchical (Иерархическая) кластеризация")

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


print(f"\nОбучение Hierarchical Clustering с k={optimal_k}...")
hierarchical = AgglomerativeClustering(n_clusters=optimal_k, linkage="ward")
hierarchical_labels = hierarchical.fit_predict(X_scaled)

df["cluster_hierarchical"] = hierarchical_labels

# Метрики Hierarchical
hierarchical_silhouette = silhouette_score(X_scaled, hierarchical_labels)
hierarchical_davies_bouldin = davies_bouldin_score(X_scaled, hierarchical_labels)
hierarchical_calinski = calinski_harabasz_score(X_scaled, hierarchical_labels)

print(f"Hierarchical Clustering обучен")
print(f"\nМетрики Hierarchical Clustering:")
print(f"  - Silhouette Score: {hierarchical_silhouette:.4f}")
print(f"  - Davies-Bouldin Index: {hierarchical_davies_bouldin:.4f}")
print(f"  - Calinski-Harabasz Index: {hierarchical_calinski:.2f}")
print(f"\nРаспределение по кластерам:")
print(pd.Series(hierarchical_labels).value_counts().sort_index())


print("\nСРАВНЕНИЕ ПРОДВИНУТЫХ МЕТОДОВ С BASELINE И ВЫБОР ПОБЕДИТЕЛЯ")

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

methods_scores = {
    "KMeans": baseline_silhouette,
    "DBSCAN": dbscan_silhouette if not np.isnan(dbscan_silhouette) else -1,
    "Hierarchical": hierarchical_silhouette,
}

winner = max(methods_scores, key=methods_scores.get)
winner_score = methods_scores[winner]

print(f"\nПОБЕДИТЕЛЬ: {winner}")
print(f"Silhouette Score: {winner_score:.4f}")


if winner == "KMeans":
    best_labels = kmeans_labels
elif winner == "DBSCAN":
    best_labels = dbscan_labels
elif winner == "Hierarchical":
    best_labels = hierarchical_labels
else:
    best_labels = kmeans_labels

best_method = winner


