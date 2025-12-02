
print("\nЭТАП 4: МЕТРИКИ И СРАВНЕНИЕ")

# Silhouette Score (коэффициент силуэта)
# Inertia (инерция) - только для KMeans
# ARI (Adjusted Rand Index) - для сравнения с "эталонной" разметкой

print("\n4.1 Вычисление метрик")

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

print("\nИТОГОВАЯ ТАБЛИЦА МЕТРИК")

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

print("\n4.2 КРАТКИЙ ВЫВОД ПО МЕТРИКАМ")

dbscan_sil_str = (
    f"{dbscan_silhouette:.4f}" if not np.isnan(dbscan_silhouette) else "N/A"
)

print(f"""
Анализ метрик показывает:

1. SILHOUETTE SCORE:
   - KMeans: {baseline_silhouette:.4f}
   - Hierarchical: {hierarchical_silhouette:.4f}
   - DBSCAN: {dbscan_sil_str}

2. INERTIA (только KMeans):
   - KMeans: {baseline_inertia:.2f}

3. ARI (сравнение с целевой переменной y):
   - KMeans: {ari_kmeans:.4f}
   - Hierarchical: {ari_hierarchical:.4f}
""")
