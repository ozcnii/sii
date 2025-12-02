
# ЭТАП 6: АНАЛИЗ ОШИБОК ИЛИ РЕЗУЛЬТАТОВ

print("\nЭТАП 6: АНАЛИЗ РЕЗУЛЬТАТОВ И ОГРАНИЧЕНИЙ")

# Вычисление силуэта для каждой точки
sample_silhouettes = silhouette_samples(X_scaled, best_labels)
df_analysis["silhouette"] = sample_silhouettes

print(f"\nСтатистика силуэта по точкам:")
print(f"  - Мин: {sample_silhouettes.min():.4f}")
print(f"  - Макс: {sample_silhouettes.max():.4f}")
print(f"  - Среднее: {sample_silhouettes.mean():.4f}")
print(f"  - Медиана: {np.median(sample_silhouettes):.4f}")

bad_points = df_analysis[df_analysis["silhouette"] < 0]
print(
    f"\nНайдено {len(bad_points)} точек с отрицательным силуэтом ({len(bad_points) / len(df_analysis) * 100:.2f}%)"
)
print("Это точки, которые ближе к соседнему кластеру, чем к своему.")


plt.figure(figsize=(12, 6))
for cluster_id in unique_clusters:
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

print("\nПримеры точек с низким силуэтом")
print("\nТоп-5 точек с самым низким силуэтом:")
worst_points = df_analysis.nsmallest(5, "silhouette")[
    clustering_features + ["Cluster", "silhouette"]
]
display(worst_points)
