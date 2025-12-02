# ЭТАП 5: ИНТЕРПРЕТАЦИЯ И ВИЗУАЛИЗАЦИЯ

print("\nЭТАП 5: ИНТЕРПРЕТАЦИЯ И ВИЗУАЛИЗАЦИЯ")

print("\n5.1 Визуализация кластеров (PCA)")

pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_scaled)

print(
    f"PCA: объяснённая дисперсия = {pca.explained_variance_ratio_.sum() * 100:.2f}%"
)
print(f"  - PC1: {pca.explained_variance_ratio_[0] * 100:.2f}%")
print(f"  - PC2: {pca.explained_variance_ratio_[1] * 100:.2f}%")

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

# 5.2 Профили кластеров
print("\n5.2 Профили кластеров")

df_analysis = df_cluster.copy()
df_analysis["Cluster"] = best_labels

numeric_features = ["age", "campaign", "pdays", "previous"]

cluster_profiles_numeric = df_analysis.groupby("Cluster")[numeric_features].agg(
    ["mean", "median", "std"]
)
display(cluster_profiles_numeric.round(2))

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

# 5.3 Интерпретация кластеров
print("\n5.3 Интерпретация кластеров")

unique_clusters = sorted([c for c in df_analysis["Cluster"].unique() if c != -1])

for cluster_id in unique_clusters:
    cluster_data = df_analysis[df_analysis["Cluster"] == cluster_id]
    print(f"\nКЛАСТЕР {cluster_id}")
    print(
        f"Размер: {len(cluster_data)} клиентов ({len(cluster_data) / len(df_analysis) * 100:.1f}%)"
    )
    print(f"\nЧисловые характеристики:")
    print(f"  - Средний возраст: {cluster_data['age'].mean():.1f} лет")
    print(f"  - Среднее кол-во контактов: {cluster_data['campaign'].mean():.1f}")
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
    print(f"  - Преобладающая профессия: {top_job}")
    print(f"  - Преобладающее сем. положение: {top_marital}")

print("\nСВЯЗЬ КЛАСТЕРОВ С ЦЕЛЕВОЙ ПЕРЕМЕННОЙ (СОГЛАСИЕ НА ВКЛАД)")

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

