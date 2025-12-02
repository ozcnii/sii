
# ЭТАП 7: РЕПРОДУЦИРУЕМОСТЬ

print("\nЭТАП 7: РЕПРОДУЦИРУЕМОСТЬ")

print(f"\nФиксированный random_state")
print(f"Для воспроизводимости использован: random_state = {RANDOM_STATE}")

print("\nВерсии библиотек")
print(f"  - Python:       {platform.python_version()}")
print(f"  - pandas:       {pd.__version__}")
print(f"  - numpy:        {np.__version__}")
print(f"  - seaborn:      {sns.__version__}")
print(f"  - scikit-learn: {sklearn.__version__}")
print(f"  - matplotlib:   {plt.matplotlib.__version__}")

print("\nИнформация о системе")
print(f"  - ОС: {platform.system()} {platform.release()}")
print(f"  - Архитектура: {platform.machine()}")

df_results = df.copy()
df_results["cluster"] = best_labels
df_results.to_csv("clustered_clients.csv", index=False)
print("\nРезультаты сохранены в 'clustered_clients.csv'")
