import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
import numpy as np

df = pd.read_csv("results/experiments.csv")

print("Análisis de resultados:")
print(df.groupby("n_point")["F"].describe())
#print(df.groupby("n_point")["bestcp"].describe())

#Grafico de histogramas de F
plt.hist(df["F"], bins=20, edgecolor="black")
plt.xlabel("F")
plt.ylabel("Frecuencia")
plt.title("Distribución del radio de Oertel estimado")
plt.show()

#Gráfico de cajas de F por n_point
df.boxplot(column="F", by="n_point")
plt.xlabel("Número de puntos (n_point)")
plt.ylabel("F")
plt.title("Distribución de F por número de puntos")
plt.suptitle("")
plt.show()

# Prueba de hipótesis: comparar medias de F entre diferentes n_point
n_point_values = df["n_point"].unique()
for i in range(len(n_point_values)):    
    for j in range(i + 1, len(n_point_values)):
        group1 = df[df["n_point"] == n_point_values[i]]["F"]
        group2 = df[df["n_point"] == n_point_values[j]]["F"]
        t_stat, p_value = st.ttest_ind(group1, group2)
        print(f"Comparación de F entre n_point={n_point_values[i]} y n_point={n_point_values[j]}: t={t_stat:.4f}, p={p_value:.4f}")


mean_F = df["F"].mean()
sem_F = st.sem(df["F"])
ic = st.t.interval(0.95, len(df["F"])-1, loc=mean_F, scale=sem_F)
print("IC 95%:", ic)

# === 2. Agrupar por número de puntos ===
stats = df.groupby("n_point")["F"].agg(["mean", "std", "count"])
stats["sem"] = stats["std"] / np.sqrt(stats["count"])

# === 3. Datos teóricos ===
f_teor_1 = 1 / (2 * np.e)
f_teor_2 = 2 / 9

# === 4. Graficar ===
plt.figure(figsize=(6, 4))
plt.errorbar(
    stats.index, stats["mean"], yerr=stats["std"],
    fmt="o", color="blue", capsize=4, label="Promedio experimental"
)
plt.axhline(f_teor_1, color="red", linestyle="--", label=r"$1/(2e)$")
plt.axhline(f_teor_2, color="blue", linestyle="dashdot", label=r"$2/9$")

plt.xlabel("Número de puntos por fibra")
plt.ylabel(r"$F(S)$")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()