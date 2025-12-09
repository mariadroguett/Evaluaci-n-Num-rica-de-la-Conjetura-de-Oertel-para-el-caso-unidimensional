#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
analisis_resultados.py

Lee todos los archivos result_*.npz dentro de results/** (hulls y hulls_obs),
construye un DataFrame con:
    - n_per_z
    - F
    - bestcp (vector 3D sin separarlo)
    - best_u (vector de dimensión d, sin separarlo)

Luego calcula estadísticas y crea dos gráficos:
    1) Histograma por n_per_z con transparencias
    2) Boxplot por n_per_z estilo limpio
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# === Estilo bonito para gráficos ===
plt.style.use("seaborn-v0_8-muted")

# === Configuración de paths ===
BASE = Path("/home/maria/Evaluaci-n-Num-rica-de-la-Conjetura-de-Oertel-para-el-caso-unidimensional")

# Ahora buscamos en TODO results (hulls + hulls_obs)
RESULTS_DIR = BASE / "results"

rows = []
n_files = 0
n_ok = 0
n_err = 0

print(f"Buscando archivos result_*.npz en {RESULTS_DIR} ...")

# === Leer todos los .npz (tanto de hulls como de hulls_obs) ===
for npz_path in RESULTS_DIR.rglob("result_*.npz"):
    n_files += 1

    try:
        data = np.load(npz_path, allow_pickle=True)

        # Solo archivos de PUNTOS, no vértices
        if "F" not in data or "bestcp" not in data:
            continue

        F = float(data["F"])
        n_per_z = int(data["n_per_z"])
        bestcp = np.array(data["bestcp"], dtype=float)

        if bestcp.shape != (3,):
            raise ValueError(f"bestcp tiene shape raro {bestcp.shape}")

        # best_u: puede no existir en archivos viejos
        if "best_u" in data:
            best_u = np.array(data["best_u"], dtype=float)
        else:
            best_u = None  # para que no explote con archivos antiguos

        rows.append({
            "file": str(npz_path),
            "folder": npz_path.parent.name,
            "n_per_z": n_per_z,
            "F": F,
            "bestcp": bestcp,
            "best_u": best_u,
        })

        n_ok += 1

    except Exception as e:
        print(f"❌ Error leyendo {npz_path}: {e}")
        n_err += 1


# === Crear DataFrame ===
df = pd.DataFrame(rows)

print(f"\nArchivos encontrados (result_*.npz): {n_files}")
print(f"Archivos válidos (con F y bestcp): {n_ok}")
print(f"Archivos con error: {n_err}")

if df.empty:
    print("⚠️ No se encontró ningún archivo válido con F y bestcp.")
    raise SystemExit(0)


# === Estadísticas numéricas ===

print("\nPrimeras filas:")
print(df.head())

print("\nNaN por columna (solo n_per_z y F):")
print(df[["n_per_z", "F"]].isna().sum())

print("\n.describe() global (n_per_z y F):")
print(df[["n_per_z", "F"]].describe())

print("\n.describe() de F por n_per_z:")
print(df.groupby("n_per_z")["F"].describe())


# === Guardar CSV con todas las columnas (incluye bestcp y best_u) ===
out_csv = BASE / "results" / "analisis_resultados_simple.csv"
df.to_csv(out_csv, index=False)
print(f"\n✅ CSV guardado en: {out_csv}")


# ============================================================
#  📌 GRÁFICO 1 — Histograma por n_per_z
# ============================================================

plt.figure(figsize=(12, 6))

unique_n = sorted(df["n_per_z"].unique())
colors = ["#ffcc66", "#66b3ff", "#66cc99", "#ffdd77", "#6699cc"]

for i, n in enumerate(unique_n):
    subset = df[df["n_per_z"] == n]["F"]
    plt.hist(
        subset,
        bins=20,
        alpha=0.45,
        color=colors[i % len(colors)],
        label=f"n={n}",
    )

# Líneas verticales teóricas
plt.axvline(1 / (2 * np.e), color="red", linestyle="--", linewidth=2,
            label="1/(2e) ≈ 0.184")
plt.axvline(2 / 9, color="green", linestyle="--", linewidth=2,
            label="2/9 ≈ 0.222")

plt.title("Distribución de $F(S)$ por número de puntos por fibra", fontsize=16)
plt.xlabel("Radio estimado $F(S)$", fontsize=14)
plt.ylabel("Frecuencia", fontsize=14)
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
#  📌 GRÁFICO 2 — Boxplot por n_per_z
# ============================================================

plt.figure(figsize=(11, 6))

df.boxplot(
    column="F",
    by="n_per_z",
    grid=True,
    boxprops=dict(color="navy"),
    medianprops=dict(color="red"),
)

plt.axhline(1 / (2 * np.e), color="red", linestyle="--", linewidth=2,
            label="1/(2e) ≈ 0.184")
plt.axhline(2 / 9, color="green", linestyle="--", linewidth=2,
            label="2/9 ≈ 0.222")

plt.title("Variabilidad de $F(S)$ según $n_{per_z}$", fontsize=15)
plt.suptitle("")
plt.xlabel("Número de puntos por fibra $n_{per_z}$", fontsize=14)
plt.ylabel("Radio estimado $F(S)$", fontsize=14)
plt.legend()
plt.tight_layout()
plt.show()
