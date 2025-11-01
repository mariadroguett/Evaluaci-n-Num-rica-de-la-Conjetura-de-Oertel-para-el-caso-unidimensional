import os
import numpy as np

# Asegura que el script trabaje desde su propia carpeta
os.chdir(os.path.dirname(__file__))

print("CWD:", os.getcwd())  # Verifica dónde estás parada

data = np.load("hull_seed_23_npoint_5.npz")
print("Arrays disponibles:", data.files)
print("A shape:", data["A"])
print("b shape:", data["b"])
print("Seed:", data["seed"])
print("n_point:", data["n_point"])
print("F value:", data["F"])