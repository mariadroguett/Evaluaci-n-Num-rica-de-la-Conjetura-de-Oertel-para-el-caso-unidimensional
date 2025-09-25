import os
import json
import time
import math
import random
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
from ortel import ortel  

# ---------------------------
# CONFIG
# ---------------------------
p_fiber = [5, 8, 11, 15, 17, 20]
trunk_total = 50              # cuántos ACEPTADOS 
batch_size = 16               # cuántos casos lanzar por lote en paralelo
max_workers = os.cpu_count()  # ajústalo si usas SLURM: --cpus-per-task
out_dir = "results"
os.makedirs(out_dir, exist_ok=True)

# Parámetros por defecto que le pasarás a tu pipeline.

COMMON_PARAMS = dict(
    d=2,                 # continuo
    z_vals=[0, 1],       # discreto (n=1)
    N_cp=50,
    N_hip=500,
    N=100_000,
    tol=1e-9,
)
# ---------------------------

p_fiber = [5,8,11,15,17,20]
acepts = []
trunk_total = 50

for i in p_fiber:
    def trunk_acepts():
        """Trunk contiene los casos """

    def trunck_rejects():
        "Hace rejection sampling y devuelve los rechazados"



    
def build_case_for(p_points: int, seed: int) -> Dict[str, Any]:
    """
    Construye los argumentos para un caso: lo mínimo para llamar a `ortel`.
    Si en tu pipeline generas (A,b) o vértices antes, hazlo acá.
    Retorna un dict con todo lo necesario para `run_one_case`.
    """
    rng = np.random.default_rng(seed)
    # Ejemplo: acá podrías armar Ax<=b o vértices por fibra.
    # Como no tengo tu generador, paso solo metadata + seed.
    case = {
        "p_points": p_points,
        "seed": seed,
        # coloca aquí lo que necesite tu ortel(...):
        # "A": A, "b": b,
        # o "vertices_por_fibra": ...
    }
    return case


def call_ortel(case: Dict[str, Any]) -> Dict[str, Any]:
    """
    Llama a tu función `ortel` con lo que corresponda.
    Devuelve un dict con resultados. Incluye siempre los campos que uses en is_accepted.
    """
    # 🔧 EJEMPLO: solo paso parámetros comunes + seed a modo de “traza”.
    # Reemplaza por tu llamada real: ortel(A,b,d, z_vals=..., N_cp=..., ...)
    result = {
        "seed": case["seed"],
        "p_points": case["p_points"],
        # Coloca aquí lo que devuelva realmente ortel:
        # "F": ..., "best_cp": [...], "accepted": True/False, etc.
    }

    # -------------------------
    # DEMO: una “aceptación” falsa al azar (REEMPLAZA ESTO)
    # -------------------------
    rng = np.random.default_rng(case["seed"])
    fake_F = rng.uniform(0.2, 0.5)  # supongamos F(S)
    result["F"] = float(fake_F)
    # Si ya tienes criterio en ortel, elimina esta línea y setea result["accepted"] directo.
    result["accepted"] = (fake_F >= 0.3)
    # -------------------------

    return result


def is_accepted(result: Dict[str, Any]) -> bool:
    """
    Criterio de aceptación. Ajusta a lo que uses (por ejemplo, F >= umbral).
    Si tu `ortel` ya devuelve un flag claro, úsalo directo.
    """
    return bool(result.get("accepted", False))


# ---------------------------
# EJECUCIÓN DE UN CASO
# ---------------------------

def run_one_case(p_points: int, seed: int) -> Dict[str, Any]:
    case = build_case_for(p_points, seed)
    res = call_ortel(case)
    # agrega metadata útil
    res["timestamp"] = time.time()
    return res


# ---------------------------
# “BAÚL” DE ACEPTADOS Y RECHAZADOS
# ---------------------------

def trunk_accepts(p_points: int,
                  target_accepts: int,
                  seed0: int = 10_000,
                  batch: int = batch_size,
                  workers: Optional[int] = None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Sigue generando casos para un valor fijo de p_points hasta juntar `target_accepts` ACEPTADOS.
    Devuelve (aceptados, rechazados) con TODO el historial del proceso.
    """
    accepted: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []

    workers = workers or max(1, (os.cpu_count() or 1) - 1)
    seed = seed0
    round_id = 0

    while len(accepted) < target_accepts:
        needed = target_accepts - len(accepted)
        # Tiramos un lote “generoso”: lo que falta, pero acotado por batch
        this_batch = max(1, min(batch, needed * 2))  # *2 para amortizar rechazo

        futures = []
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for k in range(this_batch):
                futures.append(ex.submit(run_one_case, p_points, seed + k))
            seed += this_batch

            for fut in as_completed(futures):
                res = fut.result()
                if is_accepted(res):
                    accepted.append(res)
                else:
                    rejected.append(res)

                if len(accepted) >= target_accepts:
                    break  # ya llenamos el baúl de aceptados

        round_id += 1

    return accepted, rejected


def trunk_rejects(p_points: int,
                  max_rejects: int,
                  seed0: int = 50_000,
                  batch: int = batch_size,
                  workers: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Genera casos y se queda SOLO con los rechazados hasta llegar a `max_rejects`.
    Útil si quieres analizar sesgos del rechazo.
    """
    rejected: List[Dict[str, Any]] = []
    workers = workers or max(1, (os.cpu_count() or 1) - 1)
    seed = seed0

    while len(rejected) < max_rejects:
        remain = max_rejects - len(rejected)
        this_batch = max(1, min(batch, remain * 2))

        futures = []
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for k in range(this_batch):
                futures.append(ex.submit(run_one_case, p_points, seed + k))
            seed += this_batch

            for fut in as_completed(futures):
                res = fut.result()
                if not is_accepted(res):
                    rejected.append(res)
                if len(rejected) >= max_rejects:
                    break

    return rejected


# ---------------------------
# DRIVER: recorre todos los p_fiber
# ---------------------------

def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    acc_path = os.path.join(out_dir, f"baul_accepts_{ts}.jsonl")
    rej_path = os.path.join(out_dir, f"baul_rejects_{ts}.jsonl")

    total_ok = 0
    total_seen = 0

    with open(acc_path, "w", encoding="utf-8") as f_acc, open(rej_path, "w", encoding="utf-8") as f_rej:
        for p in p_fiber:
            print(f"→ Llenando baúl para p={p} hasta {trunk_total} aceptados…")
            accepts, rejects = trunk_accepts(p, trunk_total)

            # guarda aceptados
            for r in accepts:
                r_out = dict(r)
                r_out["p_fiber"] = p
                f_acc.write(json.dumps(r_out, ensure_ascii=False) + "\n")

            # guarda rechazados (opcional: guarda todos o un muestreo)
            for r in rejects:
                r_out = dict(r)
                r_out["p_fiber"] = p
                f_rej.write(json.dumps(r_out, ensure_ascii=False) + "\n")

            total_ok += len(accepts)
            total_seen += len(accepts) + len(rejects)
            tasa = total_ok / max(1, total_seen)
            print(f"   ✓ p={p}: aceptados={len(accepts)}, rechazados={len(rejects)}, tasa_acumulada≈{tasa:.3f}")

    print("\nListo ✨")
    print(f"Aceptados:  {acc_path}")
    print(f"Rechazados: {rej_path}")


if __name__ == "__main__":
    main()



