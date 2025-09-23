import os
import csv
from datetime import datetime
import numpy as np
from vol_reject import rejection_sampling
from vol_star import ratio_cp
from numpy.linalg import norm

def _inside(A, b, x, tol=1e-9):
    return np.all((A @ x) <= b + tol)

def ortel(
    A, b, d,
    z_vals=None,
    N_cp=50, N_hip=1000, N=80000,
    tol=1e-9, seed=None,
    batch=None, guided=False,
    max_trials_per_cp=3*10**5,
    save_dir=None,
    defer_rejection=True,
    progress=True
):
    """
    Busca un centerpoint aproximado maximizando F(cp) = (peor corte)/Vol(S).
    """
    rng = np.random.default_rng(seed)
    if z_vals is None:
        z_vals = [0, 1]
    z_vals = list(z_vals)

    # (1) Volumen total Vol(S)
    if progress:
        print("[ORTEL] Calculando volumen total por Monte Carlo...", flush=True)
    a_z = {}
    for zi in z_vals:
        seed_z = rng.integers(2**63 - 1)
        a_z[zi] = rejection_sampling(d, A, b, zi, N, tol=tol, seed=seed_z, batch=batch)
    vol_total = float(sum(a_z.values()))
    if vol_total <= 0.0:
        raise ValueError("Volumen total nulo. Revisa A,b,z_vals o aumenta N.")
    if progress:
        print(f"[ORTEL] Volumen total estimado ≈ {vol_total:.6g}", flush=True)

    bestF, bestCP = -np.inf, None
    accepted_cps = []  # lista de cps aceptados (sin F inicialmente)

    # (2) Generar candidatos y aceptar hasta tener N_cp dentro de S
    num_accepted, trials = 0, 0
    while num_accepted < N_cp and trials < N_cp * max_trials_per_cp:
        trials += 1
        zi = int(rng.choice(z_vals))
        p  = rng.random(d)
        cand = np.hstack([zi, p]).astype(float)

        if not _inside(A, b, cand, tol=tol):
            continue

        accepted_cps.append(cand.copy())
        num_accepted += 1
        if progress:
            print(f"[ORTEL] Aceptados {num_accepted}/{N_cp} (z={zi})", flush=True)

    if not accepted_cps:
        raise RuntimeError("No se aceptaron candidatos dentro de S.")

    # (3) para cada cp aceptado, hacemos rejection (ratio_cp) y calculamos F
    if defer_rejection:
        if progress:
            print(f"[ORTEL] Evaluando F(cp) para {len(accepted_cps)} candidatos...", flush=True)
        evaluated = []  # (cp, F)
        for i, cp in enumerate(accepted_cps, start=1):
            seed_ratio = rng.integers(2**63 - 1)
            peor_vol = ratio_cp(
                A, b, cp, d, z_vals, N_hip, N,
                tol=tol, seed=seed_ratio, batch=batch
            )
            F_cp = float(peor_vol / vol_total)
            evaluated.append((cp, F_cp))
            if F_cp > bestF:
                bestF, bestCP = F_cp, cp.copy()
            if progress:
                print(f"[ORTEL] Eval {i}/{len(accepted_cps)}: F≈{F_cp:.6g} | best≈{bestF:.6g}", flush=True)

        # Guardado opcional final con F
        if save_dir:
            try:
                if ts is None:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                out_path = os.path.join(save_dir, f"cps_{ts}.csv")
                header = ["z"] + [f"p{i+1}" for i in range(d)] + ["F"]
                with open(out_path, "w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow(header)
                    for cp, fval in evaluated:
                        row = [float(cp[0])] + [float(x) for x in cp[1:1+d]] + [float(fval)]
                        w.writerow(row)
            except Exception:
                pass
    else:
        # Modo antiguo: evaluar compatibilidad
        accepted = []
        for i, cp in enumerate(accepted_cps, start=1):
            seed_ratio = rng.integers(2**63 - 1)
            peor_vol = ratio_cp(
                A, b, cp, d, z_vals, N_hip, N,
                tol=tol, seed=seed_ratio, batch=batch
            )
            F_cp = float(peor_vol / vol_total)
            accepted.append((cp, F_cp))
            if F_cp > bestF:
                bestF, bestCP = F_cp, cp.copy()
            if progress:
                print(f"[ORTEL] Eval {i}/{len(accepted_cps)}: F≈{F_cp:.6g} | best≈{bestF:.6g}", flush=True)
        if save_dir:
            try:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                out_path = os.path.join(save_dir, f"cps_{ts}.csv")
                header = ["z"] + [f"p{i+1}" for i in range(d)] + ["F"]
                with open(out_path, "w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow(header)
                    for cp, fval in accepted:
                        row = [float(cp[0])] + [float(x) for x in cp[1:1+d]] + [float(fval)]
                        w.writerow(row)
            except Exception:
                pass
    if progress:
        print(f"[ORTEL] Mejor F≈{bestF:.6g} | BestCP={bestCP}", flush=True)
    return bestCP, bestF
