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
    max_trials_per_cp=200
):
    """
    Busca un centerpoint aproximado maximizando F(cp) = (peor corte)/Vol(S).
    """
    rng = np.random.default_rng(seed)
    if z_vals is None:
        z_vals = [0, 1]
    z_vals = list(z_vals)

    # (1) Volumen total Vol(S)
    a_z = {}
    for zi in z_vals:
        seed_z = rng.integers(2**63 - 1)
        a_z[zi] = rejection_sampling(d, A, b, zi, N, tol=tol, seed=seed_z, batch=batch)
    vol_total = float(sum(a_z.values()))
    if vol_total <= 0.0:
        raise ValueError("Volumen total nulo. Revisa A,b,z_vals o aumenta N.")

    bestF, bestCP = -np.inf, None

    # (2) Generar y evaluar N_cp candidatos dentro de S
    num_tested, trials = 0, 0
    while num_tested < N_cp and trials < N_cp * max_trials_per_cp:
        trials += 1
        zi = int(rng.choice(z_vals))
        p  = rng.random(d)
        cand = np.hstack([zi, p]).astype(float)

        if not _inside(A, b, cand, tol=tol):
            continue

        seed_ratio = rng.integers(2**63 - 1)
        peor_vol = ratio_cp(
            A, b, cand, d, z_vals, N_hip, N,
            tol=tol, seed=seed_ratio, batch=batch, guided=guided
        )
        F_cand = peor_vol / vol_total
        if F_cand > bestF:
            bestF, bestCP = float(F_cand), cand.copy()
        num_tested += 1

    if bestCP is None:
        raise RuntimeError("No se aceptaron candidatos dentro de S. Revisa la geometría o aumenta max_trials_per_cp.")
    return bestCP, bestF