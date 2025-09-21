import numpy as np
from vol_star import ratio_cp
from numpy.linalg import norm
from trunk import baul_rejection_parallel

def _inside(A, b, x, tol=1e-9):
    return np.all((A @ x) <= b + tol)

def ortel(
    A, b, d,
    z_vals=None,
    N_cp=50, N_hip=1000, N=80000,
    tol=1e-9, seed=None,
    batch=None, guided=False,
    max_trials_per_cp=200,
    into_baul: int = 5000,
    use_parallel: bool = False,
    chunk_N: int | None = None,
    max_workers: int | None = None,
    max_draws: int | None = None,
):
    """
    Busca un centerpoint aproximado maximizando F(cp) = (peor corte)/Vol(S).
    """
    rng = np.random.default_rng(seed)
    if z_vals is None:
        z_vals = [0, 1]
    z_vals = list(z_vals)

    # (1) Volumen total Vol(S) (estimación por baúl por fibra) usando motor paralelo unificado
    a_z = {}
    chunkN = (chunk_N or N)
    workers = (max_workers if use_parallel else 1)
    for zi in z_vals:
        seed_z = rng.integers(2**63 - 1)
        _, stats_z = baul_rejection_parallel(
            d, A, b, zi,
            into=into_baul,
            chunk_N=int(chunkN),
            tol=tol,
            seed=int(seed_z),
            max_workers=workers,
            max_draws=max_draws,
        )
        a_z[zi] = float(stats_z.get('vol_est', 0.0))
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
            tol=tol, seed=seed_ratio, batch=batch, guided=guided,
            into_baul=into_baul,
            use_parallel=use_parallel,
            chunk_N=chunk_N,
            max_workers=max_workers,
            max_draws=max_draws,
        )
        F_cand = peor_vol / vol_total
        if F_cand > bestF:
            bestF, bestCP = float(F_cand), cand.copy()
        num_tested += 1

    if bestCP is None:
        raise RuntimeError("No se aceptaron candidatos dentro de S. Revisa la geometría o aumenta max_trials_per_cp.")
    return bestCP, bestF
