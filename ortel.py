# vol_reject.py
import numpy as np

def _choose_batch(n_ineq: int, target_mb: float | None) -> int:
    """Elige tamaño de lote aproximando memoria ~ target_mb MiB para lhs."""
    if target_mb is None:
        target_mb = 64.0  # por defecto
    bytes_target = int(target_mb * 1024 * 1024)
    # lhs es (m, n_ineq) de float64 => 8 bytes * m * n_ineq
    m = bytes_target // (8 * max(1, n_ineq))
    return max(1, int(m))

def rejection_sampling(
    d: int,
    A: np.ndarray,
    b: np.ndarray,
    z: int | float,
    N: int,
    tol: float = 1e-9,
    seed: int | None = None,
    batch: int | None = None,
    target_mb: float | None = None,
) -> float:
    """
    Estima Vol_rel(S_z) = P_p[(z,p) ∈ C] con p ~ U([0,1]^d), donde C = {x: A x ≤ b}.
    - Trabaja por fibra fija z (mixto entero); sólo muestrea p en [0,1]^d.
    - Control de memoria por lotes (batch) o memoria objetivo (target_mb).

    Retorna:
        float en [0,1]: proporción aceptada en la fibra z.
    """
    d = int(d); N = int(N)
    if N <= 0 or d <= 0:
        return 0.0

    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    if A.ndim != 2 or A.shape[1] != 1 + d:
        raise ValueError(f"A debe tener 1+d columnas (tiene {A.shape[1]}), con d={d}.")

    rng = np.random.default_rng(seed)
    z_val = float(int(z))

    # Descomponer A = [A0 | Ap] y desplazar por z
    A0 = A[:, 0]
    Ap = A[:, 1:]
    b_shift = b - A0 * z_val
    n_ineq = A.shape[0]
    ApT = Ap.T  # para @ rápido

    # Lote automático si no se especifica
    if batch is None:
        m_auto = _choose_batch(n_ineq, target_mb)
        batch = min(N, max(1000, m_auto))  # buen mínimo
    else:
        batch = max(1, int(batch))

    aceptados = 0
    generados = 0

    while generados < N:
        m = min(batch, N - generados)
        # p ~ U([0,1]^d)
        P = rng.random((m, d))                 # (m, d)
        # Ap p <= b_shift + tol  ⇒  (P @ ApT) <= b_shift
        lhs = P @ ApT                           # (m, n_ineq)
        inside = np.all(lhs <= (b_shift + tol), axis=1)
        aceptados += int(inside.sum())
        generados += m

    return aceptados / float(N)
