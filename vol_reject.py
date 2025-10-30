# vol_reject.py
import numpy as np

def _choose_batch(n_ineq, target_mb=None):
    """
    Elige tamaño de batch en función del nº de desigualdades y una meta de memoria (MiB).
    Si target_mb es None, usa ~64 MiB como referencia.
    """
    if target_mb is None:
        target_mb = 64  # MiB por defecto
    bytes_target = int(target_mb * 1024 * 1024)
    # Aproximación: cada muestra multiplica contra Ap.T -> ~8 bytes * n_ineq
    m = max(1, bytes_target // (8 * max(1, n_ineq)))
    return int(m)

def rejection_sampling(d, A, b, z, N, tol=1e-9, seed=None, batch=None, target_mb=None):
    """
    Estima Vol_rel(S_z) = P[(z,p) ∈ C] con p ~ U([0,1]^d), i.e., volumen relativo en la fibra z.

    Parámetros
    ----------
    d      : int                  dimensión continua
    A, b   : Ax ≤ b en R^{1+d}    (normales orientadas al interior)
    z      : int/float            fibra entera (se castea a int)
    N      : int                  nº de muestras MC
    tol    : float                tolerancia Ax ≤ b
    seed   : int/None             semilla RNG
    batch  : int/None             tamaño de lote (si None, se calcula automático)
    target_mb : float/None        memoria objetivo para elegir batch cuando batch=None

    Retorna
    -------
    float in [0,1]: proporción aceptada (volumen relativo de S_z en [0,1]^d)
    """
    d = int(d); N = int(N)
    if N <= 0 or d <= 0:
        return 0.0

    A = np.asarray(A, float)
    b = np.asarray(b, float)
    if A.shape[1] != 1 + d:
        raise ValueError(f"A tiene {A.shape[1]} columnas; d={d} ⇒ 1+d={1+d}.")

    rng = np.random.default_rng(seed)
    z_val = float(int(z))

    # Precompute para la fibra fija z
    Ap = A[:, 1:]                  # (#ineq, d)
    b_shift = b - A[:, 0] * z_val  # (#ineq,)
    n_ineq = A.shape[0]

    # Elegir tamaño de lote
    if batch is None:
        m_auto = _choose_batch(n_ineq, target_mb=target_mb)
        batch = min(N, max(1000, m_auto))  # al menos 1000, sin pasar de N
    else:
        batch = int(batch)
        if batch <= 0:
            batch = min(N, max(1000, _choose_batch(n_ineq, target_mb=target_mb)))

    aceptados = 0
    generados = 0
    while generados < N:
        m = min(batch, N - generados)

        # p ~ U([0,1]^d)
        p_samples = rng.random((m, d))       # (m, d)
        lhs = p_samples @ Ap.T               # (m, #ineq)
        inside = np.all(lhs <= (b_shift + tol), axis=1)

        aceptados += int(inside.sum())
        generados += m

    return aceptados / float(N)
