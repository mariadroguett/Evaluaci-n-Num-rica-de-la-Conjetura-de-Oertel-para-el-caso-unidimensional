import numpy as np

def _choose_batch(n_ineq, target_mb=None):
    """
    Elige un tamaño de lote m para que lhs ≈ (m × n_ineq).
    Aproximación: float64 -> 8 bytes por entrada.
    """
    bytes_target = int(target_mb * 1024 * 1024)
    # Evita división por cero y asegura al menos 1
    m = max(1, bytes_target // (8 * max(1, n_ineq)))
    return int(m)

def rejection_sampling(d, A, b, z, N, tol=1e-9, seed=None, batch=None, target_mb=None):
    """
    Estima Vol_rel(S_z) = P[(z,p) ∈ C] con p ~ U([0,1]^d).
    - Evita construir X_full (usa Ap y b_shift).
    - Controla memoria con lotes. Si batch=None, elige un tamaño seguro.

    Parámetros
    ----------
    d      : int
    A, b   : Ax ≤ b en R^{1+d}
    z      : fibra entera
    N      : nº de muestras Monte Carlo
    tol    : tolerancia Ax ≤ b
    seed   : semilla RNG (None -> aleatorio)
    batch  : tamaño de lote; si None, se elige automáticamente
    target_mb : memoria objetivo para lhs en MiB (si batch=None)

    Retorna
    -------
    float : proporción aceptada (volumen relativo de S_z en [0,1]^d)
    """
    # --- blindajes ---
    d = int(d); N = int(N)
    if N <= 0 or d <= 0:
        return 0.0

    A = np.asarray(A, float)
    b = np.asarray(b, float)
    if A.shape[1] != 1 + d:
        raise ValueError(f"A tiene {A.shape[1]} columnas; d={d} ⇒ 1+d={1+d}.")

    rng = np.random.default_rng(seed)
    z_val = float(int(z))

    # Precompute para z fijo
    Ap = A[:, 1:]                  # (#ineq, d)
    b_shift = b - A[:, 0] * z_val  # (#ineq,)
    n_ineq = A.shape[0]

    # --- tamaño de lote ---
    if batch is None:
        # elige m para ~ target_mb MiB y pon un mínimo razonable
        m_auto = _choose_batch(n_ineq, target_mb=target_mb)
        batch = min(N, max(1000, m_auto))   # al menos 1000, pero no más que N
    else:
        batch = int(batch)
        if batch <= 0:
            batch = min(N, max(1000, _choose_batch(n_ineq, target_mb=target_mb)))

    aceptados = 0
    generados = 0
    while generados < N:
        m = min(batch, N - generados)

        # p ~ U([0,1]^d)
        p_samples = rng.random((m, d))         # (m, d)

        # Ax <= b  <=>  Ap p <= b - A0 z
        lhs = p_samples @ Ap.T                 # (m, #ineq)
        inside = np.all(lhs <= (b_shift + tol), axis=1)  # (m,)

        aceptados += int(inside.sum())
        generados += m

    return aceptados / float(N)
