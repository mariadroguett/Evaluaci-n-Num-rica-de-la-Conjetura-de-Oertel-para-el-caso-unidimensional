def rejection_sampling(d, A, b, z, N, tol=1e-9, batch=None, target_mb=None):
    """
    Estima Vol_rel(S_z) = P[(z,p) ∈ C] con p ~ U([0,1]^d), i.e.,
    volumen relativo en la fibra z dentro de [0,1]^d.

    Parámetros
    ----------
    d         : int                  dimensión continua
    A, b      : Ax ≤ b en R^{1+d}    (normales orientadas al interior)
    z         : int/float            fibra entera (se castea a float)
    N         : int                  número total de muestras
    tol       : float                tolerancia numérica para Ax ≤ b + tol
    batch     : int/None             tamaño de lote; si None, se calcula automático
    target_mb : float/None           memoria objetivo para calcular el batch

    Retorna
    -------
    float en [0,1]: proporción aceptada (volumen relativo de S_z)
    """
    d = int(d)
    N = int(N)
    if N <= 0 or d <= 0:
        return 0.0

    A = np.asarray(A, float)
    b = np.asarray(b, float)
    z_val = float(z)

    # Verificación de dimensiones
    if A.shape[1] != 1 + d:
        raise ValueError(f"A tiene {A.shape[1]} columnas; se esperaba 1 + d = {1 + d}.")

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

    # Bucle de muestreo por lotes
    while generados < N:
        m = min(batch, N - generados)

        # Puntos aleatorios en [0,1]^d
        p_samples = np.random.rand(m, d)   # (m, d)
        lhs = p_samples @ Ap.T             # (m, #ineq)

        # Chequeo de pertenencia al poliedro
        inside = np.all(lhs <= (b_shift + tol), axis=1)

        aceptados += int(inside.sum())
        generados += m

    return aceptados / float(N)
