import numpy as np
from numpy.linalg import norm
from vol_reject import rejection_sampling 

def ratio_cp(
    A, b, cp_full, d, z_vals, N_hip, N,
    tol=1e-9, seed=None, batch=None, guided=False
):
    """
    PEOR volumen NO normalizado de cortes H que pasan por cp.
    - a_z = Vol(S_z) con rejection_sampling(...)
    - Para cada dirección u: añadimos -u^T x <= -u^T cp, re-muestreamos y
      sumamos a_z * min(frac, 1-frac). Tomamos el mínimo sobre u.
    """
    rng = np.random.default_rng(seed)

    cp_full = np.asarray(cp_full, float)
    d = int(d); N = int(N); N_hip = int(N_hip)
    z_vals = list(z_vals)
    if cp_full.size != 1 + d:
        raise ValueError(f"cp_full debe tener tamaño 1+d={1+d}, recibido {cp_full.size}.")

    # 1) Volúmenes por fibra (a_z)
    a_z = {}
    for zi in z_vals:
        seed_z = rng.integers(2**63 - 1)
        a_z[zi] = rejection_sampling(d, A, b, zi, N, tol=tol, seed=seed_z, batch=batch)
    volS = sum(a_z.values())
    if volS <= 0.0:
        return 0.0

    # 2) Direcciones 
    def _rand_u():
        v = rng.integers(0, 2, size=1 + d) * 2 - 1
        u = v.astype(float)
        u /= (norm(u) + 1e-12)
        return u

    directions = [_rand_u() for _ in range(N_hip)]

    # 3) Evaluación por dirección
    worst = float('inf')
    for u in directions:
        A_aug = np.vstack([A, -u.reshape(1, -1)])
        b_aug = np.hstack([b, -float(u @ cp_full)])

        val_u = 0.0
        for zi in z_vals:
            if a_z[zi] <= 0.0:
                continue
            seed_cut = rng.integers(2**63 - 1)
            vol_cut  = rejection_sampling(d, A_aug, b_aug, zi, N, tol=tol, seed=seed_cut, batch=batch)
            frac     = vol_cut / (a_z[zi] + 1e-12)
            val_u   += a_z[zi] * min(frac, 1.0 - frac)

        if val_u < worst:
            worst = val_u
            if worst <= 1e-12:  # early exit si ya es prácticamente 0
                break

    return float(worst)  # normaliza afuera: F(cp)=worst / sum(a_z)
