# ortel.py
import numpy as np
from numpy.linalg import norm
from typing import Dict, List, Tuple, Optional

from vol_reject import rejection_sampling
from vol_star import ratio_cp


def _inside(A: np.ndarray, b: np.ndarray, x: np.ndarray, tol: float = 1e-9) -> bool:
    """
    Chequea si x cumple Ax <= b (con tolerancia).
    """
    return bool(np.all(A @ x <= b + tol))


def ortel(
    A: np.ndarray,
    b: np.ndarray,
    d: int,
    z_vals: Optional[List[int]] = None,
    N_cp: int = 50,         # nº de CPs candidatos
    N_hip: int = 1000,      # nº de hiperplanos aleatorios para evaluar "peor corte"
    N: int = 80_000,        # nº de muestras MC para volúmenes relativos
    tol: float = 1e-9,
    seed: Optional[int] = None,
    batch: Optional[int] = None,
    guided: bool = False,   # mantenido por compatibilidad; no usamos heurística especial aquí
) -> Tuple[np.ndarray, float]:
    """
    Busca un centerpoint aproximado maximizando F(cp) = (peor corte) / Vol(S).

    Parámetros
    ----------
    A, b     : descripción Ax <= b de la envolvente convexa en R^{1+d},
               donde la primera coordenada es z (discreta) y las d restantes son continuas.
    d        : dimensión continua.
    z_vals   : valores enteros de z (fibras). Si None -> [0, 1].
    N_cp     : nº de candidatos cp a evaluar.
    N_hip    : nº de hiperplanos aleatorios para estimar el "peor corte".
    N        : nº de muestras Monte Carlo para estimaciones de volumen.
    tol      : tolerancia numérica para Ax <= b.
    seed     : semilla RNG global (reproducibilidad).
    batch    : tamaño de lote para rejection_sampling (mem-friendly).
    guided   : parámetro decorativo (no se usa una guía especial acá).

    Retorna
    -------
    bestCP : np.ndarray de shape (1+d,)
    bestF  : float
    """
    rng = np.random.default_rng(seed)

    # -------- fibras (z) --------
    if z_vals is None:
        z_vals = [0, 1]
    z_vals = [int(z) for z in z_vals]
    z_vals_arr = np.array(z_vals, dtype=int)

    # -------- chequeos básicos --------
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    if A.ndim != 2 or b.ndim != 1 or A.shape[0] != b.shape[0] or A.shape[1] != 1 + d:
        raise ValueError(
            f"Dimensiones incompatibles: A {A.shape}, b {b.shape}, d={d} (esperado A[:, :].shape[1] = 1+d)."
        )

    # -------- Vol(S) por fibras: a_z[z] = P[(z,p) ∈ C] con p ~ U([0,1]^d) --------
    # (Esto se usa dentro de ratio_cp típicamente para normalizar cortes;
    #  si tu ratio_cp no los usa internamente, igual es útil tenerlos estimados.)
    a_z: Dict[int, float] = {}
    for zi in z_vals:
        seed_z = int(rng.integers(2**63 - 1))
        a_z[zi] = rejection_sampling(
            d, A, b, zi, N, tol=tol, seed=seed_z, batch=batch
        )

    # -------- búsqueda de CP --------
    bestF: float = -np.inf
    bestCP: Optional[np.ndarray] = None

    for _ in range(N_cp):
        # muestreamos un cp candidato adentro del dominio z × [0,1]^d
        # estrategia simple: z al azar y p ~ U([0,1]^d)
        z_cp = int(rng.choice(z_vals_arr))
        p_cp = rng.random(d)  # (d,)
        cp = np.concatenate([[float(z_cp)], p_cp]).astype(float)

        # Si no cae dentro de la envolvente, lo descartamos rápido (no debería pasar a menudo)
        if not _inside(A, b, cp, tol=tol):
            continue

        # OJO: ratio_cp de tu vol_star.py requiere N explícito (esta era la causa del error).
        # Firma esperada: ratio_cp(A, b, cp, z_vals, N_hip, d, N, tol=..., seed=...)
        F = ratio_cp(
            A, b, cp, z_vals, N_hip, d, N,
            tol=tol,
            seed=int(rng.integers(2**63 - 1))
        )

        if F > bestF:
            bestF = float(F)
            bestCP = cp.copy()

    # si nunca mejoró (muy raro), devolvemos algún punto válido (centroide seguro)
    if bestCP is None:
        # fallback: promedio de un par de puntos internos simples
        # intentamos varias veces encontrar algo interior
        for _ in range(1000):
            z_cp = int(rng.choice(z_vals_arr))
            p_cp = rng.random(d)
            cp_try = np.concatenate([[float(z_cp)], p_cp])
            if _inside(A, b, cp_try, tol=tol):
                bestCP = cp_try
                bestF = ratio_cp(
                    A, b, bestCP, z_vals, N_hip, d, N,
                    tol=tol,
                    seed=int(rng.integers(2**63 - 1))
                )
                break

    if bestCP is None:
        # Último recurso: devolver algo consistente
        bestCP = np.zeros(1 + d, dtype=float)

    return bestCP, float(bestF)
