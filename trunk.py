import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

def baul_rejection(
    d,
    A,
    b,
    z,
    into,
    N,
    tol: float = 1e-9,
    seed: int | None = None,
    max_rounds: int | None = None,
    max_draws: int | None = None,
):
    """
    Wrapper compatible que delega en baul_rejection_parallel con un solo worker.
    Conserva la firma anterior (N se usa como chunk_N). max_rounds no aplica y se ignora.
    """
    # max_rounds queda obsoleto en el motor paralelo; se mantiene por compatibilidad
    return baul_rejection_parallel(
        d, A, b, z,
        into=into,
        chunk_N=int(N),
        tol=tol,
        seed=seed,
        max_workers=1,
        max_draws=max_draws,
    )


__all__ = ["baul_rejection"]


def _worker_chunk(seed: int, d: int, Ap: np.ndarray, b_shift: np.ndarray, tol: float, draws: int, take_limit: Optional[int]) -> np.ndarray:
    rng = np.random.default_rng(seed)
    p = rng.random((int(draws), int(d)))
    inside = np.all(p @ Ap.T <= (b_shift + tol), axis=1)
    pts = p[inside]
    if take_limit is not None and pts.shape[0] > int(take_limit):
        pts = pts[: int(take_limit)]
    return pts


def baul_rejection_parallel(
    d,
    A,
    b,
    z,
    into,
    chunk_N,
    tol: float = 1e-9,
    seed: int | None = None,
    max_workers: Optional[int] = None,
    max_draws: int | None = None,
):
    """
    Variante paralela: lanza tareas en chunks de `chunk_N` muestras hasta llenar el baúl.

    - Crea hasta `max_workers` procesos. Cada tarea genera `chunk_N` puntos y devuelve los aceptados.
    - El coordinador agrega resultados y sigue lanzando tareas hasta alcanzar `into` o `max_draws`.

    Retorna (pts, stats) con el mismo formato que baul_rejection.
    """
    d = int(d); into = int(into); chunk_N = int(chunk_N)
    if d <= 0 or into <= 0 or chunk_N <= 0:
        return np.empty((0, d), float), dict(
            into_target=into, into_got=0, rounds=0, draws=0, accepts=0,
            vol_est=0.0, stopped_reason="invalid-args",
        )

    A = np.asarray(A, float)
    b = np.asarray(b, float)
    if A.shape[1] != 1 + d:
        raise ValueError(f"A tiene {A.shape[1]} columnas; d={d} ⇒ 1+d={1+d}.")

    z_val = float(int(z))
    Ap = A[:, 1:]
    b_shift = b - A[:, 0] * z_val

    # RNG maestro para semillas de workers
    rng = np.random.default_rng(seed)

    bucket = []
    tasks_in_flight = {}
    accepts_total = 0
    draws_total = 0
    rounds = 0
    stopped_reason = "filled"

    # Número de workers
    if max_workers is None:
        import os as _os
        max_workers = max(1, _os.cpu_count() or 1)

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        # función para lanzar una nueva tarea si corresponde
        def submit_one():
            nonlocal draws_total, rounds
            if max_draws is not None and draws_total >= int(max_draws):
                return False
            # Ajustar tamaño de chunk si se acerca a max_draws
            this_N = chunk_N
            if max_draws is not None:
                remaining = int(max_draws) - draws_total
                if remaining <= 0:
                    return False
                this_N = min(this_N, remaining)
            seed_w = int(rng.integers(2**63 - 1))
            fut = ex.submit(_worker_chunk, seed_w, d, Ap, b_shift, tol, this_N, into - len(bucket))
            tasks_in_flight[fut] = this_N
            draws_total += int(this_N)
            rounds += 1
            return True

        # Inicial: llenar cola de tareas
        for _ in range(max_workers):
            if len(bucket) >= into:
                break
            if not submit_one():
                break

        # Bucle de recolección
        while tasks_in_flight and len(bucket) < into:
            for fut in as_completed(list(tasks_in_flight.keys()), timeout=None):
                this_N = tasks_in_flight.pop(fut)
                try:
                    pts = fut.result()
                    if pts.size:
                        bucket.extend(pts)
                        accepts_total += int(pts.shape[0])
                except Exception:
                    # ignora fallo puntual y continúa
                    pass
                if len(bucket) >= into:
                    break
                submit_one()

        # recorta si sobran
        if len(bucket) > into:
            bucket = bucket[:into]

    vol_est = (accepts_total / float(draws_total)) if draws_total > 0 else 0.0
    if len(bucket) < into:
        if max_draws is not None and draws_total >= int(max_draws):
            stopped_reason = "max_draws"
        else:
            stopped_reason = "exhausted"

    pts = np.asarray(bucket, dtype=float)
    stats = dict(
        into_target=into,
        into_got=int(pts.shape[0]),
        rounds=int(rounds),
        draws=int(draws_total),
        accepts=int(accepts_total),
        vol_est=float(vol_est),
        stopped_reason=stopped_reason,
        parallel=True,
        chunk_N=chunk_N,
        max_workers=int(max_workers),
    )
    return pts, stats
