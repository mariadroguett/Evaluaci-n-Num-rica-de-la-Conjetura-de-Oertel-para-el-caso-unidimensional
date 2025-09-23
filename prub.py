from concurrent.futures import ProcessPoolExecutor, as_completed

def pesado(x):
    # cálculo CPU-intensivo
    return x, x**2

nums = [5, 8, 11, 14]

with ProcessPoolExecutor() as ex:
    futures = [ex.submit(pesado, n) for n in nums]
    for fut in as_completed(futures):
        n, r = fut.result()     # puede levantar la excepción si falló
        print(n, r)             # llega el que terminó primero
