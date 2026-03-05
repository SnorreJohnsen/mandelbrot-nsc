import psutil, os, random, time, statistics

from multiprocessing import Pool 

def estimate_pi_serial(num_samples):
    inside_circle = 0
    for _ in range(num_samples):
        x, y = random.random(), random.random()
        if x*x + y*y <= 1:
            inside_circle += 1
    return 4 * inside_circle / num_samples

def estimate_pi_chunk(num_samples):
    inside_circle = 0
    for _ in range(num_samples):
        x, y = random.random(), random.random()
        if x*x + y*y <= 1:
            inside_circle += 1
    return inside_circle

def estimate_pi_parallel(num_samples, num_processes=2):
    samples_per_process = num_samples // num_processes
    tasks = [samples_per_process] * num_processes
    with Pool(processes=num_processes) as pool:
        results = pool.map(estimate_pi_chunk, tasks)
    return 4 * sum(results) / num_samples

if __name__ == '__main__':
    num_samples = 10_000_000
    t_serial = None

    for num_proc in range(1, psutil.cpu_count(logical=False) + 1):
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            pi_est = estimate_pi_parallel(num_samples, num_proc)
            times.append(time.perf_counter() - t0)
        t = statistics.median(times)

        if num_proc == 1:
            t_serial = t

        speedup = t_serial / t
        efficiency = (speedup / num_proc) * 100

        print(f"{num_proc=:2d} | workers: {t:.3f}s | {speedup=:.2f} | {efficiency=:.2f} | {pi_est=:.6f}")
        
        if not num_proc == 1:
            serial_fraction = (1/speedup - 1/num_proc) / (1 - 1/num_proc)
            print(f"{serial_fraction=} workers={num_proc}")
