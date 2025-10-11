import math, random, time
from concurrent.futures import ProcessPoolExecutor


def pi_part(samples: int) -> int:
    inside = 0
    rnd = random.Random()  # 每个进程独立的随机数生成器
    for _ in range(samples):
        x, y = rnd.random(), rnd.random()
        inside += (x * x + y * y) <= 1.0
    return inside

def cpu_demo():
    workers = 24
    samples_per_worker = 50000000
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(pi_part, samples_per_worker) for _ in range(workers)]
        inside_total = sum(f.result() for f in futures)
    pi_estimate = 4 * inside_total / (samples_per_worker * workers)
    return pi_estimate

if __name__ == "__main__":
    t0 = time.time()
    print("π ≈", cpu_demo())
    print(f"CPU 并行总耗时：{time.time() - t0:.2f}s")
