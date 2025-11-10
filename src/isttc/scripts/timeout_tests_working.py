# timeout_tests_working.py
import time
import multiprocessing as mp

# --- top-level worker (must NOT be nested) ---
def _wrap_fn(q, fn, a):
    try:
        q.put(("ok", fn(a)))
    except Exception as e:
        q.put(("err", repr(e)))

def run_with_timeout(fn, arg, timeout_s=5):
    ctx = mp.get_context("spawn")   # safest on Windows/Jupyter
    q = ctx.Queue()
    p = ctx.Process(target=_wrap_fn, args=(q, fn, arg))
    p.start()
    p.join(timeout_s)

    if p.is_alive():
        p.terminate()
        p.join()
        print(f"Timed out after {timeout_s} sec.")
        return None

    status, payload = q.get()
    if status == "ok":
        return payload
    else:
        raise RuntimeError(payload)

# --- simple long task (top-level) ---
def long_task(x):
    print(f"Starting task {x}")
    time.sleep(x)
    print(f"Finished task {x}")
    return x * 2

if __name__ == "__main__":
    mp.freeze_support()  # required on Windows when spawning

    for k in [1, 3, 7]:
        result = run_with_timeout(long_task, k, timeout_s=5)
        if result is None:
            print(f"Trial {k} skipped (too slow)\n")
            continue
        print(f"Trial {k} done → result = {result}\n")

