#!/usr/bin/env python
import os
import sys


LOCAL_PYTHON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lam_env", "bin", "python")

if (
    os.path.exists(LOCAL_PYTHON)
    and os.environ.get("LAM_ENV_REEXEC") != "1"
    and os.path.abspath(sys.executable) != os.path.abspath(LOCAL_PYTHON)
):
    os.environ["LAM_ENV_REEXEC"] = "1"
    os.execv(LOCAL_PYTHON, [LOCAL_PYTHON, os.path.abspath(__file__), *sys.argv[1:]])

print("[1/5] Starting app initialization...")
sys.stdout.flush()

try:
    print("[2/5] Importing modules...", flush=True)
    os.environ['APP_INFER'] = './configs/inference/lam-20k-8gpu.yaml'
    os.environ['APP_TYPE'] = 'infer.lam'
    os.environ['NUMBA_THREADING_LAYER'] = 'omp'

    print("[3/5] Importing app...", flush=True)
    from app_lam import launch_gradio_app

    print("[4/5] Starting Gradio...", flush=True)
    launch_gradio_app()

except Exception as e:
    print(f"ERROR: {e}", flush=True)
    import traceback
    traceback.print_exc()
