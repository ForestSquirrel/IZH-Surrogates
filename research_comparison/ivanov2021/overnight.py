"""
overnight.py -- the zoo comparison on MNIST: identical topology and input code, only the neuron
model swapped.

    conditions  lif | <zoo>@fastin | sur:<zoo>@fastin   (zoo: hh, ml_c1, adex_adapting)
    1. static   campaign.py --stage final: sweep the surrogate conditions, extend sweep optima
                that sit on a grid edge, full runs (3 nets, 50k/10k/10k), confusion matrices
    2. figures  figs.py
Every step resumes from what is on disk. Logs: data/overnight/<step>.log.

    PY=python
    $PY overnight.py --plan          # print the steps and the ETA, run nothing
    $PY overnight.py                 # run everything (~3 h)
    $PY overnight.py --only figures  # one phase
"""
import os, sys, time, argparse, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
ZOO = ("hh", "ml_c1", "adex_adapting")
FINAL = ["lif"] + [f"{z}@fastin" for z in ZOO] + [f"sur:{z}@fastin" for z in ZOO]   # gain-matched surrogates
LOGS = os.path.join(HERE, "data", "overnight")
STATIC_ETA_MIN = 180


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(os.path.join(LOGS, "overnight.log"), "a") as f:
        f.write(line + "\n")


def run(cmd, name):
    lf = open(os.path.join(LOGS, f"{name}.log"), "a")
    log(f"start {name}: {' '.join(cmd)}")
    rc = subprocess.Popen(cmd, cwd=HERE, stdout=lf, stderr=subprocess.STDOUT).wait()
    log(f"end {name}: exit {rc}")
    return rc


SMOKE = False
ROOT = None                 # --root: campaign_mnist/ under it (smoke rehearsal)


def campaign_dir():
    return os.path.join(ROOT, "campaign_mnist") if ROOT else os.path.join(HERE, "data", "campaign_mnist")


def static():
    cmd = [PY, "-u", "campaign.py", "--stage", "final", "--out", campaign_dir()]
    return run(cmd + (["--smoke"] if SMOKE else []), "static")


def figures():
    extra = ["--campaign", campaign_dir()] + (["--out", os.path.join(ROOT, "out")] if ROOT else [])
    return run([PY, "-u", "figs.py"] + extra, "figs")


def plan():
    print("conditions:", ", ".join(FINAL))
    print(f"1. static : campaign --stage final                            ~{STATIC_ETA_MIN / 60:.1f} h")
    print("2. figures: figs.py                                              ~10 min")
    print(f"total ~{(STATIC_ETA_MIN + 10) / 60:.1f} h; outputs data/campaign_mnist, figs/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--only", choices=("static", "figures"))
    ap.add_argument("--smoke", action="store_true", help="smoke sizes (use with --root <scratch>)")
    ap.add_argument("--root", default=None)
    a = ap.parse_args()
    if a.plan:
        return plan()
    global SMOKE, ROOT, LOGS
    SMOKE, ROOT = a.smoke, a.root
    if a.smoke and not a.root:
        raise SystemExit("--smoke writes into --root only (never into data/)")
    if ROOT:
        LOGS = os.path.join(ROOT, "logs")
    os.makedirs(LOGS, exist_ok=True)
    t0 = time.time()
    for name, fn in [("static", static), ("figures", figures)]:
        if a.only and name != a.only:
            continue
        if fn() and name == "static":
            log("static stage failed; stopping before the figures")
            return 1
    log(f"overnight done in {(time.time() - t0) / 3600:.2f} h")


if __name__ == "__main__":
    sys.exit(main())
