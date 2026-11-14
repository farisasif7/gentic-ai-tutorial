"""
run_all.py -- runs the orchestrator over every sample report in reports/,
printing progress and saving one transcript file per report into runs/.

python run_all.py
"""
import glob
import os
from datetime import datetime

from orchestrator import run_orchestrator

HERE = os.path.dirname(__file__)
REPORTS_DIR = os.path.join(HERE, "reports")
RUNS_DIR = os.path.join(HERE, "runs")


def run_one(report_path, stamp):
    name = os.path.splitext(os.path.basename(report_path))[0]
    with open(report_path, "r", encoding="utf-8") as f:
        report_text = f.read()

    lines = [f"=== Run for {name} ({stamp}) ===", "", "--- Medical report ---", report_text, ""]
    lines.append("--- Orchestrator trace ---")

    def log(line):
        print(line)
        lines.append(line)

    print(f"\nRunning orchestrator on {name} ...")
    final_answer, _ = run_orchestrator(report_text, log=log)

    lines.append("")
    lines.append("--- Final summary ---")
    lines.append(final_answer)
    print("Final summary:\n" + final_answer)

    out_path = os.path.join(RUNS_DIR, f"{stamp}_{name}.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved run transcript -> {out_path}")


def main():
    os.makedirs(RUNS_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_paths = sorted(glob.glob(os.path.join(REPORTS_DIR, "*.txt")))
    if not report_paths:
        print(f"No reports found in {REPORTS_DIR}")
        return
    for path in report_paths:
        run_one(path, stamp)


if __name__ == "__main__":
    main()
