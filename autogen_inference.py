import argparse
from tqdm import tqdm
import subprocess
import sys
import os
import json
import tempfile
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent


def main(argv=None):
    load_dotenv(ROOT / ".env", encoding="utf-8-sig")
    parser = argparse.ArgumentParser(description="Run AutoGen tasks across attacks and scenarios.")
    parser.add_argument("--attack", type=str.lower,
                        choices=["byzantine", "colluding", "contradicting", "dpi", "ipi", "impersonation"],
                        help="Attack to run (default: all attacks)")
    parser.add_argument("--scenario", choices=["education", "finance", "healthcare", "legal", "news"],
                        help="Scenario to run (default: all scenarios)")
    parser.add_argument("--config", required=True, choices=["Magentic_one", "RoundRobin", "Swarm"])
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL"),
                        help="Model name (defaults to OPENAI_MODEL in .env)")
    args = parser.parse_args(argv)
    if not args.model:
        parser.error("Set OPENAI_MODEL in .env or pass --model.")
    if not os.getenv("OPENAI_API_KEY"):
        parser.error("Set OPENAI_API_KEY in .env or the environment.")

    # Read actual filenames: news_contradinting.json has a typo in the dataset.
    paths = [path for path in sorted((ROOT / "data").glob("*/*.json"))
             if (args.attack is None or path.parent.name.lower() == args.attack)
             and (args.scenario is None or path.stem.split("_", 1)[0] == args.scenario)]
    if not paths:
        parser.error("No datasets match the selected attack and scenario.")
    datasets = [(path, json.loads(path.read_text(encoding="utf-8"))) for path in paths]
    total = sum(len(tasks) for _, tasks in datasets)
    print(f"Running {len(datasets)} datasets, {total} tasks with {args.config}.", flush=True)
    failures = []

    for path, tasks in datasets:
        attack = path.parent.name if path.parent.name in ("DPI", "IPI") else path.parent.name.lower()
        scenario = path.stem.split("_", 1)[0]
        logs_dir = ROOT / "logs_qwen" / attack / args.config
        logs_dir.mkdir(parents=True, exist_ok=True)

        for i, task in tqdm(enumerate(tasks), total=len(tasks), desc=f"{attack}/{scenario}"):
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", encoding="utf-8", delete=False) as temp_json:
                json.dump(task, temp_json)
                temp_json_path = temp_json.name

            log_file = logs_dir / f"{attack}_{scenario}_{i}.txt"
            try:
                with log_file.open("w", encoding="utf-8") as log:
                    subprocess.run(
                        [sys.executable, str(ROOT / "autogen_run.py"), "--json_data", temp_json_path,
                         "--model", args.model, "--config", args.config],
                        cwd=ROOT,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        # 子进程不会继承父进程日志文件的 encoding，需单独设置输出编码。
                        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                        check=True,
                    )
            except subprocess.CalledProcessError:
                failures.append(log_file)
                tqdm.write(f"FAILED: {log_file}; continuing with the next task.")
            finally:
                Path(temp_json_path).unlink(missing_ok=True)

    print(f"Finished: {total - len(failures)} succeeded, {len(failures)} failed, {total} total.")
    for log_file in failures:
        print(f"Failed task log: {log_file}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
