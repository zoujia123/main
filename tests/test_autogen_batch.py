import asyncio
import contextlib
import io
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

import autogen_inference


ROOT = Path(__file__).resolve().parents[1]


class AutoGenBatchTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        for path in (ROOT / "data").glob("*/*.json"):
            target = self.root / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)

    def run_batch(self, *args, fail_first=False, remove_temp=False):
        calls = []
        output = io.StringIO()

        def run(command, *, cwd, stdout, stderr, env, check):
            task_path = Path(command[command.index("--json_data") + 1])
            task = json.loads(task_path.read_text(encoding="utf-8"))
            self.assertIn("agents", task)
            self.assertIn("user query", task)
            self.assertEqual(command[0], sys.executable)
            self.assertEqual(Path(command[1]), self.root / "autogen_run.py")
            self.assertEqual(command[command.index("--model") + 1], "test-model")
            self.assertEqual(command[command.index("--config") + 1], "Magentic_one")
            self.assertEqual(cwd, self.root)
            self.assertEqual(stderr, subprocess.STDOUT)
            self.assertTrue(check)
            calls.append((Path(stdout.name), task_path))
            stdout.write("simulated task output\n")
            if remove_temp and len(calls) == 1:
                task_path.unlink()
            if fail_first and len(calls) == 1:
                raise subprocess.CalledProcessError(1, command)

        with patch.object(autogen_inference, "ROOT", self.root), \
             patch.object(autogen_inference, "load_dotenv"), \
             patch.dict(os.environ, {"OPENAI_MODEL": "test-model", "OPENAI_API_KEY": "test-key"}), \
             patch.object(autogen_inference.subprocess, "run", side_effect=run), \
             patch.object(autogen_inference, "tqdm", side_effect=lambda items, **kwargs: items), \
             contextlib.redirect_stdout(output):
            status = autogen_inference.main(["--config", "Magentic_one", *args])
        self.assertTrue(all(not temporary.exists() for _, temporary in calls))
        self.assertTrue(all(log.read_text(encoding="utf-8") == "simulated task output\n"
                            for log, _ in calls))
        return status, [log for log, _ in calls], output.getvalue()

    def test_all_attacks_and_scenarios(self):
        status, logs, output = self.run_batch()
        self.assertEqual(status, 0)
        self.assertEqual(len(logs), 300)
        self.assertEqual(len(set(logs)), 300)
        combinations = {(log.parent.parent.name, log.stem.split("_")[1]) for log in logs}
        self.assertEqual(combinations, {
            (attack, scenario)
            for attack in ("byzantine", "colluding", "contradicting", "DPI", "IPI", "impersonation")
            for scenario in ("education", "finance", "healthcare", "legal", "news")
        })
        self.assertIn("300 succeeded, 0 failed, 300 total", output)

    def test_optional_filters(self):
        for args, expected_count, expected_attack, expected_scenario in [
            (["--attack", "byzantine"], 50, "byzantine", None),
            (["--scenario", "news"], 60, None, "news"),
            (["--attack", "DPI", "--scenario", "finance"], 10, "DPI", "finance"),
            (["--attack", "contradicting", "--scenario", "news"], 10, "contradicting", "news"),
        ]:
            with self.subTest(args=args):
                status, logs, _ = self.run_batch(*args)
                self.assertEqual(status, 0)
                self.assertEqual(len(logs), expected_count)
                for log in logs:
                    if expected_attack:
                        self.assertEqual(log.parent.parent.name, expected_attack)
                    if expected_scenario:
                        self.assertEqual(log.stem.split("_")[1], expected_scenario)

    def test_failure_continues_and_is_reported(self):
        status, logs, output = self.run_batch("--attack", "byzantine", "--scenario", "news", fail_first=True)
        self.assertEqual(status, 1)
        self.assertEqual(len(logs), 10)
        self.assertIn("9 succeeded, 1 failed, 10 total", output)
        self.assertIn(f"Failed task log: {logs[0]}", output)

    def test_missing_temp_file_does_not_interrupt_batch(self):
        for fail_first in (False, True):
            with self.subTest(fail_first=fail_first):
                status, logs, output = self.run_batch(
                    "--attack", "byzantine", "--scenario", "news",
                    fail_first=fail_first, remove_temp=True,
                )
                self.assertEqual(status, int(fail_first))
                self.assertEqual(len(logs), 10)
                expected = "9 succeeded, 1 failed" if fail_first else "10 succeeded, 0 failed"
                self.assertIn(f"{expected}, 10 total", output)

    def test_every_dataset_role_can_be_created_with_tools(self):
        from autogen_agentchat.agents import AssistantAgent

        roles = {}
        for path in (ROOT / "data").glob("*/*.json"):
            for task in json.loads(path.read_text(encoding="utf-8")):
                for agent in task["agents"]:
                    roles[agent["agent_name"]] = agent
        task_path = self.root / "all_roles.json"
        task_path.write_text(json.dumps({"agents": list(roles.values()), "user query": "test"}), encoding="utf-8")
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "OPENAI_BASE_URL": "http://127.0.0.1:1/v1"}), \
             patch("sys.argv", ["autogen_run.py", "--json_data", str(task_path),
                                "--model", "test-model", "--config", "RoundRobin"]), \
             patch("autogen_agentchat.ui.Console", new_callable=AsyncMock), \
             patch("autogen_agentchat.agents.AssistantAgent", wraps=AssistantAgent) as constructor:
            namespace = runpy.run_path(str(ROOT / "autogen_run.py"), run_name="__main__")
        asyncio.run(namespace["model_client"].close())
        self.assertEqual(constructor.call_count, len(roles))
        for call in constructor.call_args_list:
            self.assertTrue(call.kwargs["tools"], call.kwargs["name"])


if __name__ == "__main__":
    unittest.main()
