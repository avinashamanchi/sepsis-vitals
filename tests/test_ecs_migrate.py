"""
tests/test_ecs_migrate.py — the pre-serve migration step runs alembic once in
a private one-off task and refuses to continue unless it exited 0. No AWS
calls: the command runner is replaced.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ecs_migrate", ROOT / "scripts" / "ecs_migrate.py")
ecs_migrate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ecs_migrate)

ARGS = ("sepsis-cluster", "sepsis-api:7", ["subnet-a", "subnet-b"], "sg-1")


def _fake(exit_code):
    calls = []

    def runner(cmd):
        calls.append(list(cmd))
        if cmd[:3] == ["aws", "ecs", "run-task"]:
            return json.dumps({"tasks": [{"taskArn": "arn:task/1"}], "failures": []})
        if cmd[:3] == ["aws", "ecs", "describe-tasks"]:
            return json.dumps({"tasks": [{"stoppedReason": "Essential container in task exited",
                                          "containers": [{"name": "api", "exitCode": exit_code}]}]})
        return ""
    return runner, calls


def test_dry_run_executes_nothing():
    lines = []
    runner, calls = _fake(0)
    assert ecs_migrate.migrate(*ARGS, execute=False, runner=runner, log=lines.append) == 0
    assert calls == [] and "Dry run" in lines[0]


def test_runs_alembic_once_in_a_private_task_and_checks_the_exit_code():
    runner, calls = _fake(0)
    assert ecs_migrate.migrate(*ARGS, execute=True, runner=runner, log=lambda _: None) == 0
    start = calls[0]
    overrides = json.loads(start[start.index("--overrides") + 1])["containerOverrides"][0]
    assert overrides["command"] == ["python", "-m", "alembic", "upgrade", "head"]
    network = json.loads(start[start.index("--network-configuration") + 1])["awsvpcConfiguration"]
    assert network["assignPublicIp"] == "DISABLED" and network["subnets"] == ["subnet-a", "subnet-b"]
    assert [c[:3] for c in calls] == [["aws", "ecs", "run-task"], ["aws", "ecs", "wait"],
                                      ["aws", "ecs", "describe-tasks"]]


def test_a_failed_migration_blocks_the_deploy():
    lines = []
    runner, _ = _fake(1)
    assert ecs_migrate.migrate(*ARGS, execute=True, runner=runner, log=lines.append) == 1
    assert "Do not update the service" in lines[-1]


def test_outcome_without_a_task_is_a_failure():
    assert ecs_migrate.outcome(json.dumps({"tasks": []})) == {"ok": False, "reason": "task not found"}
