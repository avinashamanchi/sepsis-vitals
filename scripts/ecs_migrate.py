#!/usr/bin/env python3
"""Run `alembic upgrade head` as a one-off ECS task, before the service is updated.

Pre-serve migration step for the Terraform/ECS deployment
(docs/pilot/runbook.md):

1. Register the new task definition revision only:
       terraform -chdir=terraform apply -target=aws_ecs_task_definition.api -var image_tag=<tag>
2. Run this script with that revision. It starts one task with the command
   overridden to `python -m alembic upgrade head`, waits for it to stop and
   fails unless the container exited 0.
3. Only then update the service:
       terraform -chdir=terraform apply -var image_tag=<tag>

New tasks report ready (/ready, used by the ALB health check) only when the
database is at the migration head and the audit table is append-only. If
step 2 was skipped, the deployment does not become healthy and the ECS
circuit breaker rolls it back.

The default is a dry run: the commands are printed and nothing is executed.
Pass --execute to run them (this needs AWS credentials for the target
account and is an operator action, not something CI does).

    python scripts/ecs_migrate.py --cluster C --task-definition FAMILY:REV \\
        --subnets subnet-a,subnet-b --security-group sg-x [--execute]

`--from-terraform terraform/` reads cluster, task definition, subnets and
security group from `terraform output -json`.
"""

from __future__ import annotations

import argparse
import json
import subprocess  # nosec B404 - fixed aws/terraform commands, no shell
import sys
from typing import Any, Callable, Dict, List, Sequence

MIGRATION_COMMAND = ["python", "-m", "alembic", "upgrade", "head"]
CONTAINER = "api"

Runner = Callable[[Sequence[str]], str]


def _run(cmd: Sequence[str]) -> str:
    out = subprocess.run(list(cmd), capture_output=True, text=True, check=True, timeout=900)  # nosec B603
    return out.stdout


def run_task_command(cluster: str, task_definition: str, subnets: List[str], security_group: str) -> List[str]:
    overrides = {"containerOverrides": [{
        "name": CONTAINER,
        "command": MIGRATION_COMMAND,
        # The image entrypoint runs migrations itself only when this is true;
        # the command above runs them exactly once either way.
        "environment": [{"name": "SEPSIS_RUN_MIGRATIONS", "value": "false"}],
    }]}
    network = {"awsvpcConfiguration": {
        "subnets": subnets, "securityGroups": [security_group], "assignPublicIp": "DISABLED",
    }}
    return [
        "aws", "ecs", "run-task", "--cluster", cluster, "--task-definition", task_definition,
        "--launch-type", "FARGATE", "--count", "1", "--started-by", "pre-serve-migration",
        "--network-configuration", json.dumps(network), "--overrides", json.dumps(overrides),
        "--output", "json",
    ]


def outcome(describe_output: str) -> Dict[str, object]:
    """Exit code and reason of the migration container from `describe-tasks` JSON."""
    tasks = json.loads(describe_output).get("tasks", [])
    if not tasks:
        return {"ok": False, "reason": "task not found"}
    task = tasks[0]
    container: Dict[str, Any] = next(
        (c for c in task.get("containers", []) if c.get("name") == CONTAINER), {})
    code = container.get("exitCode")
    return {
        "ok": code == 0,
        "exit_code": code,
        "reason": container.get("reason") or task.get("stoppedReason"),
    }


def from_terraform(directory: str, runner: Runner = _run) -> Dict[str, Any]:
    raw = json.loads(runner(["terraform", f"-chdir={directory}", "output", "-json"]))
    return {
        "cluster": raw["ecs_cluster_name"]["value"],
        "task_definition": raw["api_task_definition_arn"]["value"],
        "subnets": raw["private_subnet_ids"]["value"],
        "security_group": raw["api_security_group_id"]["value"],
    }


def migrate(cluster: str, task_definition: str, subnets: List[str], security_group: str,
            execute: bool, runner: Runner = _run, log: Callable[[str], None] = print) -> int:
    start = run_task_command(cluster, task_definition, subnets, security_group)
    if not execute:
        log("Dry run (pass --execute to run):")
        log("  " + " ".join(start))
        log(f"  aws ecs wait tasks-stopped --cluster {cluster} --tasks <task-arn>")
        log(f"  aws ecs describe-tasks --cluster {cluster} --tasks <task-arn>")
        return 0
    started = json.loads(runner(start))
    if started.get("failures") or not started.get("tasks"):
        log(f"Migration task did not start: {started.get('failures')}")
        return 1
    arn = started["tasks"][0]["taskArn"]
    log(f"Migration task started: {arn}")
    runner(["aws", "ecs", "wait", "tasks-stopped", "--cluster", cluster, "--tasks", arn])
    result = outcome(runner(["aws", "ecs", "describe-tasks", "--cluster", cluster, "--tasks", arn,
                             "--output", "json"]))
    if result["ok"]:
        log("Migrations applied (exit code 0). The service may now be updated.")
        return 0
    log(f"Migration FAILED (exit code {result.get('exit_code')}, {result.get('reason')}). "
        "Do not update the service; see the task's CloudWatch log stream.")
    return 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cluster")
    parser.add_argument("--task-definition", help="family:revision or ARN of the NEW revision")
    parser.add_argument("--subnets", help="comma-separated private subnet ids")
    parser.add_argument("--security-group")
    parser.add_argument("--from-terraform", metavar="DIR")
    parser.add_argument("--execute", action="store_true", help="actually run the commands (default: dry run)")
    opts = parser.parse_args(argv)

    values: Dict[str, Any]
    if opts.from_terraform:
        values = from_terraform(opts.from_terraform)
    else:
        missing = [n for n in ("cluster", "task_definition", "subnets", "security_group") if not getattr(opts, n)]
        if missing:
            parser.error("missing " + ", ".join("--" + m.replace("_", "-") for m in missing))
        values = {"cluster": opts.cluster, "task_definition": opts.task_definition,
                  "subnets": opts.subnets.split(","), "security_group": opts.security_group}
    return migrate(str(values["cluster"]), str(values["task_definition"]), list(values["subnets"]),
                   str(values["security_group"]), opts.execute)


if __name__ == "__main__":
    sys.exit(main())
