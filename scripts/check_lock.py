#!/usr/bin/env python3
"""Check that requirements/*.txt still satisfy pyproject.toml.

For every direct requirement of each installation set, and for each supported
Python version on Linux, the lock must contain exactly one pin that applies
and satisfies the declared specifier. Every pin must carry hashes. This check
does not depend on what PyPI offers today, so it stays stable over time; to
move pins forward, regenerate the locks (docs/dependencies.md).

    python scripts/check_lock.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import tomllib
from packaging.markers import Marker
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parents[1]
PYTHONS = ("3.10", "3.11", "3.12")
LOCKS = {
    "requirements/deploy.txt": ("api", "ml"),
    "requirements/dev.txt": ("api", "ml", "train", "copilot", "integrations", "dev"),
}
_PIN = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)(?:\[[^\]]*\])?==([^\s;\\]+)(?:\s*;\s*([^\\]+?))?\s*\\?$")


def _env(python: str) -> dict:
    return {"python_version": python, "python_full_version": f"{python}.0", "sys_platform": "linux",
            "platform_system": "Linux", "platform_machine": "x86_64", "os_name": "posix",
            "implementation_name": "cpython", "platform_python_implementation": "CPython", "extra": ""}


def _pins(path: Path) -> tuple[list[tuple[str, str, str]], list[str]]:
    pins, problems = [], []
    lines = path.read_text().splitlines()
    for i, line in enumerate(lines):
        match = _PIN.match(line.strip())
        if not match:
            continue
        pins.append((canonicalize_name(match.group(1)), match.group(2), (match.group(3) or "").strip()))
        if i + 1 >= len(lines) or "--hash=sha256:" not in lines[i + 1]:
            problems.append(f"{path.name}: {match.group(1)}=={match.group(2)} has no hash")
    return pins, problems


def main() -> int:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    problems: list[str] = []
    for lock, extras in LOCKS.items():
        pins, hash_problems = _pins(ROOT / lock)
        problems += hash_problems
        wanted = [Requirement(r) for r in project["dependencies"]]
        for extra in extras:
            wanted += [Requirement(r) for r in project["optional-dependencies"][extra]]
        for python in PYTHONS:
            env = _env(python)
            for req in wanted:
                if req.marker is not None and not req.marker.evaluate(env):
                    continue
                name = canonicalize_name(req.name)
                applicable = [v for n, v, m in pins if n == name and (not m or Marker(m).evaluate(env))]
                if len(applicable) != 1:
                    problems.append(f"{lock} (Python {python}): {req.name} pinned {len(applicable)} times")
                elif not req.specifier.contains(applicable[0], prereleases=True):
                    problems.append(f"{lock} (Python {python}): {req.name}=={applicable[0]} violates '{req.specifier}'")
    for problem in problems:
        print(problem)
    print("locks satisfy pyproject.toml" if not problems else f"{len(problems)} problem(s): regenerate the locks")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
