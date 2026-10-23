"""
tests/test_deploy_config.py — static checks on the shipped deployment config.

Regression: docker/docker-compose.yml was invalid YAML (missing space after a
key), so `docker compose up` could never have run. These checks run in
ordinary CI; the compose-smoke job starts the real stack.
"""

from __future__ import annotations

import ipaddress
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def compose():
    return yaml.safe_load((ROOT / "docker" / "docker-compose.yml").read_text())


def test_compose_is_valid_and_has_core_services(compose):
    assert {"db", "redis", "api", "dashboard"} <= set(compose["services"])


def test_only_the_dashboard_is_published_beyond_loopback(compose):
    for name, svc in compose["services"].items():
        for port in svc.get("ports", []):
            if name == "dashboard":
                continue
            assert str(port).startswith("127.0.0.1:"), f"{name} publishes {port} on all interfaces"


def test_api_trusts_only_the_compose_network(compose):
    subnet = compose["networks"]["sepsis_net"]["ipam"]["config"][0]["subnet"]
    env = compose["services"]["api"]["environment"]
    assert ipaddress.ip_network(env["TRUSTED_PROXIES"]) == ipaddress.ip_network(subnet)


def test_model_is_mounted_read_only_and_state_is_separate(compose):
    volumes = compose["services"]["api"]["volumes"]
    assert any(v.endswith(":/models:ro") for v in volumes)
    assert any(v.endswith(":/app/state") for v in volumes)


def test_frozen_copilot_key_is_optional(compose):
    env = compose["services"]["api"]["environment"]
    assert ":?" not in str(env.get("ANTHROPIC_API_KEY", ""))


def test_postgres_bootstrap_creates_no_tables():
    sql = (ROOT / "docker" / "postgres" / "init.sql").read_text().upper()
    assert "CREATE TABLE" not in sql  # Alembic owns the schema


def test_password_reset_has_a_token_secret_where_jwts_use_rsa(compose):
    """N50: with RSA-signed JWTs and no SEPSIS_TOKEN_SECRET/SEPSIS_JWT_SECRET,
    every password-reset request failed with a 500."""
    env = compose["services"]["api"]["environment"]
    assert "JWT_PRIVATE_KEY" in env and "SEPSIS_TOKEN_SECRET" in env
    terraform = (ROOT / "terraform" / "main.tf").read_text()
    assert 'name = "SEPSIS_TOKEN_SECRET"' in terraform


def test_terraform_tasks_can_read_their_secrets():
    """N51: the ECS execution role had no policies, so tasks could not pull
    the image, write logs or resolve secrets."""
    terraform = (ROOT / "terraform" / "main.tf").read_text()
    assert "AmazonECSTaskExecutionRolePolicy" in terraform
    assert "secretsmanager:GetSecretValue" in terraform
