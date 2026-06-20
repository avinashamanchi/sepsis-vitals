"""
sepsis_vitals.state — Ephemeral patient state store (rolling windows).

Uses Redis for production (distributed, survives container restarts)
with in-memory fallback for development.

IMPORTANT: This store is the *fast, ephemeral* layer for computing
rolling means, deltas, and observation gaps.  Data here expires after
24 hours (VITALS_WINDOW_TTL).  It is NOT the system of record.

The *permanent* clinical audit trail lives in Postgres:
- ``prediction_records`` table (every ML prediction, immutable)
- ``audit_log`` table (every PHI access, HIPAA § 164.312(b))
- ``vitals`` table (raw vital sign observations)

Redis = fast math.  Postgres = legal truth.
"""

import asyncio
import json
import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Default TTL for patient vitals windows (24 hours)
VITALS_WINDOW_TTL = 86400
# Maximum observations to keep per patient per vital
MAX_WINDOW_SIZE = 100


@dataclass
class VitalObservation:
    """A single vital sign observation with timestamp."""
    timestamp: float
    value: float
    source: str = "manual"


@dataclass
class PatientState:
    """Aggregated patient state computed from rolling windows."""
    patient_id: str
    latest_vitals: Dict[str, float] = field(default_factory=dict)
    rolling_means: Dict[str, float] = field(default_factory=dict)
    rolling_stds: Dict[str, float] = field(default_factory=dict)
    deltas: Dict[str, float] = field(default_factory=dict)
    baseline_risk: float = 0.0
    last_observation_time: float = 0.0
    observation_count: int = 0
    obs_gap_minutes: float = 0.0


class RedisPatientStateStore:
    """Redis-backed patient state store for production deployments.

    Stores patient vitals as sorted sets (timestamp → value) enabling
    efficient rolling window calculations across distributed workers.

    Key schema:
        sv:vitals:{patient_id}:{vital_name}  → sorted set (score=timestamp, member=json)
        sv:risk:{patient_id}                  → hash (baseline_risk, last_update, etc.)
        sv:meta:{patient_id}                  → hash (obs_count, last_obs_time, etc.)
    """

    KEY_PREFIX = "sv"

    def __init__(self, redis_url: Optional[str] = None):
        self._redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self._redis = None
        self._connected = False

    async def connect(self) -> bool:
        """Establish Redis connection. Returns True if successful."""
        try:
            import redis.asyncio as aioredis
            self._redis = aioredis.from_url(
                self._redis_url,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
            )
            await self._redis.ping()
            self._connected = True
            logger.info("Redis patient state store connected: %s", self._redis_url.split("@")[-1])
            return True
        except Exception as exc:
            logger.warning("Redis connection failed (%s), using in-memory fallback", exc)
            self._connected = False
            return False

    async def close(self):
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()
            self._connected = False

    def _key(self, *parts: str) -> str:
        return f"{self.KEY_PREFIX}:{':'.join(parts)}"

    async def record_vital(
        self, patient_id: str, vital_name: str, value: float,
        timestamp: Optional[float] = None, source: str = "manual",
    ) -> None:
        """Record a single vital sign observation."""
        ts = timestamp or time.time()
        member = json.dumps({"v": value, "s": source, "t": ts})
        key = self._key("vitals", patient_id, vital_name)

        pipe = self._redis.pipeline()
        pipe.zadd(key, {member: ts})
        pipe.expire(key, VITALS_WINDOW_TTL)
        # Trim to MAX_WINDOW_SIZE most recent
        pipe.zremrangebyrank(key, 0, -(MAX_WINDOW_SIZE + 1))
        await pipe.execute()

        # Update metadata
        meta_key = self._key("meta", patient_id)
        await self._redis.hset(meta_key, mapping={
            "last_obs_time": str(ts),
            "last_vital": vital_name,
        })
        await self._redis.hincrby(meta_key, "obs_count", 1)
        await self._redis.expire(meta_key, VITALS_WINDOW_TTL)

    async def record_vitals_batch(
        self, patient_id: str, vitals: Dict[str, float],
        timestamp: Optional[float] = None, source: str = "manual",
    ) -> None:
        """Record multiple vital signs at once (single round-trip)."""
        ts = timestamp or time.time()
        pipe = self._redis.pipeline()

        for vital_name, value in vitals.items():
            member = json.dumps({"v": value, "s": source, "t": ts})
            key = self._key("vitals", patient_id, vital_name)
            pipe.zadd(key, {member: ts})
            pipe.expire(key, VITALS_WINDOW_TTL)
            pipe.zremrangebyrank(key, 0, -(MAX_WINDOW_SIZE + 1))

        meta_key = self._key("meta", patient_id)
        pipe.hset(meta_key, mapping={
            "last_obs_time": str(ts),
            "obs_count_inc": str(len(vitals)),
        })
        pipe.hincrby(meta_key, "obs_count", len(vitals))
        pipe.expire(meta_key, VITALS_WINDOW_TTL)

        await pipe.execute()

    async def get_patient_state(self, patient_id: str) -> PatientState:
        """Compute aggregated patient state from rolling windows."""
        state = PatientState(patient_id=patient_id)

        vital_names = [
            "temperature", "heart_rate", "resp_rate", "sbp", "dbp",
            "spo2", "gcs", "map", "lactate", "wbc", "procalcitonin",
        ]

        pipe = self._redis.pipeline()
        for vname in vital_names:
            key = self._key("vitals", patient_id, vname)
            pipe.zrange(key, 0, -1, withscores=True)

        meta_key = self._key("meta", patient_id)
        pipe.hgetall(meta_key)

        risk_key = self._key("risk", patient_id)
        pipe.hgetall(risk_key)

        results = await pipe.execute()

        for i, vname in enumerate(vital_names):
            observations = results[i]
            if not observations:
                continue

            values = []
            for member, score in observations:
                try:
                    data = json.loads(member)
                    values.append((data["t"], data["v"]))
                except (json.JSONDecodeError, KeyError):
                    continue

            if not values:
                continue

            values.sort(key=lambda x: x[0])
            latest_val = values[-1][1]
            all_vals = [v[1] for v in values]

            state.latest_vitals[vname] = latest_val

            # Rolling mean
            n = len(all_vals)
            mean = sum(all_vals) / n
            state.rolling_means[f"{vname}_roll_mean"] = round(mean, 4)

            # Rolling std
            if n > 1:
                variance = sum((x - mean) ** 2 for x in all_vals) / (n - 1)
                state.rolling_stds[f"{vname}_roll_std"] = round(variance ** 0.5, 4)
            else:
                state.rolling_stds[f"{vname}_roll_std"] = 0.0

            # Delta (latest - previous)
            if n >= 2:
                state.deltas[f"{vname}_delta"] = round(values[-1][1] - values[-2][1], 4)
            else:
                state.deltas[f"{vname}_delta"] = 0.0

        # Metadata
        meta = results[-2]  # second to last result
        if meta:
            state.last_observation_time = float(meta.get("last_obs_time", 0))
            state.observation_count = int(meta.get("obs_count", 0))
            if state.last_observation_time > 0:
                state.obs_gap_minutes = (time.time() - state.last_observation_time) / 60.0

        # Risk baseline
        risk = results[-1]  # last result
        if risk:
            state.baseline_risk = float(risk.get("baseline_risk", 0))

        return state

    async def set_baseline_risk(self, patient_id: str, risk: float) -> None:
        """Update the baseline risk score for a patient."""
        key = self._key("risk", patient_id)
        await self._redis.hset(key, mapping={
            "baseline_risk": str(risk),
            "updated_at": str(time.time()),
        })
        await self._redis.expire(key, VITALS_WINDOW_TTL)

    async def get_recent_patients(self, limit: int = 100) -> List[str]:
        """Get patient IDs with recent observations."""
        # Scan for meta keys
        patient_ids = []
        async for key in self._redis.scan_iter(
            match=f"{self.KEY_PREFIX}:meta:*", count=limit
        ):
            pid = key.split(":")[-1]
            patient_ids.append(pid)
            if len(patient_ids) >= limit:
                break
        return patient_ids

    async def delete_patient(self, patient_id: str) -> None:
        """Remove all state for a patient."""
        keys = []
        async for key in self._redis.scan_iter(
            match=f"{self.KEY_PREFIX}:*:{patient_id}*"
        ):
            keys.append(key)
        if keys:
            await self._redis.delete(*keys)


class InMemoryPatientStateStore:
    """In-memory fallback for development (NOT for production).

    WARNING: Data is lost on process restart. Not shared across workers.
    Only use for local development and testing.
    """

    def __init__(self):
        self._vitals: Dict[str, Dict[str, List[Tuple[float, float]]]] = defaultdict(lambda: defaultdict(list))
        self._meta: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self._risk: Dict[str, float] = {}
        self._connected = True
        logger.warning(
            "Using in-memory patient state store. "
            "Data will NOT survive restarts. Set REDIS_URL for production."
        )

    async def connect(self) -> bool:
        return True

    async def close(self):
        pass

    async def record_vital(
        self, patient_id: str, vital_name: str, value: float,
        timestamp: Optional[float] = None, source: str = "manual",
    ) -> None:
        ts = timestamp or time.time()
        window = self._vitals[patient_id][vital_name]
        window.append((ts, value))
        if len(window) > MAX_WINDOW_SIZE:
            window.pop(0)
        self._meta[patient_id]["last_obs_time"] = ts
        self._meta[patient_id]["obs_count"] = self._meta[patient_id].get("obs_count", 0) + 1

    async def record_vitals_batch(
        self, patient_id: str, vitals: Dict[str, float],
        timestamp: Optional[float] = None, source: str = "manual",
    ) -> None:
        for vname, value in vitals.items():
            await self.record_vital(patient_id, vname, value, timestamp, source)

    async def get_patient_state(self, patient_id: str) -> PatientState:
        state = PatientState(patient_id=patient_id)

        for vname, observations in self._vitals.get(patient_id, {}).items():
            if not observations:
                continue
            observations.sort()
            values = [v for _, v in observations]

            state.latest_vitals[vname] = values[-1]

            n = len(values)
            mean = sum(values) / n
            state.rolling_means[f"{vname}_roll_mean"] = round(mean, 4)

            if n > 1:
                variance = sum((x - mean) ** 2 for x in values) / (n - 1)
                state.rolling_stds[f"{vname}_roll_std"] = round(variance ** 0.5, 4)
            else:
                state.rolling_stds[f"{vname}_roll_std"] = 0.0

            if n >= 2:
                state.deltas[f"{vname}_delta"] = round(values[-1] - values[-2], 4)
            else:
                state.deltas[f"{vname}_delta"] = 0.0

        meta = self._meta.get(patient_id, {})
        state.last_observation_time = meta.get("last_obs_time", 0)
        state.observation_count = meta.get("obs_count", 0)
        if state.last_observation_time > 0:
            state.obs_gap_minutes = (time.time() - state.last_observation_time) / 60.0
        state.baseline_risk = self._risk.get(patient_id, 0)

        return state

    async def set_baseline_risk(self, patient_id: str, risk: float) -> None:
        self._risk[patient_id] = risk

    async def get_recent_patients(self, limit: int = 100) -> List[str]:
        return list(self._vitals.keys())[:limit]

    async def delete_patient(self, patient_id: str) -> None:
        self._vitals.pop(patient_id, None)
        self._meta.pop(patient_id, None)
        self._risk.pop(patient_id, None)


async def create_state_store():
    """Factory: create the appropriate state store based on environment.

    Returns RedisPatientStateStore if REDIS_URL is set and reachable,
    otherwise falls back to InMemoryPatientStateStore with a warning.
    """
    redis_url = os.getenv("REDIS_URL")
    if redis_url:
        store = RedisPatientStateStore(redis_url)
        if await store.connect():
            return store
        logger.warning("Redis unavailable, falling back to in-memory store")

    return InMemoryPatientStateStore()
