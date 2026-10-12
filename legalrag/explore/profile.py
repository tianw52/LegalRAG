"""User profiles for the exploration app.

DEV-ONLY persistence: one JSON file per user under ``EXPLORE_PROFILES_DIR``
and name-only "login" (no password).  Replace before any public deployment.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from legalrag.prompts.loader import load_prompt


class UserProfile(BaseModel):
    user_id: str
    display_name: str
    province: str = ""                  # jurisdiction code; "" = not specified
    familiarity: str = ""               # key from explore_policy.yaml profile.familiarity
    goal: str = ""                      # key from explore_policy.yaml profile.goals
    problem_summary: str = ""           # optional, when goal == specific_problem
    interests: list[str] = Field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""

    @property
    def onboarded(self) -> bool:
        return bool(self.familiarity and self.goal)


class ProfileUpdate(BaseModel):
    province: str = ""
    familiarity: str
    goal: str
    problem_summary: str = ""
    interests: list[str] = Field(default_factory=list)


def profile_options() -> dict:
    """Onboarding options from explore_policy.yaml (served to the UI)."""
    return load_prompt("explore_policy")["profile"]


def validate_update(update: ProfileUpdate) -> None:
    opts = profile_options()
    if update.province not in {p["code"] for p in opts["provinces"]}:
        raise ValueError(f"Unknown province code: {update.province!r}")
    if update.familiarity not in {f["key"] for f in opts["familiarity"]}:
        raise ValueError(f"Unknown familiarity: {update.familiarity!r}")
    if update.goal not in {g["key"] for g in opts["goals"]}:
        raise ValueError(f"Unknown goal: {update.goal!r}")


class ProfileStore:
    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def user_id_for(display_name: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", display_name.strip().lower()).strip("-")
        if not slug:
            raise ValueError("display_name must contain letters or digits")
        return slug[:64]

    def _path(self, user_id: str) -> Path:
        return self._root / f"{user_id}.json"

    def get(self, user_id: str) -> UserProfile | None:
        path = self._path(user_id)
        if not path.exists():
            return None
        return UserProfile.model_validate_json(path.read_text(encoding="utf-8"))

    def login(self, display_name: str) -> UserProfile:
        user_id = self.user_id_for(display_name)
        existing = self.get(user_id)
        if existing:
            return existing
        now = _now()
        profile = UserProfile(
            user_id=user_id, display_name=display_name.strip(), created_at=now, updated_at=now
        )
        self._save(profile)
        return profile

    def update(self, user_id: str, update: ProfileUpdate) -> UserProfile:
        profile = self.get(user_id)
        if profile is None:
            raise KeyError(user_id)
        validate_update(update)
        profile = profile.model_copy(update={**update.model_dump(), "updated_at": _now()})
        self._save(profile)
        return profile

    def _save(self, profile: UserProfile) -> None:
        self._path(profile.user_id).write_text(
            json.dumps(profile.model_dump(), indent=2), encoding="utf-8"
        )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
