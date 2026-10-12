"""FastAPI backend for the exploration UI (web/).

Run (from repo root, OpenSearch up, index ingested):

    conda activate legalrag
    uvicorn legalrag.api.app:app --reload --port 8000

Endpoints
---------
GET  /api/health                  index + doc count
GET  /api/options                 profile onboarding options (explore_policy.yaml)
POST /api/login                   {display_name} → profile (DEV-ONLY: name-only login)
GET  /api/profile/{user_id}
PUT  /api/profile/{user_id}       onboarding answers
GET  /api/examples                sample CanLegalRAGBench questions (?province=BC)
POST /api/explore                 NDJSON stream of exploration events (see explore/engine.py)
GET  /api/document/{doc_id}       full text + metadata for the document reader
"""

from __future__ import annotations

import json
import logging
import random
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from legalrag.core.config import settings
from legalrag.explore.engine import ExplorationEngine
from legalrag.explore.profile import ProfileStore, ProfileUpdate, UserProfile, profile_options
from legalrag.utils.logging import configure_logging

logger = logging.getLogger(__name__)

_QUERIES_PATH = Path("data/CanLegalRAGBench/queries.json")
_state: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    configure_logging(level="INFO", log_file=None)
    _state["engine"] = ExplorationEngine()
    _state["profiles"] = ProfileStore(settings.explore.profiles_dir)
    yield
    _state.clear()


app = FastAPI(title="LegalRAG Explore API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.explore.cors_origins.split(",") if o.strip()],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _engine() -> ExplorationEngine:
    return _state["engine"]


def _profiles() -> ProfileStore:
    return _state["profiles"]


class LoginRequest(BaseModel):
    display_name: str = Field(min_length=1, max_length=80)


class ExploreRequest(BaseModel):
    question: str = Field(min_length=3, max_length=4000)
    user_id: str | None = None
    scope: Literal["profile", "all"] = "profile"


def _profile_out(p: UserProfile) -> dict:
    return {**p.model_dump(), "onboarded": p.onboarded}


@app.get("/api/health")
def health() -> dict:
    client = _engine().os_client
    count = client._client.count(index=client.index_name)["count"]
    return {"status": "ok", "index": client.index_name, "chunks": count}


@app.get("/api/options")
def options() -> dict:
    return profile_options()


@app.post("/api/login")
def login(req: LoginRequest) -> dict:
    try:
        return _profile_out(_profiles().login(req.display_name))
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.get("/api/profile/{user_id}")
def get_profile(user_id: str) -> dict:
    profile = _profiles().get(user_id)
    if profile is None:
        raise HTTPException(404, "profile not found")
    return _profile_out(profile)


@app.put("/api/profile/{user_id}")
def put_profile(user_id: str, update: ProfileUpdate) -> dict:
    try:
        return _profile_out(_profiles().update(user_id, update))
    except KeyError as exc:
        raise HTTPException(404, "profile not found") from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@lru_cache(maxsize=1)
def _dataset_queries() -> list[dict]:
    if not _QUERIES_PATH.exists():
        return []
    with _QUERIES_PATH.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


@app.get("/api/examples")
def examples(province: str = "", n: int = 4) -> list[dict]:
    labels = {p["code"]: p["label"] for p in profile_options()["provinces"]}
    pool = _dataset_queries()
    if province in labels and province:
        pool = [q for q in pool if q.get("province") == labels[province]] or pool
    picked = random.sample(pool, min(n, len(pool)))
    return [{"query_id": q["query_id"], "text": q["query_text"], "province": q.get("province")} for q in picked]


@app.post("/api/explore")
async def explore(req: ExploreRequest) -> StreamingResponse:
    profile = _profiles().get(req.user_id) if req.user_id else None

    async def stream():
        try:
            async for event in _engine().explore(req.question, profile, req.scope):
                yield json.dumps(event, default=str) + "\n"
        except Exception as exc:  # surface failures to the UI instead of a dropped stream
            logger.exception("Exploration failed")
            yield json.dumps({"event": "error", "message": str(exc)}) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


@app.get("/api/document/{doc_id}")
def document(doc_id: str) -> dict:
    doc = _engine().document(doc_id)
    if doc is None:
        raise HTTPException(404, "document not found")
    return doc
