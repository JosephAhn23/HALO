"""
Aggregator service that fans out retrieval to shard servers.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx
from fastapi import FastAPI
from pydantic import BaseModel

logger = logging.getLogger(__name__)

app = FastAPI(title="FAISS Shard Aggregator")


class SearchRequest(BaseModel):
    query_vector: list[float]
    top_k: int = 10


def _shard_urls() -> list[str]:
    urls = os.getenv("SHARD_URLS", "")
    return [u.strip() for u in urls.split(",") if u.strip()]


def _search_one(url: str, payload: dict) -> list[dict]:
    try:
        resp = httpx.post(f"{url}/search", json=payload, timeout=10.0)
        resp.raise_for_status()
        return resp.json().get("results", [])
    except Exception as exc:
        logger.warning("Shard %s failed: %s", url, exc)
        return []


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "shards": len(_shard_urls())}


@app.post("/search")
def search(req: SearchRequest) -> dict:
    payload = {"query_vector": req.query_vector, "top_k": req.top_k}
    urls = _shard_urls()
    if not urls:
        logger.warning("No shard URLs configured")
        return {"results": []}

    all_results: list[dict] = []
    with ThreadPoolExecutor(max_workers=len(urls)) as executor:
        futures = [executor.submit(_search_one, u, payload) for u in urls]
        for fut in as_completed(futures):
            all_results.extend(fut.result())

    all_results.sort(key=lambda x: x.get("retrieval_score", 0.0), reverse=True)
    return {"results": all_results[: req.top_k]}
