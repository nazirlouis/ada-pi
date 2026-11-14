import asyncio
import os
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from backend import bge_memory_shadow
from backend.memory_ops import memory_search
from tests.scenario_engine import FakeMddb, build_registry


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class FakeClient:
    def __init__(self, **_kwargs):
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def post(self, url, json, **_kwargs):
        self.calls.append((url, json))
        if url.endswith("/vector-search"):
            docs = {
                "collection-a": [
                    {"key": "a", "contentMd": "candidate a",
                     "meta": {"status": ["active"]}, "score": 0.8},
                    {"key": "shared", "contentMd": "candidate shared",
                     "meta": {"status": ["active"]}, "score": 0.7},
                ],
                "collection-b": [
                    {"key": "b", "contentMd": "candidate b",
                     "meta": {"status": ["active"]}, "score": 0.9},
                ],
            }[json["collection"]]
            return FakeResponse({
                "results": [
                    {"document": {k: v for k, v in doc.items() if k != "score"},
                     "score": doc["score"]}
                    for doc in docs
                ],
            })
        return FakeResponse({"scores": [0.2, 0.9, 0.1]})


class BgeMemoryShadowTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {
            "ADA_BGE_SHADOW_MDDB_URL": "http://mddb-embed/v1",
            "ADA_BGE_SHADOW_RERANK_URL": "http://reranker/rerank",
            "ADA_BGE_SHADOW_TIMEOUT_S": "2",
        })
        self.env.start()

    def tearDown(self):
        self.env.stop()

    async def test_shadow_merges_collections_and_returns_only_metrics(self):
        banks = [
            SimpleNamespace(name="bank-a", mddb_collection="collection-a",
                            writable=True, status_outcome=False, scope="shared"),
            SimpleNamespace(name="bank-b", mddb_collection="collection-b",
                            writable=True, status_outcome=False, scope="shared"),
        ]
        client = FakeClient()

        def client_factory(**kwargs):
            return client

        metrics = await bge_memory_shadow.run_shadow(
            query="private test query",
            banks=banks,
            baseline_ids=[("bank-b", "b"), ("bank-a", "a")],
            instance="tony",
            client_factory=client_factory,
        )

        self.assertEqual(metrics["status"], "ok")
        self.assertEqual(metrics["candidate_count"], 3)
        self.assertEqual(metrics["reranked_count"], 3)
        self.assertEqual(metrics["overlap_at_5"], 2)
        self.assertNotIn("private test query", repr(metrics))
        self.assertNotIn("candidate a", repr(metrics))
        self.assertEqual(len(client.calls), 3)
        ranked_docs = client.calls[-1][1]["docs"]
        self.assertEqual([doc["text"] for doc in ranked_docs],
                         ["candidate b", "candidate a", "candidate shared"])

    def test_schedule_is_disabled_by_default(self):
        with patch.dict(os.environ, {"ADA_BGE_SHADOW_ENABLED": "0"}):
            accepted = bge_memory_shadow.schedule(
                "query", [SimpleNamespace(name="general")], [],
                instance="tony",
            )
        self.assertFalse(accepted)

    async def test_shadow_failure_does_not_change_gemini_result(self):
        mddb = FakeMddb()
        registry = build_registry("tony")
        bank = registry.bank("general")
        await mddb.add_document(
            bank.mddb_collection, "general/test", "en", "known result",
            {"status": ["active"], "kind": ["fact"]},
        )
        with patch.dict(os.environ, {
            "ADA_BGE_SHADOW_ENABLED": "1",
            "ADA_BGE_SHADOW_MDDB_URL": "http://mddb-embed/v1",
            "ADA_BGE_SHADOW_RERANK_URL": "http://reranker/rerank",
        }), patch(
            "backend.bge_memory_shadow.run_shadow",
            new_callable=AsyncMock,
            side_effect=RuntimeError("synthetic failure"),
        ) as shadow:
            result = await memory_search(
                mddb, registry, "general", "known result", limit=5)
            await asyncio.sleep(0)

        self.assertEqual(result["count"], 1)
        self.assertEqual(result["hits"][0]["key"], "general/test")
        self.assertFalse(result["degraded"])
        shadow.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
