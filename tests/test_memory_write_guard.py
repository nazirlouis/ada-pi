"""Card ada-memory-write-caps: hard size caps on every memory write path
with a self-correctable refusal {ok:false, error:'memory_cap', size, cap,
hint}. The module's own --selftest covers the pure boundary math; these
tests cover the wiring into ada_remember / guest / vocab paths."""
from __future__ import annotations

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

# Guard refusals/warns audit to events.md — keep tests hermetic (the real
# default is ~/.local/share/ada/events.md).
os.environ["ADA_EVENTS_FILE"] = os.path.join(
    tempfile.mkdtemp(), "events.md")

from backend import memory_ops
from backend import memory_write_guard as guard
from backend.tool_runner import ToolRunner
from tests.scenario_engine import FakeMddb, build_registry


class GuardUnitTests(unittest.TestCase):
    def setUp(self):
        # These are process-global throttles; isolate tests from memory-write
        # activity in other test modules and from repeated CI runs.
        guard._last_doc_warn.clear()
        guard._last_doc_check.clear()

    def test_at_and_under_cap_pass(self):
        cap = guard.entry_cap()
        self.assertIsNone(guard.check_entry("x" * cap))
        self.assertIsNone(guard.check_entry("x" * (cap - 1)))

    def test_over_cap_refusal_shape(self):
        cap = guard.entry_cap()
        out = guard.check_entry("x" * (cap + 1))
        self.assertFalse(out["ok"])
        self.assertEqual(out["error"], "memory_cap")
        self.assertEqual(out["size"], cap + 1)
        self.assertEqual(out["cap"], cap)
        self.assertIn("shorten by 1 ", out["hint"])
        self.assertIn("split into 2 entries", out["hint"])

    def test_field_cap(self):
        fcap = guard.field_cap()
        self.assertIsNone(guard.check_field("key", "k" * fcap))
        out = guard.check_field("key", "k" * (fcap + 1))
        self.assertEqual(out["error"], "memory_cap")
        self.assertEqual(out["field"], "key")

    def test_per_bank_override(self):
        cap = guard.entry_cap()
        guard.BANK_CAPS["archive-x"] = {"entry": cap * 10}
        try:
            self.assertIsNone(
                guard.check_entry("x" * (cap * 2), bank="archive-x"))
        finally:
            del guard.BANK_CAPS["archive-x"]

    def test_doc_warn_fires_once_and_throttles(self):
        thr = guard.doc_warn_threshold()
        self.assertIsNone(guard.note_doc_count("t-u", thr - 1))
        guard._last_doc_warn.pop("t-o", None)
        first = guard.note_doc_count("t-o", thr + 1)
        second = guard.note_doc_count("t-o", thr + 2)
        self.assertTrue(first["warned"])
        self.assertFalse(second["warned"])


class RememberCapTests(unittest.IsolatedAsyncioTestCase):
    """memory_ops.remember — the ada_remember curated-bank path."""

    async def asyncSetUp(self):
        self.mddb = FakeMddb()
        self.registry = build_registry("tony")
        # The guard throttles by module-global bank/collection keys.
        guard._last_doc_warn.clear()
        guard._last_doc_check.clear()

    async def test_over_cap_text_refused_before_write(self):
        out = await memory_ops.remember(
            self.mddb, self.registry, "test", "personal",
            "x " * (guard.entry_cap() // 2 + 1))
        self.assertFalse(out["ok"])
        self.assertEqual(out["error"], "memory_cap")
        self.assertEqual(self.mddb.collections, {})

    async def test_at_cap_text_lands(self):
        out = await memory_ops.remember(
            self.mddb, self.registry, "test", "personal",
            "x " * (guard.entry_cap() // 2))
        self.assertEqual(out["verb"], "create")
        coll = self.registry.bank("personal").mddb_collection
        self.assertEqual(len(self.mddb.collections[coll]), 1)

    async def test_over_cap_key_refused(self):
        out = await memory_ops.remember(
            self.mddb, self.registry, "test", "personal", "short text",
            key="k " * (guard.field_cap() // 2 + 1))
        self.assertEqual(out["error"], "memory_cap")
        self.assertEqual(out["field"], "key")
        self.assertEqual(self.mddb.collections, {})

    async def test_supersede_counts_new_body_not_delta(self):
        coll = self.registry.bank("personal").mddb_collection
        await self.mddb.add_document(
            coll, "personal/old", "en", "old body",
            {"status": ["active"], "kind": ["note"]})
        out = await memory_ops.remember(
            self.mddb, self.registry, "test", "personal",
            "x " * (guard.entry_cap() // 2 + 1), supersedes="personal/old")
        self.assertEqual(out["error"], "memory_cap")
        # The refused supersede left the old doc untouched.
        doc = await self.mddb.get_document(coll, "personal/old")
        self.assertEqual(doc["meta"]["status"], ["active"])

    async def test_bank_doc_warn_fires_on_large_bank(self):
        coll = self.registry.bank("personal").mddb_collection
        thr = guard.doc_warn_threshold("personal")
        for i in range(thr):
            await self.mddb.add_document(
                coll, f"personal/doc-{i}", "en", "x", {})
        guard._last_doc_check.pop(coll, None)
        guard._last_doc_warn.pop("personal", None)
        out = await guard.warn_if_bank_large(self.mddb, coll, "personal")
        self.assertTrue(out["warned"])
        self.assertEqual(out["doc_count"], thr)


class ToolRunnerCapTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.ha_client = AsyncMock()
        self.ha_client.base_url = "http://test:8123"
        self.runner = ToolRunner(self.ha_client, instance_id="test")
        self.runner._banks = build_registry("tony")
        self.runner.mddb = AsyncMock()
        self.runner.mddb.is_ops_routed = lambda c: False
        self.runner.mddb.search_documents.return_value = []
        self.runner.mddb.get_document.return_value = None
        self.runner.mddb.add_document.return_value = {}
        self.runner.mddb.update_document.return_value = {}

    async def test_ada_remember_over_cap_returns_structured_refusal(self):
        out = await self.runner.execute(
            "ada_remember",
            {"bank": "personal", "confirmed": True,
             "text": "x " * (guard.entry_cap() // 2 + 1)})
        self.assertFalse(out["ok"])
        self.assertEqual(out["error"], "memory_cap")
        self.assertIn("hint", out)
        self.runner.mddb.add_document.assert_not_called()

    async def test_guest_remember_inherits_caps(self):
        self.runner.chaba = SimpleNamespace()
        self.runner.chaba.remember = Mock(return_value={"ok": True})
        out = await self.runner.execute(
            "ada_remember",
            {"kind": "guest", "text": "x " * (guard.entry_cap() // 2 + 1)})
        self.assertFalse(out["ok"])
        self.assertEqual(out["error"], "memory_cap")
        self.runner.chaba.remember.assert_not_called()

    async def test_guest_remember_private_inherits_caps(self):
        self.runner.chaba = SimpleNamespace()
        self.runner.chaba.remember_private = Mock(
            return_value={"ok": True})
        out = await self.runner.execute(
            "guest_remember_private",
            {"key": "wifi", "text": "x " * (guard.entry_cap() // 2 + 1)})
        self.assertFalse(out["ok"])
        self.assertEqual(out["error"], "memory_cap")
        self.runner.chaba.remember_private.assert_not_called()

    async def test_vocab_append_counts_final_merged_size(self):
        # A small vocab line appended to a near-full log must refuse on
        # the merged size, not the delta.
        body = "# Vocabulary\n" + "x " * (guard.entry_cap() // 2)
        self.runner.mddb.get_document.return_value = {
            "key": "vocab/log", "contentMd": body, "meta": {}}
        out = await self.runner.execute(
            "ada_remember", {"kind": "vocab", "text": "a → b"})
        self.assertFalse(out["ok"])
        self.assertEqual(out["error"], "memory_cap")
        self.assertGreater(out["size"], guard.entry_cap())
        self.runner.mddb.update_document.assert_not_called()

    async def test_vocab_append_under_cap_lands(self):
        self.runner.mddb.get_document.return_value = None
        self.runner.mddb.add_document.return_value = {"key": "vocab/log"}
        out = await self.runner.execute(
            "ada_remember", {"kind": "vocab", "text": "a → b"})
        self.assertEqual(out["status"], "noted")
        self.runner.mddb.add_document.assert_called_once()


if __name__ == "__main__":
    unittest.main()
