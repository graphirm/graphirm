"""Tests for completeness.py on a fixture with the real graph schema.

    python3 -m unittest eval/structure/test_completeness.py
"""

import json
import os
import sqlite3
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(__file__))
import completeness as sc  # noqa: E402

SCHEMA = """
CREATE TABLE nodes (id TEXT PRIMARY KEY, node_type TEXT NOT NULL, data TEXT NOT NULL,
    metadata TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
CREATE TABLE edges (id TEXT PRIMARY KEY, edge_type TEXT NOT NULL, source_id TEXT NOT NULL,
    target_id TEXT NOT NULL, weight REAL NOT NULL DEFAULT 1.0, metadata TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL);
"""


class Fixture:
    def __init__(self):
        self.nodes, self.edges = [], []

    def node(self, nid, ntype, data, meta=None):
        self.nodes.append((nid, ntype, json.dumps(data), json.dumps(meta or {}), "t", "t"))
        return nid

    def edge(self, etype, src, dst, meta=None):
        self.edges.append((f"e{len(self.edges)}", etype, src, dst, 1.0, json.dumps(meta or {}), "t"))

    def assistant(self, nid, content="", calls=(), meta=None):
        meta = dict(meta or {})
        if calls:
            meta["tool_calls"] = [{"id": c, "name": n, "arguments": a} for c, n, a in calls]
        self.node(nid, "interaction", {"role": "assistant", "content": content}, meta)
        self.edge("produces", "agent", nid)

    def tool(self, nid, call_id, content="ok", is_error=False):
        self.node(nid, "interaction", {"role": "tool", "content": content},
                  {"tool_call_id": call_id, "is_error": is_error})
        self.edge("produces", "agent", nid)

    def write(self, directory):
        path = os.path.join(directory, "graph.db")
        con = sqlite3.connect(path)
        con.executescript(SCHEMA)
        con.executemany("INSERT INTO nodes VALUES (?,?,?,?,?,?)", self.nodes)
        con.executemany("INSERT INTO edges VALUES (?,?,?,?,?,?,?)", self.edges)
        con.commit()
        con.close()
        return path


def complete_session():
    f = Fixture()
    f.node("agent", "agent", {"name": "s"})
    f.assistant("a1", "Plan first.", [("c1", "edit", {"path": "src/x.py"}),
                                       ("c2", "bash", {"command": "ls"})])
    f.node("p1", "content", {"content_type": "statement", "body": "Plan first."})
    f.edge("contains", "a1", "p1")
    f.tool("t1", "c1")
    f.edge("approved_by", "t1", "agent")
    f.node("file1", "content", {"content_type": "file", "path": "/work/src/x.py"})
    f.edge("modifies", "a1", "file1")
    f.tool("t2", "c2", content="rejected by user: no")
    f.node("rej", "content", {"content_type": "tool_rejection"})
    f.edge("produces", "a1", "rej")
    f.edge("rejected_by", "rej", "agent")
    f.assistant("a2", "Done.", meta={"tool_call_leak": True})
    return f


class StructureCompletenessTest(unittest.TestCase):
    def report(self, fixture):
        with tempfile.TemporaryDirectory() as d:
            return sc.run([fixture.write(d)], "test")

    def test_complete_session_passes_every_check(self):
        rep = self.report(complete_session())
        self.assertTrue(rep["passes"], rep)
        self.assertEqual(rep["checks"]["C1_reply_parts"]["total"], 2)
        self.assertEqual(rep["checks"]["C4_approvals"]["total"], 2)
        self.assertEqual(rep["reported"]["bash_calls"], 1)

    def test_each_gap_fails_its_own_check(self):
        f = Fixture()
        f.node("agent", "agent", {"name": "s"})
        f.assistant("a1", "Text with no parts.", [("c1", "write", {"path": "a.txt"}),
                                                  ("c2", "read", {"path": "b.txt"})])
        f.tool("t1", "c1")
        f.node("orphan", "interaction", {"role": "user", "content": "hi"})
        f.edge("relates_to", "a1", "missing-node")
        checks = self.report(f)["checks"]
        self.assertEqual(checks["C1_reply_parts"]["ok"], 0)
        self.assertEqual((checks["C2_tool_results"]["ok"], checks["C2_tool_results"]["total"]),
                         (1, 2))
        self.assertEqual(checks["C3_file_changes"]["ok"], 0)
        self.assertEqual(checks["C4_approvals"]["ok"], 0)
        self.assertFalse(checks["C5_produced_by_agent"]["passes"])

    def test_dangling_edge_fails_the_corpus(self):
        f = complete_session()
        f.edge("relates_to", "a1", "missing-node")
        rep = self.report(f)
        self.assertEqual(rep["reported"]["dangling_edges"], 1)
        self.assertFalse(rep["passes"])

    def test_failed_write_is_not_expected_to_modify(self):
        f = Fixture()
        f.node("agent", "agent", {"name": "s"})
        f.assistant("a1", calls=[("c1", "write", {"path": "a.txt"})])
        f.tool("t1", "c1", is_error=True)
        self.assertEqual(self.report(f)["checks"]["C3_file_changes"]["total"], 0)

    def test_path_match_needs_a_whole_component(self):
        self.assertTrue(sc.path_matches("/work/src/x.py", "./src/x.py"))
        self.assertTrue(sc.path_matches("/work/src/x.py", "/work/src/x.py"))
        self.assertFalse(sc.path_matches("/work/src/xx.py", "x.py"))

    def test_string_arguments_are_parsed(self):
        node = {"meta": {"tool_calls": [{"id": "c", "name": "edit",
                                         "arguments": "{\"path\": \"a\"}"}]}}
        self.assertEqual(sc.tool_calls(node), [("c", "edit", {"path": "a"})])


if __name__ == "__main__":
    unittest.main()
