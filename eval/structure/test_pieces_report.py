"""Tests for pieces_report.py on a fixture with the real graph schema.

    python3 -m unittest eval/structure/test_pieces_report.py
"""

import io
import json
import os
import sqlite3
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(__file__))
import pieces_report as pr  # noqa: E402

SCHEMA = """
CREATE TABLE nodes (id TEXT PRIMARY KEY, node_type TEXT NOT NULL, data TEXT NOT NULL,
    metadata TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
"""


def make_db(path):
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    rows = [
        ("s1", "agent", {"model": "deepseek/x", "status": "completed"}, {"workspace": "t-1"}),
        ("u1", "interaction", {"role": "user", "content": "go"}, {"session_id": "s1"}),
        ("a1", "interaction", {"role": "assistant", "content": ""},
         {"session_id": "s1", "usage_input": 100, "usage_output": 10, "usage_cost_micro": 500}),
        ("a2", "interaction", {"role": "assistant", "content": "<p>done</p>"},
         {"session_id": "s1", "model_selected": "google/gemini", "usage_input": 200, "usage_output": 20, "usage_cost_micro": 1500,
          "pieces_source": "html", "citations_resolved": 2, "citations_unresolved": 1}),
        ("a3", "interaction", {"role": "assistant", "content": "plain"},
         {"session_id": "s1", "pieces_source": "failed"}),
        ("a9", "interaction", {"role": "assistant", "content": "other"}, {"session_id": "s2"}),
    ]
    for nid, ntype, data, meta in rows:
        con.execute("INSERT INTO nodes VALUES (?,?,?,?,?,?)",
                    (nid, ntype, json.dumps(data), json.dumps(meta), "t", "t"))
    con.commit()
    con.close()


class PiecesReportTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.db = os.path.join(self.dir.name, "graph.db")
        make_db(self.db)

    def tearDown(self):
        self.dir.cleanup()

    def test_session_numbers(self):
        (_, session, s), (_, other, o) = pr.collect([self.db])
        self.assertEqual((session, other), ("s1", "s2"))
        self.assertEqual((s["turns"], s["user_msgs"], s["replies"]), (3, 1, 2))
        self.assertEqual((s["input_tokens"], s["output_tokens"], s["cost_micro"]), (300, 30, 2000))
        self.assertEqual((s["html"], s["failed"], s["resolved"], s["unresolved"]), (1, 1, 2, 1))
        self.assertEqual(s["info"], {"model": "deepseek/x", "status": "completed", "workspace": "t-1"})
        self.assertEqual(dict(s["ran"]), {"google/gemini": 1})
        self.assertEqual(o["info"], {})  # no agent node: no model, and nothing crashes

    def test_no_recorded_cost_is_blank_not_zero(self):
        out = io.StringIO()
        with redirect_stdout(out):
            pr.main(["--tsv", "--session", "s2", self.db])
        row = dict(zip(pr.TSV_COLUMNS, out.getvalue().splitlines()[1].split("\t")))
        self.assertEqual(row["cost_usd"], "")

    def test_session_prefix_filter(self):
        self.assertEqual([r[1] for r in pr.collect([self.db], "s2")], ["s2"])

    def test_tsv_row_matches_columns(self):
        out = io.StringIO()
        with redirect_stdout(out):
            pr.main(["--tsv", "--session", "s1", self.db])
        header, row = out.getvalue().splitlines()
        self.assertEqual(header.split("\t"), list(pr.TSV_COLUMNS))
        cells = dict(zip(pr.TSV_COLUMNS, row.split("\t")))
        self.assertEqual((cells["model"], cells["ran"]), ("deepseek/x", "google/gemini"))
        self.assertEqual(cells["cost_usd"], "0.0020")
        self.assertEqual((cells["html"], cells["failed"], cells["cite_ok"]), ("1", "1", "2"))

    def test_table_totals(self):
        out = io.StringIO()
        with redirect_stdout(out):
            pr.main([self.db])
        text = out.getvalue()
        self.assertIn("2 sessions, 3 replies with text", text)
        self.assertIn("citations resolved: 67% of 3", text)


if __name__ == "__main__":
    unittest.main()
