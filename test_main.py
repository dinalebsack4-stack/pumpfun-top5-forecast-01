import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import main


class ForecastIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.input_dir = self.root / "input"
        self.output_dir = self.root / "output"
        self.input_dir.mkdir()

        task = {
            "output": "forecast.json",
            "places": 5,
            "reads": [
                "2026-01-01T01:00:00Z",
                "2026-01-01T06:00:00Z",
            ],
            "size": {"edges": [1.0, 2.0, 3.0, 4.0, 5.0]},
        }
        (self.input_dir / "task.json").write_text(json.dumps(task))

        # Eight observed candidates across two marks; values are lamports.
        marks = []
        for mark_index, timestamp in enumerate(
            ("2026-01-01T00:00:00Z", "2026-01-01T00:30:00Z")
        ):
            rows = [
                {
                    "mint": f"mint-{i}",
                    "rank": i + 1,
                    "market_cap_lamports": str((i + 1 + mark_index) * 10**10),
                }
                for i in range(8)
            ]
            marks.append({"at": timestamp, "ok": True, "rows": rows})
        (self.input_dir / "history.ndjson").write_text(
            "".join(json.dumps(mark) + "\n" for mark in marks)
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def run_forecast(self):
        with patch.object(main, "INPUT_DIR", self.input_dir), patch.object(
            main, "OUTPUT_DIR", self.output_dir
        ):
            main.main()
        return json.loads((self.output_dir / "forecast.json").read_text())

    def test_output_schema_and_probability_invariants(self):
        result = self.run_forecast()

        self.assertEqual(set(result), {"probabilities", "top5", "sizes"})
        self.assertEqual(len(result["top5"]), 5)
        self.assertEqual(len(set(result["top5"])), 5)
        self.assertTrue(all(isinstance(mint, str) for mint in result["top5"]))

        for place in range(1, 6):
            named_mass = sum(
                item["p"]
                for item in result["probabilities"]
                if item["place"] == place
            )
            self.assertGreaterEqual(named_mass, 0.0)
            self.assertLessEqual(named_mass, 1.0)

            size_mass = sum(
                item["p"] for item in result["sizes"] if item["place"] == place
            )
            self.assertAlmostEqual(size_mass, 1.0, places=6)

        for item in result["probabilities"] + result["sizes"]:
            self.assertGreaterEqual(item["p"], 0.0)
            self.assertLessEqual(item["p"], 1.0)

    def test_output_is_deterministic(self):
        first = self.run_forecast()
        (self.output_dir / "forecast.json").unlink()
        second = self.run_forecast()
        self.assertEqual(first, second)

    def test_fails_when_history_has_no_successful_marks(self):
        (self.input_dir / "history.ndjson").write_text(
            json.dumps({"at": "2026-01-01T00:00:00Z", "ok": False, "rows": []})
            + "\n"
        )
        with patch.object(main, "INPUT_DIR", self.input_dir), patch.object(
            main, "OUTPUT_DIR", self.output_dir
        ):
            with self.assertRaisesRegex(RuntimeError, "no successful marks"):
                main.main()


if __name__ == "__main__":
    unittest.main()
