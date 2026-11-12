"""Uji API Flask memakai predictor palsu, sehingga bisa dijalankan tanpa PyTorch dan tanpa file .pth.

Jalankan dari folder proyek:  python -m unittest discover -s tests
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import create_app, group_features, parse_features  # noqa: E402


class FakePredictor:
    meta = {
        "feature_names": ["mean radius", "radius error", "worst radius"],
        "feature_min": [6.0, 0.1, 7.0],
        "feature_max": [28.0, 2.9, 36.0],
        "class_names": ["Malignant (ganas)", "Benign (jinak)"],
        "examples": {
            "benign": {"mean radius": 10.0, "radius error": 0.2, "worst radius": 11.0},
            "malignant": {"mean radius": 20.0, "radius error": 1.0, "worst radius": 25.0},
        },
        "metrics": {"test_accuracy": 0.97, "f1": 0.98},
    }

    def predict(self, values):
        return 0.9 if values[0] < 15 else 0.1


class AppTest(unittest.TestCase):
    def setUp(self):
        self.client = create_app(FakePredictor()).test_client()

    def post(self, features):
        return self.client.post("/api/predict", json={"features": features})

    def test_index_and_health(self):
        self.assertEqual(self.client.get("/").status_code, 200)
        self.assertEqual(self.client.get("/health").get_json()["status"], "ok")

    def test_predict_benign_and_malignant(self):
        ok = self.post({"mean radius": 10, "radius error": "0.2", "worst radius": 11}).get_json()
        self.assertEqual(ok["label_index"], 1)
        self.assertAlmostEqual(sum(ok["probabilities"].values()), 1.0)
        bad = self.post({"mean radius": 20, "radius error": 1, "worst radius": 25}).get_json()
        self.assertEqual(bad["label_index"], 0)

    def test_out_of_range_is_reported(self):
        data = self.post({"mean radius": 99, "radius error": 1, "worst radius": 25}).get_json()
        self.assertEqual(data["out_of_range"], ["mean radius"])

    def test_invalid_inputs_are_rejected(self):
        cases = [
            {"mean radius": 10, "radius error": 1},                                   # fitur hilang
            {"mean radius": "abc", "radius error": 1, "worst radius": 2},             # bukan angka
            {"mean radius": -1, "radius error": 1, "worst radius": 2},                # negatif
            {"mean radius": "", "radius error": 1, "worst radius": 2},                # kosong
            {"mean radius": float("inf"), "radius error": 1, "worst radius": 2},      # tak hingga
            {"mean radius": True, "radius error": 1, "worst radius": 2},              # boolean
        ]
        for features in cases:
            with self.subTest(features=features):
                res = self.client.post("/api/predict", json={"features": features})
                self.assertEqual(res.status_code, 400)

    def test_non_json_body_is_rejected(self):
        res = self.client.post("/api/predict", data="bukan json", content_type="text/plain")
        self.assertEqual(res.status_code, 400)

    def test_helpers(self):
        groups = group_features(FakePredictor.meta)
        self.assertEqual([len(g[1]) for g in groups], [1, 1, 1])
        values, errors = parse_features({"features": {"a": "1.5"}}, ["a"])
        self.assertEqual((values, errors), ([1.5], []))


if __name__ == "__main__":
    unittest.main()
