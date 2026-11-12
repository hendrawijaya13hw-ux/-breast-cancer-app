"""Aplikasi web klasifikasi Breast Cancer (Flask) yang memakai model PyTorch (.pth).

Jalankan:  python app.py   lalu buka http://127.0.0.1:5000
"""
import math
import os
from pathlib import Path

from flask import Flask, jsonify, render_template, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.environ.get("MODEL_PATH", BASE_DIR / "model" / "breast_cancer_model.pth"))

GROUP_TITLES = [
    ("mean", "Nilai rata-rata (mean)"),
    ("error", "Standard error"),
    ("worst", "Nilai terburuk (worst)"),
    ("other", "Lainnya"),
]


def group_features(meta):
    """Kelompokkan fitur berdasarkan awalan/akhiran namanya agar form mudah dibaca."""
    groups = {key: [] for key, _ in GROUP_TITLES}
    for i, name in enumerate(meta["feature_names"]):
        if name.startswith("mean "):
            key, label = "mean", name[len("mean "):]
        elif name.endswith(" error"):
            key, label = "error", name[: -len(" error")]
        elif name.startswith("worst "):
            key, label = "worst", name[len("worst "):]
        else:
            key, label = "other", name
        groups[key].append({
            "id": f"f{i}",
            "name": name,
            "label": label.capitalize(),
            "min": meta["feature_min"][i],
            "max": meta["feature_max"][i],
        })
    return [(title, groups[key]) for key, title in GROUP_TITLES if groups[key]]


def parse_features(payload, names):
    """Validasi input JSON. Mengembalikan (daftar_nilai, daftar_error)."""
    features = payload.get("features") if isinstance(payload, dict) else None
    if not isinstance(features, dict):
        return None, ["Body harus berupa JSON dengan objek 'features'."]
    values, errors = [], []
    for name in names:
        raw = features.get(name)
        if raw is None or isinstance(raw, bool) or (isinstance(raw, str) and not raw.strip()):
            errors.append(f"'{name}' belum diisi.")
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            errors.append(f"'{name}' bukan angka.")
            continue
        if not math.isfinite(value) or value < 0:
            errors.append(f"'{name}' harus angka bernilai 0 atau lebih.")
            continue
        values.append(value)
    return values, errors


def create_app(predictor=None):
    app = Flask(__name__)

    if predictor is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"File model tidak ditemukan: {MODEL_PATH}\n"
                "Unduh breast_cancer_model.pth dari Google Colab lalu letakkan di folder model/ "
                "(lihat README.md)."
            )
        from model import Predictor  # diimpor di sini agar pengujian tanpa PyTorch tetap bisa jalan

        predictor = Predictor(MODEL_PATH)

    meta = predictor.meta
    names = meta["feature_names"]

    @app.get("/")
    def index():
        return render_template(
            "index.html",
            groups=group_features(meta),
            examples=meta["examples"],
            metrics=meta.get("metrics", {}),
            n_features=len(names),
        )

    @app.get("/health")
    def health():
        return jsonify(status="ok", n_features=len(names))

    @app.post("/api/predict")
    def predict():
        payload = request.get_json(silent=True)
        values, errors = parse_features(payload, names)
        if errors:
            return jsonify(error="Input tidak valid.", details=errors), 400

        p_benign = float(predictor.predict(values))
        p_malignant = 1.0 - p_benign
        label_index = 1 if p_benign >= 0.5 else 0
        out_of_range = [
            n for n, v, lo, hi in zip(names, values, meta["feature_min"], meta["feature_max"])
            if v < lo or v > hi
        ]
        return jsonify(
            label=meta["class_names"][label_index],
            label_index=label_index,
            probabilities={"malignant": p_malignant, "benign": p_benign},
            confidence=max(p_benign, p_malignant),
            out_of_range=out_of_range,
        )

    return app


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    create_app().run(host="127.0.0.1", port=port, debug=False)
