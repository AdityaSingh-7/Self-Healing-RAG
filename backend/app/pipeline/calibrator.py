"""
pipeline/calibrator.py — Confidence Calibration

PROBLEM: The LLM-as-judge says "CONFIDENCE: 0.8" but is it ACTUALLY 80% accurate?
Usually not. LLMs are poorly calibrated — their stated confidence doesn't match reality.

SOLUTION: Platt Scaling — fit a logistic regression to map raw confidence → actual correctness.

HOW IT WORKS:
1. Collect labeled examples: (raw_confidence, was_actually_correct) pairs
2. Fit logistic regression: P(correct | raw_conf) = sigmoid(a * raw_conf + b)
3. Now when LLM says 0.8, we compute: calibrated = sigmoid(a * 0.8 + b)
   If the LLM is overconfident, calibrated < 0.8. If underconfident, calibrated > 0.8.

WHY THIS MATTERS:
- Uncalibrated: threshold of 0.8 might reject 30% of correct answers (overconfident LLM)
- Calibrated: threshold of 0.8 means ACTUALLY 80% chance of being correct

REFERENCES:
- Platt Scaling (Platt, 1999)
- "Calibration of Pre-trained Transformers" (Desai & Durrett, 2020)
- Temperature Scaling (Guo et al, 2017)
"""

import math
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent.parent.parent / "healing.db"


class ConfidenceCalibrator:
    """
    Maps raw LLM confidence to calibrated probability using Platt Scaling.

    Requires at least 20 labeled examples to be meaningful.
    Below that, returns raw confidence (passthrough).
    """

    MIN_SAMPLES = 20  # Minimum data points before calibration kicks in

    def __init__(self):
        self._ensure_table()
        self._a = 1.0   # Logistic params: sigmoid(a*x + b)
        self._b = 0.0   # Default = identity (no calibration)
        self._is_fitted = False
        self._fit()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_table(self):
        conn = self._get_connection()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS calibration_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                raw_confidence REAL NOT NULL,
                was_correct BOOLEAN NOT NULL,
                question TEXT,
                timestamp TEXT
            )
        """)
        conn.commit()
        conn.close()

    def add_sample(self, raw_confidence: float, was_correct: bool, question: str = ""):
        """
        Add a labeled data point for calibration.

        Call this when you KNOW if an answer was correct (e.g., from user feedback
        or benchmark ground truth).
        """
        from datetime import datetime, timezone

        conn = self._get_connection()
        conn.execute(
            "INSERT INTO calibration_data (raw_confidence, was_correct, question, timestamp) VALUES (?, ?, ?, ?)",
            (raw_confidence, was_correct, question, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
        conn.close()

        # Re-fit after new data
        self._fit()

    def calibrate(self, raw_confidence: float) -> float:
        """
        Map raw LLM confidence → calibrated probability.

        Returns raw confidence if not enough data to calibrate.
        """
        if not self._is_fitted:
            return raw_confidence

        # Platt scaling: P(correct) = sigmoid(a * raw_conf + b)
        z = self._a * raw_confidence + self._b
        calibrated = self._sigmoid(z)

        # Clamp to [0.01, 0.99] to avoid extreme values
        return max(0.01, min(0.99, calibrated))

    def _fit(self):
        """
        Fit Platt Scaling parameters using gradient descent.

        Minimizes log-loss: -Σ [y*log(p) + (1-y)*log(1-p)]
        where p = sigmoid(a*x + b) and y = was_correct
        """
        conn = self._get_connection()
        rows = conn.execute(
            "SELECT raw_confidence, was_correct FROM calibration_data"
        ).fetchall()
        conn.close()

        if len(rows) < self.MIN_SAMPLES:
            self._is_fitted = False
            return

        # Extract data
        X = [row["raw_confidence"] for row in rows]
        y = [int(row["was_correct"]) for row in rows]

        # Gradient descent to fit a, b
        a, b = 1.0, 0.0
        lr = 0.1  # Learning rate

        for _ in range(1000):  # Iterations
            # Compute predictions
            preds = [self._sigmoid(a * xi + b) for xi in X]

            # Compute gradients
            grad_a = sum((preds[i] - y[i]) * X[i] for i in range(len(X))) / len(X)
            grad_b = sum((preds[i] - y[i]) for i in range(len(X))) / len(X)

            # Update
            a -= lr * grad_a
            b -= lr * grad_b

        self._a = a
        self._b = b
        self._is_fitted = True

    def _sigmoid(self, z: float) -> float:
        """Numerically stable sigmoid."""
        if z >= 0:
            return 1.0 / (1.0 + math.exp(-z))
        else:
            ez = math.exp(z)
            return ez / (1.0 + ez)

    def get_calibration_info(self) -> dict:
        """Get current calibration state for debugging/display."""
        conn = self._get_connection()
        count = conn.execute("SELECT COUNT(*) as c FROM calibration_data").fetchone()["c"]
        conn.close()

        return {
            "is_calibrated": self._is_fitted,
            "data_points": count,
            "min_required": self.MIN_SAMPLES,
            "parameters": {"a": round(self._a, 4), "b": round(self._b, 4)} if self._is_fitted else None,
            "example_mappings": {
                "raw_0.5": round(self.calibrate(0.5), 4),
                "raw_0.7": round(self.calibrate(0.7), 4),
                "raw_0.8": round(self.calibrate(0.8), 4),
                "raw_0.9": round(self.calibrate(0.9), 4),
            } if self._is_fitted else None,
        }


# Global singleton
calibrator = ConfidenceCalibrator()
