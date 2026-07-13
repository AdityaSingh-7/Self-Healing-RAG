"""
pipeline/bandit.py — Thompson Sampling for Strategy Selection

REPLACES: sorted(strategies, key=success_rate)

WHY THOMPSON SAMPLING:
The old approach always picks the "best" strategy. But:
- What if a strategy is actually good but failed on its first 2 tries? (bad luck)
- What if the best strategy changes over time as docs change?
- How do you discover a better strategy if you never try the "bad" ones?

THOMPSON SAMPLING solves this:
- Model each strategy's success rate as a Beta distribution
- Each decision: SAMPLE from each distribution, pick the highest sample
- Good strategies get picked more (exploitation)
- Uncertain strategies occasionally get picked (exploration)
- Converges to optimal with MATHEMATICAL GUARANTEES

THE MATH:
- Beta(α, β) where α = successes + 1, β = failures + 1
- Sample from Beta(α, β) for each strategy
- Pick strategy with highest sample

Example:
  Strategy A: 8 wins, 2 losses → Beta(9, 3) → samples around 0.75
  Strategy B: 2 wins, 1 loss → Beta(3, 2) → samples vary wildly (0.3 to 0.9)

  Strategy B sometimes samples HIGH because we're uncertain about it.
  That's exploration. Over time, if B is actually good, it'll accumulate wins.
  If it's bad, it'll accumulate losses and stop getting picked.

REFERENCE: "An Empirical Evaluation of Thompson Sampling" (Chapelle & Li, 2011)
"""

import random
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent.parent.parent / "healing.db"


class ThompsonSamplingSelector:
    """
    Multi-armed bandit using Thompson Sampling for strategy selection.

    Each strategy is an "arm" with a Beta(α, β) posterior.
    α = number of successes + 1 (prior)
    β = number of failures + 1 (prior)
    """

    def __init__(self):
        """Load success/failure counts from the database."""
        self._ensure_table()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_table(self):
        conn = self._get_connection()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS bandit_arms (
                strategy_name TEXT PRIMARY KEY,
                successes INTEGER DEFAULT 0,
                failures INTEGER DEFAULT 0,
                total_reward REAL DEFAULT 0.0
            )
        """)
        conn.commit()
        conn.close()

    def select_strategy(self, available_strategies: list) -> list:
        """
        Return strategies ordered by Thompson Sampling.

        For each strategy:
        1. Get its (successes, failures) from DB
        2. Sample from Beta(successes + 1, failures + 1)
        3. Sort by sampled value (highest first)

        Returns all strategies in recommended order (not just top-1),
        so the orchestrator can try them in sequence.
        """
        conn = self._get_connection()

        scored = []
        for strategy in available_strategies:
            row = conn.execute(
                "SELECT successes, failures FROM bandit_arms WHERE strategy_name = ?",
                (strategy.name,),
            ).fetchone()

            if row:
                alpha = row["successes"] + 1  # +1 is the Beta prior
                beta = row["failures"] + 1
            else:
                # No data yet — uniform prior Beta(1, 1)
                alpha = 1
                beta = 1

            # Sample from Beta distribution
            # Higher alpha relative to beta → samples closer to 1.0
            # Uncertain arms (low alpha+beta) → high variance → exploration
            sample = random.betavariate(alpha, beta)

            scored.append((sample, strategy))

        # Sort by sampled score (highest first)
        scored.sort(key=lambda x: x[0], reverse=True)

        conn.close()
        return [strategy for _, strategy in scored]

    def update(self, strategy_name: str, success: bool, reward: float = 0.0):
        """
        Update the arm after observing a result.

        Parameters:
        -----------
        strategy_name : str
            Which strategy was used
        success : bool
            Did it heal successfully (confidence >= threshold)?
        reward : float
            The confidence improvement (for finer-grained tracking)
        """
        conn = self._get_connection()
        conn.execute("""
            INSERT INTO bandit_arms (strategy_name, successes, failures, total_reward)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(strategy_name) DO UPDATE SET
                successes = successes + ?,
                failures = failures + ?,
                total_reward = total_reward + ?
        """, (
            strategy_name,
            int(success), int(not success), reward,
            int(success), int(not success), reward,
        ))
        conn.commit()
        conn.close()

    def get_stats(self) -> list[dict]:
        """Get current arm statistics for display."""
        conn = self._get_connection()
        rows = conn.execute("""
            SELECT strategy_name, successes, failures, total_reward,
                   CAST(successes AS REAL) / MAX(successes + failures, 1) as empirical_rate
            FROM bandit_arms
            ORDER BY empirical_rate DESC
        """).fetchall()
        conn.close()

        return [
            {
                "strategy": row["strategy_name"],
                "successes": row["successes"],
                "failures": row["failures"],
                "total_pulls": row["successes"] + row["failures"],
                "empirical_success_rate": round(row["empirical_rate"], 4),
                "total_reward": round(row["total_reward"], 4),
            }
            for row in rows
        ]

    def get_expected_values(self, available_strategies: list) -> dict:
        """
        Get the EXPECTED value (mean of Beta distribution) for each arm.
        Useful for displaying "what the bandit thinks" without randomness.
        """
        conn = self._get_connection()
        result = {}

        for strategy in available_strategies:
            row = conn.execute(
                "SELECT successes, failures FROM bandit_arms WHERE strategy_name = ?",
                (strategy.name,),
            ).fetchone()

            if row:
                alpha = row["successes"] + 1
                beta = row["failures"] + 1
            else:
                alpha = 1
                beta = 1

            # Expected value of Beta(α, β) = α / (α + β)
            expected = alpha / (alpha + beta)
            # Uncertainty (variance) = αβ / ((α+β)²(α+β+1))
            variance = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))

            result[strategy.name] = {
                "expected_success_rate": round(expected, 4),
                "uncertainty": round(variance ** 0.5, 4),  # Standard deviation
                "alpha": alpha,
                "beta": beta,
            }

        conn.close()
        return result


# Global singleton
bandit = ThompsonSamplingSelector()
