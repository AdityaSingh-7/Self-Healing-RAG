"""
strategies/base.py — Base Strategy Interface

Every healing strategy follows the same pattern:
1. Take the original question + failed results
2. Try a different retrieval approach
3. Return new results

This base class defines the contract all strategies must follow.
"""

from abc import ABC, abstractmethod


class HealingStrategy(ABC):
    """
    Base class for all healing strategies.

    Every strategy must implement:
    - name: unique identifier
    - description: what it does (for logging)
    - execute(): run the strategy and return new results
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique strategy identifier."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of what this strategy does."""
        ...

    @abstractmethod
    async def execute(
        self,
        question: str,
        original_results: list[dict],
        user_id: str,
        validation_issues: str,
    ) -> dict:
        """
        Execute the healing strategy.

        Parameters:
        -----------
        question : str
            The original question
        original_results : list[dict]
            The results from the failed attempt
        user_id : str
            For namespace isolation
        validation_issues : str
            What the validator said was wrong (guides strategy)

        Returns:
        --------
        dict with:
            - results: list[dict] (new search results)
            - modified_question: str (the query used, may differ from original)
            - metadata: dict (strategy-specific info for logging)
        """
        ...
