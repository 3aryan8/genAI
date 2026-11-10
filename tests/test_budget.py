"""Tests for the verified HuBERT LoRA parameter budget."""

from __future__ import annotations

import unittest

from src.budget import lora_param_cost, total_lora_params, uniform_test_rank_map


class LoraBudgetTests(unittest.TestCase):
    def test_uniform_hubert_target_budgets(self) -> None:
        self.assertEqual(total_lora_params(uniform_test_rank_map(4)), 663_552)
        self.assertEqual(total_lora_params(uniform_test_rank_map(16)), 2_654_208)

    def test_rank_step_costs(self) -> None:
        self.assertEqual(lora_param_cost((768, 3072), 1), 3840)
        self.assertEqual(lora_param_cost((768, 768), 1), 1536)


if __name__ == "__main__":
    unittest.main()
