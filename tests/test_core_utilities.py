from __future__ import annotations

import math
import unittest
from pathlib import Path

import numpy as np


class CoreUtilityTests(unittest.TestCase):
    def test_pair_builder_is_deterministic_and_balanced(self) -> None:
        from research_v2.src.data import build_verification_pairs

        paths = [Path(f"person_{label}_{idx}.jpg") for label in range(3) for idx in range(2)]
        labels = [label for label in range(3) for _ in range(2)]

        first = build_verification_pairs(paths, labels, n_pos=4, n_neg=4, seed=7)
        second = build_verification_pairs(paths, labels, n_pos=4, n_neg=4, seed=7)

        self.assertEqual(first.pairs, second.pairs)
        self.assertEqual(len(first), 8)
        self.assertEqual(sum(label for _, _, label in first.pairs), 4)

    def test_pair_builder_reports_invalid_protocols(self) -> None:
        from research_v2.src.data import build_verification_pairs

        with self.assertRaisesRegex(ValueError, "same length"):
            build_verification_pairs([Path("a.jpg")], [], n_pos=0, n_neg=0)

        with self.assertRaisesRegex(ValueError, "positive pairs"):
            build_verification_pairs([Path("a.jpg"), Path("b.jpg")], [0, 1], n_pos=1, n_neg=0)

        with self.assertRaisesRegex(ValueError, "negative pairs"):
            build_verification_pairs([Path("a.jpg"), Path("b.jpg")], [0, 0], n_pos=0, n_neg=1)

    def test_pairset_split_rejects_invalid_fold_counts(self) -> None:
        from research_v2.src.data import PairSet

        pairs = PairSet([(Path("a.jpg"), Path("b.jpg"), 1)])
        with self.assertRaisesRegex(ValueError, "cannot split"):
            list(pairs.split(10))
        with self.assertRaisesRegex(ValueError, "positive"):
            list(pairs.split(0))

    def test_identity_balanced_sampler_batches_are_well_formed(self) -> None:
        from research_v2.src.data import IdentityBalancedSampler

        sampler = IdentityBalancedSampler(
            labels=[0, 0, 1, 1, 2, 2],
            classes_per_batch=2,
            samples_per_class=2,
            num_batches=3,
            seed=11,
        )

        batches = list(sampler)
        self.assertEqual(len(batches), 3)
        self.assertTrue(all(len(batch) == 4 for batch in batches))
        self.assertTrue(all(len(set(batch)) == 4 for batch in batches))

    def test_metrics_handle_clear_separation(self) -> None:
        from research_v2.src.eval import score_pairs, summarize_run

        left = np.eye(3, dtype=np.float32)
        right = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
        labels = np.array([1, 1, 0], dtype=np.int32)

        scores = score_pairs(left, right)
        summary = summarize_run(scores, labels)

        self.assertGreater(summary["auc"], 0.99)
        self.assertLess(summary["eer"], 0.5)
        self.assertEqual(summary["n_pairs"], 3)

    def test_empty_quick_auc_is_dependency_light(self) -> None:
        from research_v2.src.eval import quick_verification_auc

        value = quick_verification_auc(
            encode=lambda batch: batch,
            pairs=[],
            device="cpu",
        )
        self.assertTrue(math.isnan(value))


if __name__ == "__main__":
    unittest.main()
