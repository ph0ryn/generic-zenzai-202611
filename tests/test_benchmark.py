import json
import unittest

from generic_zenzai_202611.benchmark import parse_items, score_prediction, summarize


class BenchmarkTest(unittest.TestCase):
    def test_scores_follow_official_exact_match_and_minimum_cer_definition(self):
        for references, hypothesis, accuracy, cer in (
            (["hello"], "hello", 1, 0),
            (["hello"], "hell", 0, 1 / 5),
            (["hello"], "hellao", 0, 1 / 5),
            (["hello"], "hallo", 0, 1 / 5),
            (["hallo", "hollo"], "hello", 0, 1 / 5),
            (["暑い", "熱い"], "熱い", 1, 0),
            (["熱い"], " 熱い", 0, 1 / 2),
            (["熱い"], "", 0, 1),
            (["a"], "aaaa", 0, 3),
            # Minimize normalized distance, not raw distance.
            (["a", "bbbb"], "bb", 0, 1 / 2),
        ):
            with self.subTest(references=references, hypothesis=hypothesis):
                self.assertEqual(
                    score_prediction(references, hypothesis),
                    {"accuracy_at1": accuracy, "min_cer": cer},
                )

    def test_missing_groups_have_null_scores_not_a_fictitious_zero(self):
        scores = summarize(
            [{"context_text": "", "accuracy_at1": 1, "min_cer": 0, "truncated": False}]
        )
        self.assertEqual(scores["all"]["accuracy_at1"], 1)
        self.assertEqual(scores["with_context"]["count"], 0)
        self.assertIsNone(scores["with_context"]["accuracy_at1"])
        self.assertIsNone(scores["with_context"]["mean_min_cer"])

    def test_invalid_data_is_rejected_not_skipped(self):
        row = {
            "index": "a",
            "input": "アツイ",
            "context_text": "",
            "expected_output": ["熱い"],
        }
        self.assertEqual(parse_items(json.dumps([row]).encode()), [row])
        invalid = [[], {}, [row, row]]
        for key, value in (
            ("input", ""),
            ("context_text", None),
            ("expected_output", []),
            ("expected_output", [""]),
            ("expected_output", "熱い"),
            ("index", 1),
        ):
            invalid.append([{**row, key: value}])
        for key in row:
            invalid.append(
                [{name: value for name, value in row.items() if name != key}]
            )
        for rows in invalid:
            with self.subTest(rows=rows), self.assertRaises((TypeError, ValueError)):
                parse_items(json.dumps(rows).encode())


if __name__ == "__main__":
    unittest.main()
