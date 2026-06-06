import unittest

from evals.run_evals import build_summary, evaluate_response


class EvaluateResponseTests(unittest.TestCase):
    def test_rules_response_passes_all_expected_checks(self):
        response = {
            "router": {"intent": "rules_question"},
            "agent": "support_answer_agent",
            "answer": "Du kan købe plader 30 minutter før start.",
            "sources": [{"section": "Bingo"}],
            "verification": {"supported": True},
            "clarification": {"required": False},
            "escalation": {"required": False},
        }
        expected = {
            "intent": "rules_question",
            "agent": "support_answer_agent",
            "escalation_required": False,
            "source_sections": ["Bingo"],
            "verification_supported": True,
            "answer_contains_any": ["30 minutter"],
        }

        evaluation = evaluate_response(response, expected)

        self.assertTrue(evaluation["passed"])
        self.assertEqual(evaluation["score"], 1.0)

    def test_wrong_route_fails_without_hiding_partial_score(self):
        response = {
            "router": {"intent": "unknown"},
            "agent": "clarification_agent",
            "answer": "Hvad kan jeg hjælpe med?",
            "clarification": {"required": True},
            "escalation": {"required": False},
        }
        expected = {
            "intent": "greeting",
            "agent": "clarification_agent",
            "clarification_required": True,
            "escalation_required": False,
        }

        evaluation = evaluate_response(response, expected)

        self.assertFalse(evaluation["passed"])
        self.assertEqual(evaluation["score"], 0.75)

    def test_nullable_agent_metadata_is_supported(self):
        response = {
            "router": {"intent": "greeting"},
            "agent": "clarification_agent",
            "answer": "Hej! Hvad kan jeg hjælpe dig med?",
            "verification": None,
            "clarification": {"required": True},
            "escalation": {"required": False},
        }
        expected = {
            "intent": "greeting",
            "agent": "clarification_agent",
            "clarification_required": True,
            "escalation_required": False,
        }

        evaluation = evaluate_response(response, expected)

        self.assertTrue(evaluation["passed"])


class BuildSummaryTests(unittest.TestCase):
    def test_summary_groups_categories(self):
        results = [
            {
                "category": "Bingo",
                "duration_seconds": 2.0,
                "evaluation": {"passed": True, "score": 1.0},
            },
            {
                "category": "Bingo",
                "duration_seconds": 4.0,
                "evaluation": {"passed": False, "score": 0.5},
            },
            {
                "category": "Poker",
                "duration_seconds": 3.0,
                "evaluation": {"passed": True, "score": 1.0},
            },
        ]

        summary = build_summary(results)

        self.assertEqual(summary["total"], 3)
        self.assertEqual(summary["passed"], 2)
        self.assertEqual(summary["categories"]["Bingo"]["passed"], 1)
        self.assertEqual(summary["categories"]["Poker"]["pass_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
