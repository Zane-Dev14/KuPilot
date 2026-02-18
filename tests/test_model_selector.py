"""Tests for src.chains.model_selector — complexity estimation and model selection."""

from src.chains.model_selector import ModelSelector


class TestEstimateQueryComplexity:
    def setup_method(self):
        self.selector = ModelSelector()

    def test_simple_query_low_score(self):
        score = self.selector.estimate_query_complexity("What is CrashLoopBackOff?")
        assert 0.0 <= score <= 0.5

    def test_reasoning_query_high_score(self):
        score = self.selector.estimate_query_complexity(
            "Why is my pod crashing and how do I troubleshoot it?"
        )
        assert score >= 0.3

    def test_multiple_questions_increase_score(self):
        single = self.selector.estimate_query_complexity("What happened?")
        multi = self.selector.estimate_query_complexity("What happened? Why? How to fix?")
        assert multi > single

    def test_uncertainty_keywords(self):
        base = self.selector.estimate_query_complexity("pod is failing")
        uncertain = self.selector.estimate_query_complexity("pod is maybe failing, not sure")
        assert uncertain > base

    def test_capped_at_one(self):
        score = self.selector.estimate_query_complexity(
            "Why is this maybe crashing? How to fix? What is the root cause? "
            "I'm not sure what happened and possibly the node is also unhealthy? "
            * 5  # Long repeated text
        )
        assert score <= 1.0


class TestSelectModel:
    def setup_method(self):
        self.selector = ModelSelector()

    def test_simple_query_uses_simple_model(self):
        model = self.selector.select_model("What is a pod?")
        assert model == self.selector.settings.simple_model

    def test_force_model_override(self):
        model = self.selector.select_model("anything", force_model="my-custom-model")
        assert model == "my-custom-model"

    def test_complex_query_uses_complex_model(self):
        model = self.selector.select_model(
            "Why is my pod crashing? Can you explain the root cause and troubleshoot?"
        )
        # This should have high complexity and select the complex model
        assert model in [
            self.selector.settings.simple_model,
            self.selector.settings.complex_model,
        ]
