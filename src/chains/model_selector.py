import logging
from src.config import get_settings

logger = logging.getLogger(__name__)

class ModelSelector:
    """Select between simple and complex LLM based on query complexity."""
    
    # Keywords that indicate reasoning-heavy queries
    REASONING_KEYWORDS = [
        "why", "explain", "diagnose", "troubleshoot", "how", "reason",
        "root cause", "what went wrong", "understand",
    ]
    
    # Keywords that indicate uncertainty/high stakes
    UNCERTAINTY_KEYWORDS = [
        "maybe", "could", "might", "possibly", "seems", "appears",
        "not sure", "uncertain", "confused",
    ]
    
    def __init__(self):
        self.settings = get_settings()
    
    def estimate_query_complexity(self, query: str) -> float:
        """
        Estimate query complexity on scale 0.0-1.0.
        
        Heuristics:
        - Reasoning keywords: +0.3
        - Multiple questions: +0.2 per additional question
        - Long query (>200 chars): +0.1
        - Uncertainty keywords: +0.2
        """
        score = 0.0
        query_lower = query.lower()
        
        # Reasoning keywords
        for kw in self.REASONING_KEYWORDS:
            if kw in query_lower:
                score += 0.3
                logger.debug(f"Found reasoning keyword: {kw}")
                break  # Count once
        
        # Multiple questions
        question_count = query_lower.count("?")
        if question_count > 1:
            score += min(0.2 * (question_count - 1), 0.4)
            logger.debug(f"Found {question_count} questions: +{0.2 * (question_count - 1)}")
        
        # Long query
        if len(query) > 200:
            score += 0.1
            logger.debug(f"Long query (>{200} chars): +0.1")
        
        # Uncertainty keywords
        for kw in self.UNCERTAINTY_KEYWORDS:
            if kw in query_lower:
                score += 0.2
                logger.debug(f"Found uncertainty keyword: {kw}")
                break  # Count once
        
        score = min(score, 1.0)  # Cap at 1.0
        logger.info(f"Query complexity: {score:.2f}")
        
        return score
    
    def select_model(self, query: str, force_model: str | None = None) -> str:
        """
        Select model based on query complexity.
        
        Args:
            query: User query
            force_model: Override model selection
        
        Returns:
            Model name (e.g., "llama3.1" or "deepseek-r1:32b")
        """
        if force_model:
            logger.info(f"Using forced model: {force_model}")
            return force_model
        
        complexity = self.estimate_query_complexity(query)
        
        if complexity >= self.settings.query_complexity_threshold:
            model = self.settings.complex_model
            logger.info(f"Selected complex model ({complexity:.2f} >= {self.settings.query_complexity_threshold})")
        else:
            model = self.settings.simple_model
            logger.info(f"Selected simple model ({complexity:.2f} < {self.settings.query_complexity_threshold})")
        
        return model