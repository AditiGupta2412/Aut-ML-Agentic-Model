"""
Text Classifier Tool
Uses zero-shot classification with BART.
"""
from typing import Dict, Any, Optional, Callable, Awaitable, List
import asyncio


class TextClassifier:
    """
    Zero-shot text classification using BART-MNLI.
    Classifies text into arbitrary categories without training.
    """
    
    MODEL_NAME = "facebook/bart-large-mnli"
    
    # Default categories if none provided
    DEFAULT_CATEGORIES = [
        "Technology",
        "Business",
        "Science",
        "Health",
        "Politics",
        "Sports",
        "Entertainment",
        "Education"
    ]
    
    def __init__(self):
        self._pipeline = None
        self._initialized = False
    
    async def _initialize(self, progress_callback: Optional[Callable] = None):
        """Lazy initialization of the model."""
        if self._initialized:
            return
        
        if progress_callback:
            await progress_callback(5, "Initializing classifier", {
                "model": self.MODEL_NAME,
                "status": "loading"
            })
        
        try:
            from transformers import pipeline
            
            loop = asyncio.get_event_loop()
            self._pipeline = await loop.run_in_executor(
                None,
                lambda: pipeline(
                    "zero-shot-classification",
                    model=self.MODEL_NAME
                )
            )
            
            self._initialized = True
            
            if progress_callback:
                await progress_callback(25, "Classifier loaded", {
                    "model": self.MODEL_NAME,
                    "status": "ready"
                })
                
        except ImportError:
            raise RuntimeError(
                "transformers library not installed. "
                "Install with: pip install transformers torch"
            )
    
    async def analyze(
        self,
        text: str,
        parameters: Optional[Dict] = None,
        progress_callback: Optional[Callable[[int, str, Optional[Dict]], Awaitable[None]]] = None
    ) -> Dict[str, Any]:
        """
        Classify text into categories using zero-shot learning.
        
        Args:
            text: Text to classify
            parameters: Optional parameters (categories, multi_label)
            progress_callback: Callback for progress updates
        
        Returns:
            Classification results with scores for each category
        """
        params = parameters or {}
        categories = params.get("categories", self.DEFAULT_CATEGORIES)
        multi_label = params.get("multi_label", False)
        max_length = params.get("max_length", 1024)
        
        # Initialize model
        await self._initialize(progress_callback)
        
        if progress_callback:
            await progress_callback(35, "Preparing classification", {
                "categories": categories,
                "multi_label": multi_label
            })
        
        # Truncate if necessary
        if len(text) > max_length * 4:
            text_to_classify = text[:max_length * 4]
            truncated = True
        else:
            text_to_classify = text
            truncated = False
        
        if progress_callback:
            await progress_callback(50, "Running classification", {
                "text_length": len(text_to_classify),
                "category_count": len(categories)
            })
        
        # Run classification
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self._pipeline(
                text_to_classify,
                candidate_labels=categories,
                multi_label=multi_label
            )
        )
        
        if progress_callback:
            await progress_callback(85, "Processing results", {})
        
        # Format results
        categories_result = []
        for label, score in zip(result["labels"], result["scores"]):
            categories_result.append({
                "label": label,
                "score": round(float(score), 4),
                "confidence_level": self._get_confidence_level(score)
            })
        
        # Determine primary classification
        top_category = categories_result[0] if categories_result else None
        
        output = {
            "categories": categories_result,
            "top_category": top_category["label"] if top_category else None,
            "top_confidence": top_category["score"] if top_category else 0.0,
            "confidence": top_category["score"] if top_category else 0.0,
            "multi_label": multi_label,
            "model": self.MODEL_NAME,
            "text_length": len(text_to_classify),
            "was_truncated": truncated,
            "candidate_labels": categories,
            "explainability": {
                "method": "zero_shot_classification",
                "model_type": "BART fine-tuned on MNLI",
                "reasoning": f"Classified as '{top_category['label']}' with "
                            f"{top_category['score']:.0%} confidence. "
                            f"Zero-shot classification works by determining "
                            f"how well the text entails each candidate category "
                            f"using natural language inference."
                            if top_category else "No classification result",
                "alternative_categories": [
                    f"{c['label']}: {c['score']:.0%}"
                    for c in categories_result[1:4]
                ]
            }
        }
        
        if progress_callback:
            await progress_callback(100, "Classification complete", {
                "top_category": output["top_category"],
                "confidence": output["top_confidence"]
            })
        
        return output
    
    def _get_confidence_level(self, score: float) -> str:
        """Get human-readable confidence level."""
        if score >= 0.8:
            return "High"
        elif score >= 0.5:
            return "Medium"
        elif score >= 0.3:
            return "Low"
        else:
            return "Very Low"
