"""
Sentiment Analyzer Tool
Uses transformers for sentiment analysis.
"""
from typing import Dict, Any, Optional, Callable, Awaitable
import asyncio


class SentimentAnalyzer:
    """
    Sentiment analysis using DistilBERT.
    Provides step-by-step explainability for glass-box transparency.
    """
    
    MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
    
    def __init__(self):
        self._pipeline = None
        self._initialized = False
    
    async def _initialize(self, progress_callback: Optional[Callable] = None):
        """Lazy initialization of the model."""
        if self._initialized:
            return
        
        if progress_callback:
            await progress_callback(5, "Initializing sentiment model", {
                "model": self.MODEL_NAME,
                "status": "loading"
            })
        
        try:
            from transformers import pipeline
            
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            self._pipeline = await loop.run_in_executor(
                None,
                lambda: pipeline("sentiment-analysis", model=self.MODEL_NAME)
            )
            
            self._initialized = True
            
            if progress_callback:
                await progress_callback(20, "Model loaded", {
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
        Analyze sentiment of the given text.
        
        Args:
            text: Text to analyze
            parameters: Optional parameters (e.g., max_length)
            progress_callback: Callback for progress updates
        
        Returns:
            Sentiment analysis result with explainability data
        """
        params = parameters or {}
        max_length = params.get("max_length", 512)
        
        # Initialize model
        await self._initialize(progress_callback)
        
        if progress_callback:
            await progress_callback(30, "Preprocessing text", {
                "original_length": len(text),
                "max_length": max_length
            })
        
        # Truncate if necessary
        if len(text) > max_length * 4:  # Rough character estimate
            text_to_analyze = text[:max_length * 4]
            truncated = True
        else:
            text_to_analyze = text
            truncated = False
        
        if progress_callback:
            await progress_callback(50, "Running sentiment analysis", {
                "text_length": len(text_to_analyze),
                "truncated": truncated
            })
        
        # Run inference
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            lambda: self._pipeline(text_to_analyze)
        )
        
        if progress_callback:
            await progress_callback(80, "Processing results", {
                "raw_results": results
            })
        
        # Format results
        result = results[0] if results else {}
        
        # Calculate confidence scores for both labels
        label = result.get("label", "NEUTRAL")
        score = result.get("score", 0.0)
        
        # Normalize label
        if label.upper() in ["POSITIVE", "POS", "1"]:
            sentiment_label = "POSITIVE"
            positive_score = score
            negative_score = 1 - score
        elif label.upper() in ["NEGATIVE", "NEG", "0"]:
            sentiment_label = "NEGATIVE"
            negative_score = score
            positive_score = 1 - score
        else:
            sentiment_label = "NEUTRAL"
            positive_score = 0.5
            negative_score = 0.5
        
        output = {
            "label": sentiment_label,
            "confidence": score,
            "scores": {
                "positive": round(positive_score, 4),
                "negative": round(negative_score, 4)
            },
            "model": self.MODEL_NAME,
            "text_analyzed_length": len(text_to_analyze),
            "was_truncated": truncated,
            "explainability": {
                "method": "transformer_classification",
                "model_type": "DistilBERT fine-tuned on SST-2",
                "reasoning": f"The model classified this text as {sentiment_label} "
                            f"with {score:.0%} confidence based on learned patterns "
                            f"from the Stanford Sentiment Treebank dataset."
            }
        }
        
        if progress_callback:
            await progress_callback(100, "Analysis complete", output)
        
        return output
