"""
Topic Modeler Tool
Uses LDA for topic modeling.
"""
from typing import Dict, Any, Optional, Callable, Awaitable, List
import asyncio
import re


class TopicModeler:
    """
    Topic modeling using Latent Dirichlet Allocation (LDA).
    Discovers main topics and their key terms with full transparency.
    """
    
    def __init__(self):
        self._vectorizer = None
        self._lda_model = None
        self._initialized = False
    
    async def _initialize(self, n_topics: int, progress_callback: Optional[Callable] = None):
        """Initialize the LDA model with specified number of topics."""
        if progress_callback:
            await progress_callback(5, "Initializing topic model", {
                "n_topics": n_topics,
                "status": "loading"
            })
        
        try:
            from sklearn.feature_extraction.text import CountVectorizer
            from sklearn.decomposition import LatentDirichletAllocation
            
            self._vectorizer = CountVectorizer(
                max_df=0.95,
                min_df=2,
                max_features=1000,
                stop_words='english'
            )
            
            self._lda_model = LatentDirichletAllocation(
                n_components=n_topics,
                max_iter=10,
                learning_method='online',
                random_state=42
            )
            
            self._initialized = True
            
            if progress_callback:
                await progress_callback(15, "Topic model initialized", {
                    "status": "ready"
                })
                
        except ImportError:
            raise RuntimeError(
                "scikit-learn not installed. "
                "Install with: pip install scikit-learn"
            )
    
    async def analyze(
        self,
        text: str,
        parameters: Optional[Dict] = None,
        progress_callback: Optional[Callable[[int, str, Optional[Dict]], Awaitable[None]]] = None
    ) -> Dict[str, Any]:
        """
        Discover topics in the given text.
        
        Args:
            text: Text to analyze
            parameters: Optional parameters (n_topics, n_words)
            progress_callback: Callback for progress updates
        
        Returns:
            Topic modeling results with keywords and distributions
        """
        params = parameters or {}
        n_topics = params.get("n_topics", 5)
        n_words = params.get("n_words", 10)
        
        # Initialize model
        await self._initialize(n_topics, progress_callback)
        
        if progress_callback:
            await progress_callback(20, "Preparing documents", {
                "text_length": len(text)
            })
        
        # Split text into documents (paragraphs or sentences)
        documents = self._split_into_documents(text)
        
        if len(documents) < 3:
            # Not enough documents for meaningful topic modeling
            return {
                "topics": [],
                "error": "Text too short for topic modeling. Need multiple paragraphs.",
                "document_count": len(documents),
                "minimum_required": 3
            }
        
        if progress_callback:
            await progress_callback(40, "Vectorizing text", {
                "document_count": len(documents)
            })
        
        # Vectorize
        loop = asyncio.get_event_loop()
        try:
            doc_term_matrix = await loop.run_in_executor(
                None,
                lambda: self._vectorizer.fit_transform(documents)
            )
        except ValueError as e:
            return {
                "topics": [],
                "error": f"Vectorization failed: {str(e)}",
                "document_count": len(documents)
            }
        
        if progress_callback:
            await progress_callback(60, "Fitting LDA model", {
                "vocabulary_size": len(self._vectorizer.vocabulary_)
            })
        
        # Fit LDA
        await loop.run_in_executor(
            None,
            lambda: self._lda_model.fit(doc_term_matrix)
        )
        
        if progress_callback:
            await progress_callback(80, "Extracting topics", {})
        
        # Extract topics
        feature_names = self._vectorizer.get_feature_names_out()
        topics = []
        
        for topic_idx, topic_weights in enumerate(self._lda_model.components_):
            # Get top words for this topic
            top_word_indices = topic_weights.argsort()[:-n_words-1:-1]
            top_words = [
                {
                    "word": feature_names[i],
                    "weight": float(topic_weights[i])
                }
                for i in top_word_indices
            ]
            
            topics.append({
                "topic_id": topic_idx,
                "keywords": [w["word"] for w in top_words],
                "keyword_weights": top_words,
                "coherence_score": self._calculate_topic_coherence(top_words)
            })
        
        # Calculate document-topic distribution
        doc_topic_dist = self._lda_model.transform(doc_term_matrix)
        
        output = {
            "topics": topics,
            "topic_count": len(topics),
            "document_count": len(documents),
            "vocabulary_size": len(self._vectorizer.vocabulary_),
            "document_topic_distribution": {
                "shape": list(doc_topic_dist.shape),
                "dominant_topics": [int(dist.argmax()) for dist in doc_topic_dist]
            },
            "model_parameters": {
                "n_topics": n_topics,
                "n_words": n_words,
                "max_iterations": 10
            },
            "explainability": {
                "method": "lda_topic_modeling",
                "model_type": "Latent Dirichlet Allocation (sklearn)",
                "reasoning": f"Discovered {len(topics)} topics from {len(documents)} "
                            f"text segments. Each topic is characterized by its most "
                            f"representative keywords weighted by importance.",
                "interpretation_guide": "Topics with higher coherence scores are "
                                        "more semantically meaningful."
            }
        }
        
        if progress_callback:
            await progress_callback(100, "Topic modeling complete", {
                "topic_count": len(topics)
            })
        
        return output
    
    def _split_into_documents(self, text: str) -> List[str]:
        """Split text into documents for topic modeling."""
        # First try splitting by paragraphs
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        if len(paragraphs) >= 5:
            return paragraphs
        
        # If not enough paragraphs, split by sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        if len(sentences) >= 10:
            # Group sentences into chunks of 3
            chunks = []
            for i in range(0, len(sentences), 3):
                chunk = ' '.join(sentences[i:i+3])
                if chunk:
                    chunks.append(chunk)
            return chunks
        
        # Return whatever we have
        return paragraphs if paragraphs else sentences
    
    def _calculate_topic_coherence(self, words: List[Dict]) -> float:
        """Calculate a simple coherence score for a topic."""
        # This is a simplified coherence measure
        # A more sophisticated implementation would use PMI or other metrics
        if not words:
            return 0.0
        
        weights = [w["weight"] for w in words]
        avg_weight = sum(weights) / len(weights)
        
        # Normalize to 0-1 range
        return min(avg_weight / 10, 1.0)
