"""
General Analyzer Tool
Combines multiple analysis techniques for comprehensive text analysis.
"""
from typing import Dict, Any, Optional, Callable, Awaitable
import asyncio
import re
from collections import Counter


class GeneralAnalyzer:
    """
    General purpose text analyzer that combines multiple techniques.
    Provides an overview when no specific task is detected.
    """
    
    def __init__(self):
        pass
    
    async def analyze(
        self,
        text: str,
        parameters: Optional[Dict] = None,
        progress_callback: Optional[Callable[[int, str, Optional[Dict]], Awaitable[None]]] = None
    ) -> Dict[str, Any]:
        """
        Perform general text analysis.
        
        Args:
            text: Text to analyze
            parameters: Optional parameters
            progress_callback: Callback for progress updates
        
        Returns:
            Comprehensive analysis results
        """
        if progress_callback:
            await progress_callback(10, "Starting general analysis", {
                "text_length": len(text)
            })
        
        results = {}
        
        # Basic statistics
        if progress_callback:
            await progress_callback(20, "Computing basic statistics", {})
        
        results["statistics"] = await self._compute_statistics(text)
        
        # Word frequency
        if progress_callback:
            await progress_callback(40, "Analyzing word frequency", {})
        
        results["word_frequency"] = await self._analyze_word_frequency(text)
        
        # Sentence analysis
        if progress_callback:
            await progress_callback(60, "Analyzing sentences", {})
        
        results["sentence_analysis"] = await self._analyze_sentences(text)
        
        # Readability
        if progress_callback:
            await progress_callback(80, "Computing readability", {})
        
        results["readability"] = await self._compute_readability(text)
        
        # Summary
        results["summary"] = self._generate_summary(results)
        results["confidence"] = 0.85  # General analysis is always fairly confident
        
        results["explainability"] = {
            "method": "statistical_analysis",
            "model_type": "Rule-based and statistical methods",
            "reasoning": "Combined multiple text analysis techniques including "
                        "basic statistics, word frequency, sentence structure, "
                        "and readability metrics for comprehensive overview."
        }
        
        if progress_callback:
            await progress_callback(100, "General analysis complete", results["summary"])
        
        return results
    
    async def _compute_statistics(self, text: str) -> Dict[str, Any]:
        """Compute basic text statistics."""
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        return {
            "character_count": len(text),
            "character_count_no_spaces": len(text.replace(" ", "")),
            "word_count": len(words),
            "sentence_count": len(sentences),
            "paragraph_count": len(paragraphs),
            "average_word_length": round(sum(len(w) for w in words) / max(len(words), 1), 2),
            "average_sentence_length": round(len(words) / max(len(sentences), 1), 2)
        }
    
    async def _analyze_word_frequency(self, text: str) -> Dict[str, Any]:
        """Analyze word frequency distribution."""
        # Clean and tokenize
        words = re.findall(r'\b[a-zA-Z]{2,}\b', text.lower())
        
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
            'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
            'this', 'that', 'these', 'those', 'it', 'its', 'they', 'them', 'their',
            'he', 'she', 'we', 'you', 'i', 'me', 'my', 'your', 'our', 'us'
        }
        
        content_words = [w for w in words if w not in stop_words]
        
        word_freq = Counter(content_words)
        top_words = word_freq.most_common(20)
        
        return {
            "total_content_words": len(content_words),
            "unique_words": len(set(content_words)),
            "vocabulary_richness": round(len(set(content_words)) / max(len(content_words), 1), 3),
            "top_words": [{"word": w, "count": c} for w, c in top_words],
            "hapax_legomena": len([w for w, c in word_freq.items() if c == 1])
        }
    
    async def _analyze_sentences(self, text: str) -> Dict[str, Any]:
        """Analyze sentence structure."""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return {"error": "No sentences found"}
        
        sentence_lengths = [len(s.split()) for s in sentences]
        
        # Sentence type detection
        question_count = text.count('?')
        exclamation_count = text.count('!')
        statement_count = max(len(sentences) - question_count - exclamation_count, 0)
        
        return {
            "sentence_count": len(sentences),
            "shortest_sentence": min(sentence_lengths),
            "longest_sentence": max(sentence_lengths),
            "average_length": round(sum(sentence_lengths) / len(sentence_lengths), 2),
            "sentence_types": {
                "statements": statement_count,
                "questions": question_count,
                "exclamations": exclamation_count
            },
            "length_distribution": {
                "short (1-10 words)": len([l for l in sentence_lengths if l <= 10]),
                "medium (11-20 words)": len([l for l in sentence_lengths if 10 < l <= 20]),
                "long (21+ words)": len([l for l in sentence_lengths if l > 20])
            }
        }
    
    async def _compute_readability(self, text: str) -> Dict[str, Any]:
        """Compute readability metrics."""
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not words or not sentences:
            return {"error": "Insufficient text for readability analysis"}
        
        # Syllable count (simplified)
        def count_syllables(word):
            word = word.lower()
            count = 0
            vowels = 'aeiou'
            prev_vowel = False
            for char in word:
                is_vowel = char in vowels
                if is_vowel and not prev_vowel:
                    count += 1
                prev_vowel = is_vowel
            return max(count, 1)
        
        total_syllables = sum(count_syllables(w) for w in words)
        
        # Flesch Reading Ease
        words_per_sentence = len(words) / max(len(sentences), 1)
        syllables_per_word = total_syllables / max(len(words), 1)
        
        flesch_score = 206.835 - (1.015 * words_per_sentence) - (84.6 * syllables_per_word)
        flesch_score = max(0, min(100, flesch_score))
        
        # Determine reading level
        if flesch_score >= 90:
            level = "Very Easy (5th grade)"
        elif flesch_score >= 80:
            level = "Easy (6th grade)"
        elif flesch_score >= 70:
            level = "Fairly Easy (7th grade)"
        elif flesch_score >= 60:
            level = "Standard (8th-9th grade)"
        elif flesch_score >= 50:
            level = "Fairly Difficult (10th-12th grade)"
        elif flesch_score >= 30:
            level = "Difficult (College)"
        else:
            level = "Very Difficult (Graduate)"
        
        return {
            "flesch_reading_ease": round(flesch_score, 1),
            "reading_level": level,
            "words_per_sentence": round(words_per_sentence, 1),
            "syllables_per_word": round(syllables_per_word, 2),
            "complex_words": len([w for w in words if count_syllables(w) >= 3]),
            "interpretation": f"This text has a Flesch Reading Ease score of {flesch_score:.0f}, "
                             f"indicating a {level.lower()} reading level."
        }
    
    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate an overall summary of the analysis."""
        stats = results.get("statistics", {})
        readability = results.get("readability", {})
        word_freq = results.get("word_frequency", {})
        
        return {
            "text_length": f"{stats.get('word_count', 0)} words, "
                          f"{stats.get('sentence_count', 0)} sentences",
            "readability_level": readability.get("reading_level", "Unknown"),
            "vocabulary_richness": f"{word_freq.get('vocabulary_richness', 0):.1%}",
            "key_terms": [w["word"] for w in word_freq.get("top_words", [])[:5]],
            "overall_assessment": self._generate_assessment(stats, readability, word_freq)
        }
    
    def _generate_assessment(
        self, 
        stats: Dict, 
        readability: Dict, 
        word_freq: Dict
    ) -> str:
        """Generate a human-readable assessment."""
        parts = []
        
        word_count = stats.get("word_count", 0)
        if word_count < 100:
            parts.append("This is a short text")
        elif word_count < 500:
            parts.append("This is a medium-length text")
        else:
            parts.append("This is a longer text")
        
        flesch = readability.get("flesch_reading_ease", 50)
        if flesch >= 60:
            parts.append("with accessible language")
        else:
            parts.append("with more complex language")
        
        richness = word_freq.get("vocabulary_richness", 0)
        if richness > 0.6:
            parts.append("and diverse vocabulary")
        elif richness > 0.4:
            parts.append("and moderate vocabulary diversity")
        
        return " ".join(parts) + "."
