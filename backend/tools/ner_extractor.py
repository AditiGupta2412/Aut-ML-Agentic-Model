"""
NER Extractor Tool
Uses SpaCy for Named Entity Recognition.
"""
from typing import Dict, Any, Optional, Callable, Awaitable, List
import asyncio


class NERExtractor:
    """
    Named Entity Recognition using SpaCy.
    Extracts and categorizes named entities with full transparency.
    """
    
    MODEL_NAME = "en_core_web_sm"
    
    def __init__(self):
        self._nlp = None
        self._initialized = False
    
    async def _initialize(self, progress_callback: Optional[Callable] = None):
        """Lazy initialization of SpaCy model."""
        if self._initialized:
            return
        
        if progress_callback:
            await progress_callback(5, "Initializing SpaCy NER", {
                "model": self.MODEL_NAME,
                "status": "loading"
            })
        
        try:
            import spacy
            
            # Try to load the model
            loop = asyncio.get_event_loop()
            try:
                self._nlp = await loop.run_in_executor(
                    None,
                    lambda: spacy.load(self.MODEL_NAME)
                )
            except OSError:
                # Model not installed, try downloading
                if progress_callback:
                    await progress_callback(10, "Downloading SpaCy model", {
                        "model": self.MODEL_NAME
                    })
                
                import subprocess
                await loop.run_in_executor(
                    None,
                    lambda: subprocess.run(
                        ["python", "-m", "spacy", "download", self.MODEL_NAME],
                        check=True,
                        capture_output=True
                    )
                )
                
                self._nlp = await loop.run_in_executor(
                    None,
                    lambda: spacy.load(self.MODEL_NAME)
                )
            
            self._initialized = True
            
            if progress_callback:
                await progress_callback(20, "SpaCy model loaded", {
                    "model": self.MODEL_NAME,
                    "status": "ready"
                })
                
        except ImportError:
            raise RuntimeError(
                "spacy library not installed. "
                "Install with: pip install spacy"
            )
    
    async def analyze(
        self,
        text: str,
        parameters: Optional[Dict] = None,
        progress_callback: Optional[Callable[[int, str, Optional[Dict]], Awaitable[None]]] = None
    ) -> Dict[str, Any]:
        """
        Extract named entities from text.
        
        Args:
            text: Text to analyze
            parameters: Optional parameters
            progress_callback: Callback for progress updates
        
        Returns:
            NER results with entity details and explainability
        """
        params = parameters or {}
        
        # Initialize model
        await self._initialize(progress_callback)
        
        if progress_callback:
            await progress_callback(30, "Processing text with SpaCy", {
                "text_length": len(text)
            })
        
        # Process text
        loop = asyncio.get_event_loop()
        doc = await loop.run_in_executor(
            None,
            lambda: self._nlp(text)
        )
        
        if progress_callback:
            await progress_callback(60, "Extracting entities", {
                "token_count": len(doc)
            })
        
        # Extract entities
        entities = []
        entity_counts: Dict[str, int] = {}
        
        for ent in doc.ents:
            entity = {
                "text": ent.text,
                "type": ent.label_,
                "type_description": self._get_entity_description(ent.label_),
                "start_char": ent.start_char,
                "end_char": ent.end_char,
                "context": self._get_entity_context(text, ent.start_char, ent.end_char)
            }
            entities.append(entity)
            
            # Count by type
            if ent.label_ not in entity_counts:
                entity_counts[ent.label_] = 0
            entity_counts[ent.label_] += 1
        
        if progress_callback:
            await progress_callback(80, "Compiling results", {
                "entity_count": len(entities),
                "entity_types": list(entity_counts.keys())
            })
        
        # Group entities by type
        entities_by_type: Dict[str, List] = {}
        for ent in entities:
            etype = ent["type"]
            if etype not in entities_by_type:
                entities_by_type[etype] = []
            entities_by_type[etype].append(ent)
        
        output = {
            "entities": entities,
            "entity_count": len(entities),
            "entity_counts_by_type": entity_counts,
            "entities_by_type": entities_by_type,
            "unique_entity_types": list(entity_counts.keys()),
            "model": self.MODEL_NAME,
            "text_length": len(text),
            "token_count": len(doc),
            "explainability": {
                "method": "spacy_ner",
                "model_type": "SpaCy statistical NER model",
                "entity_types_explained": {
                    etype: self._get_entity_description(etype)
                    for etype in entity_counts.keys()
                },
                "reasoning": f"Identified {len(entities)} named entities across "
                            f"{len(entity_counts)} categories using SpaCy's "
                            f"statistical named entity recognizer."
            }
        }
        
        if progress_callback:
            await progress_callback(100, "NER complete", {
                "total_entities": len(entities)
            })
        
        return output
    
    def _get_entity_description(self, label: str) -> str:
        """Get human-readable description for entity type."""
        descriptions = {
            "PERSON": "People, including fictional",
            "NORP": "Nationalities, religious or political groups",
            "FAC": "Buildings, airports, highways, bridges, etc.",
            "ORG": "Companies, agencies, institutions, etc.",
            "GPE": "Countries, cities, states",
            "LOC": "Non-GPE locations, mountain ranges, bodies of water",
            "PRODUCT": "Objects, vehicles, foods, etc.",
            "EVENT": "Named hurricanes, battles, wars, sports events",
            "WORK_OF_ART": "Titles of books, songs, etc.",
            "LAW": "Named documents made into laws",
            "LANGUAGE": "Any named language",
            "DATE": "Absolute or relative dates or periods",
            "TIME": "Times smaller than a day",
            "PERCENT": "Percentage, including %",
            "MONEY": "Monetary values, including unit",
            "QUANTITY": "Measurements, as of weight or distance",
            "ORDINAL": "First, second, etc.",
            "CARDINAL": "Numerals that do not fall under another type"
        }
        return descriptions.get(label, f"Entity type: {label}")
    
    def _get_entity_context(self, text: str, start: int, end: int, window: int = 30) -> str:
        """Get surrounding context for an entity."""
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)
        
        context = text[context_start:context_end]
        
        # Add ellipsis if truncated
        if context_start > 0:
            context = "..." + context
        if context_end < len(text):
            context = context + "..."
        
        return context
