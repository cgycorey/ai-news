"""Simplified entity extraction for AI news - business focused."""

import re
from typing import List, Dict, Optional, Set
import json
from collections import Counter
import logging

# Simple pattern-based extraction - no heavy ML dependencies
from .text_processor import TextProcessor
from .entity_manager import EntityManager, Entity, get_entity_manager
from .spacy_utils import load_spacy_model, is_model_available
from .entity_types import ExtractedEntity, EntityType

logger = logging.getLogger(__name__)


class EntityExtractor:
    """Simplified entity extraction focused on business intelligence.
    
    Focus on practical business entities:
    - Companies (Google, Microsoft, OpenAI)
    - Products (ChatGPT, Gemini, Claude)
    - Technologies (machine learning, AI ethics, transformers)
    - People (CEOs, researchers, key figures)
    """
    
    def __init__(self, text_processor: TextProcessor, use_spacy: bool = True):
        self.text_processor = text_processor
        self.use_spacy = use_spacy and is_model_available()
        
        # Load spaCy model if available
        self.nlp = None
        if self.use_spacy:
            try:
                self.nlp = load_spacy_model()
                logger.info("✅ spaCy model loaded for entity extraction")
            except Exception as e:
                logger.warning(f"Failed to load spaCy model: {e}")
                self.use_spacy = False
        
        # Business entity patterns
        self.entity_patterns = self._load_entity_patterns()
        
        # Entity manager for learning
        self.entity_manager = get_entity_manager()
        
        # Common business entity lists
        self.tech_companies = self._load_tech_companies()
        self.ai_products = self._load_ai_products()
        self.technologies = self._load_technologies()
    
    def _load_entity_patterns(self) -> Dict[str, List[str]]:
        """Load regex patterns for entity extraction."""
        return {
            'company': [
                r'\b[A-Z][a-z]+ [A-Z][a-z]+(?: [A-Z][a-z]+)?\b',  # Google LLC, Microsoft Corp
                r'\b[A-Z]{2,}\b',  # IBM, HP, Dell (all caps)
                r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Inc|LLC|Corp|Ltd|Corporation)\b'
            ],
            'product': [
                r'\b(?:ChatGPT|Claude|Gemini|GPT-\d+|Bard|Llama|Mistral)\b',
                r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:AI|Bot|Assistant|Platform|Service)\b'
            ],
            'technology': [
                r'\b(?:machine learning|artificial intelligence|deep learning|neural networks|NLP|computer vision)\b',
                r'\b(?:transformer|LLM|generative AI|AI ethics|prompt engineering)\b',
                r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Model|Network|Algorithm|Framework)\b'
            ]
        }
    
    def _load_tech_companies(self) -> Set[str]:
        """Load known tech companies."""
        return {
            'Google', 'Microsoft', 'Apple', 'Amazon', 'Meta', 'OpenAI', 'Anthropic',
            'NVIDIA', 'AMD', 'Intel', 'Tesla', 'IBM', 'Oracle', 'Salesforce',
            'Adobe', 'Cisco', 'VMware', 'Snowflake', 'Palantir', 'Stripe',
            'HubSpot', 'Shopify', 'Square', 'Twitter', 'LinkedIn', 'Reddit'
        }
    
    def _load_ai_products(self) -> Set[str]:
        """Load known AI products and models."""
        return {
            'ChatGPT', 'GPT-4', 'GPT-3', 'Claude', 'Claude-2', 'Gemini', 'Gemini Pro',
            'Bard', 'LaMDA', 'Llama', 'Llama-2', 'Mistral', 'Mixtral',
            'DALL-E', 'Midjourney', 'Stable Diffusion', 'Copilot', 'GitHub Copilot',
            'Siri', 'Alexa', 'Cortana', 'Google Assistant', 'Bing Chat'
        }
    
    def _load_technologies(self) -> Set[str]:
        """Load known AI/ML technologies."""
        return {
            'machine learning', 'deep learning', 'neural networks', 'transformer',
            'LLM', 'large language model', 'generative AI', 'computer vision',
            'natural language processing', 'NLP', 'AI ethics', 'prompt engineering',
            'reinforcement learning', 'GAN', 'diffusion model', 'attention mechanism'
        }
    
    def extract_entities(self, text: str, confidence_threshold: float = 0.6) -> List[ExtractedEntity]:
        """Extract entities from text using pattern matching and optional spaCy."""
        if not text:
            return []
        
        entities = []
        
        # Preprocess text
        processed_text = self.text_processor.process_text(text)
        
        # Pattern-based extraction
        pattern_entities = self._extract_with_patterns(text, processed_text.tokens)
        entities.extend(pattern_entities)
        
        # spaCy-based extraction if available
        if self.use_spacy and self.nlp:
            spacy_entities = self._extract_with_spacy(text)
            entities.extend(spacy_entities)
        
        # Known entity matching
        known_entities = self._extract_known_entities(text)
        entities.extend(known_entities)
        
        # Remove duplicates and filter by confidence
        unique_entities = self._deduplicate_entities(entities)
        filtered_entities = [e for e in unique_entities if e.confidence >= confidence_threshold]
        
        # Sort by confidence and position
        filtered_entities.sort(key=lambda e: (-e.confidence, e.start_position))
        
        return filtered_entities
    
    def _extract_with_patterns(self, text: str, tokens: List[str]) -> List[ExtractedEntity]:
        """Extract entities using regex patterns."""
        entities = []
        
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    entity_text = match.group().strip()
                    
                    # Skip very short or very long matches
                    if len(entity_text) < 2 or len(entity_text) > 50:
                        continue
                    
                    # Calculate confidence based on match quality
                    confidence = self._calculate_pattern_confidence(entity_text, entity_type)
                    
                    if confidence >= 0.5:  # Minimum threshold for patterns
                        entity = ExtractedEntity(
                            text=entity_text,
                            entity_type=EntityType(entity_type),
                            start_position=match.start(),
                            end_position=match.end(),
                            confidence=confidence,
                            extraction_method="pattern_matching"
                        )
                        entities.append(entity)
        
        return entities
    
    def _extract_with_spacy(self, text: str) -> List[ExtractedEntity]:
        """Extract entities using spaCy NER."""
        entities = []
        
        try:
            doc = self.nlp(text)
            
            for ent in doc.ents:
                # Map spaCy labels to our entity types
                entity_type = self._map_spacy_label(ent.label_)
                if not entity_type:
                    continue
                
                # Calculate confidence based on entity properties
                confidence = self._calculate_spacy_confidence(ent)
                
                entity = ExtractedEntity(
                    text=ent.text.strip(),
                    entity_type=entity_type,
                    start_position=ent.start_char,
                    end_position=ent.end_char,
                    confidence=confidence,
                    extraction_method="spacy_ner",
                    metadata={"spacy_label": ent.label_}
                )
                entities.append(entity)
                
        except Exception as e:
            logger.warning(f"spaCy extraction failed: {e}")
        
        return entities
    
    def _extract_known_entities(self, text: str) -> List[ExtractedEntity]:
        """Extract entities from known entity lists."""
        entities = []
        text_lower = text.lower()
        
        # Check tech companies
        for company in self.tech_companies:
            if company.lower() in text_lower:
                start_pos = text_lower.find(company.lower())
                confidence = 0.9  # High confidence for known entities
                
                entity = ExtractedEntity(
                    text=company,
                    entity_type=EntityType.COMPANY,
                    start_position=start_pos,
                    end_position=start_pos + len(company),
                    confidence=confidence,
                    extraction_method="known_entity_matching"
                )
                entities.append(entity)
        
        # Check AI products
        for product in self.ai_products:
            if product.lower() in text_lower:
                start_pos = text_lower.find(product.lower())
                confidence = 0.85
                
                entity = ExtractedEntity(
                    text=product,
                    entity_type=EntityType.PRODUCT,
                    start_position=start_pos,
                    end_position=start_pos + len(product),
                    confidence=confidence,
                    extraction_method="known_entity_matching"
                )
                entities.append(entity)
        
        # Check technologies
        for tech in self.technologies:
            if tech.lower() in text_lower:
                start_pos = text_lower.find(tech.lower())
                confidence = 0.8
                
                entity = ExtractedEntity(
                    text=tech,
                    entity_type=EntityType.TECHNOLOGY,
                    start_position=start_pos,
                    end_position=start_pos + len(tech),
                    confidence=confidence,
                    extraction_method="known_entity_matching"
                )
                entities.append(entity)
        
        return entities
    
    def _map_spacy_label(self, spacy_label: str) -> Optional[EntityType]:
        """Map spaCy NER labels to our entity types."""
        mapping = {
            'ORG': EntityType.COMPANY,
            'PERSON': EntityType.PERSON,
            'PRODUCT': EntityType.PRODUCT,
            'EVENT': EntityType.EVENT,
            'WORK_OF_ART': EntityType.PRODUCT,  # Rough mapping
            'GPE': EntityType.LOCATION
        }
        return mapping.get(spacy_label)
    
    def _calculate_pattern_confidence(self, entity_text: str, entity_type: str) -> float:
        """Calculate confidence score for pattern-based entities."""
        base_confidence = 0.6
        
        # Boost confidence for known entities
        if entity_type == 'company' and entity_text in self.tech_companies:
            return 0.95
        elif entity_type == 'product' and entity_text in self.ai_products:
            return 0.95
        elif entity_type == 'technology' and entity_text.lower() in self.technologies:
            return 0.95
        
        # Adjust based on entity characteristics
        if entity_type == 'company':
            # Companies with suffixes like Inc, LLC are more reliable
            if any(suffix in entity_text for suffix in ['Inc', 'LLC', 'Corp', 'Ltd']):
                base_confidence += 0.2
        
        # Capitalization patterns
        if entity_text.istitle():
            base_confidence += 0.1
        
        # Length penalties
        if len(entity_text) < 3:
            base_confidence -= 0.2
        elif len(entity_text) > 30:
            base_confidence -= 0.1
        
        return min(max(base_confidence, 0.1), 1.0)
    
    def _calculate_spacy_confidence(self, ent) -> float:
        """Calculate confidence score for spaCy entities."""
        base_confidence = 0.7  # spaCy is generally reliable
        
        # Adjust based on entity characteristics
        if ent.label_ == 'ORG':  # Organizations are usually accurate
            base_confidence += 0.1
        elif ent.label_ == 'PERSON':  # Names are usually accurate
            base_confidence += 0.1
        
        # Length considerations
        if len(ent.text) < 2:
            base_confidence -= 0.3
        elif len(ent.text) > 50:
            base_confidence -= 0.2
        
        return min(max(base_confidence, 0.1), 1.0)
    
    def _deduplicate_entities(self, entities: List[ExtractedEntity]) -> List[ExtractedEntity]:
        """Remove duplicate entities, keeping highest confidence ones."""
        seen = set()
        unique_entities = []
        
        # Sort by confidence (highest first)
        entities.sort(key=lambda e: e.confidence, reverse=True)
        
        for entity in entities:
            # Create a key for deduplication (text + type, case-insensitive)
            key = (entity.text.lower(), entity.entity_type.value)
            
            if key not in seen:
                # Check for overlaps with existing entities
                is_duplicate = False
                for existing in unique_entities:
                    if (entity.entity_type == existing.entity_type and
                        abs(entity.start_position - existing.start_position) < len(entity.text)):
                        # Overlapping entities of same type - keep the higher confidence one
                        if entity.confidence > existing.confidence:
                            unique_entities.remove(existing)
                        else:
                            is_duplicate = True
                        break
                
                if not is_duplicate:
                    seen.add(key)
                    unique_entities.append(entity)
        
        return unique_entities
    
    def get_extraction_stats(self) -> Dict[str, any]:
        """Get statistics about entity extraction performance."""
        return {
            'spacy_available': self.use_spacy,
            'spacy_model_loaded': self.nlp is not None,
            'known_tech_companies': len(self.tech_companies),
            'known_ai_products': len(self.ai_products),
            'known_technologies': len(self.technologies),
            'pattern_count': sum(len(patterns) for patterns in self.entity_patterns.values())
        }


def create_entity_extractor(text_processor: TextProcessor, use_spacy: bool = True) -> EntityExtractor:
    """Create an entity extractor instance."""
    return EntityExtractor(text_processor, use_spacy)