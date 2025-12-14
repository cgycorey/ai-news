"""Entity patterns and rule-based extraction for AI news domain."""

import re
from typing import List, Dict, Optional, Pattern, Any, Set
from dataclasses import dataclass
from enum import Enum
import logging

from .text_processor import TextProcessor
from .entity_types import ExtractedEntity, EntityType
from .entity_manager import EntityPattern as Pattern

logger = logging.getLogger(__name__)


class EntityPatterns:
    """Manages entity patterns and rule-based extraction."""
    
    def __init__(self, entity_manager=None):
        """Initialize entity patterns."""
        self.entity_manager = entity_manager
        self.text_processor = TextProcessor()
        
        # AI-related keywords for context validation
        self.ai_keywords = {
            'ai', 'artificial', 'intelligence', 'machine', 'learning', 'deep',
            'neural', 'network', 'algorithm', 'model', 'automation', 'robot',
            'chatbot', 'assistant', 'llm', 'transformer', 'generative', 'startup',
            'funding', 'investment', 'acquisition', 'launch', 'release', 'update'
        }
        
        # Common false positives to filter
        self.false_positives = {
            'AI', 'ML', 'LLM', 'API', 'CEO', 'CTO', 'CFO', 'CIO', 'Mr', 'Mrs', 'Dr', 'Prof',
            'Inc', 'Corp', 'LLC', 'Ltd', 'Co', 'The', 'This', 'That', 'These', 'Those'
        }
        
        # Entity validation patterns by type
        self.validation_patterns = self._initialize_validation_patterns()
        
        # Relationship extraction patterns
        self.relationship_patterns = self._initialize_relationship_patterns()
    
    def _initialize_validation_patterns(self) -> Dict[str, List[Pattern]]:
        """Initialize validation patterns for different entity types."""
        return {
            'company': [
                Pattern(
                    name="company_suffix",
                    pattern=r'\b[A-Z][a-zA-Z0-9\s&\-\']+(?:\s+(?:Inc|Corp|LLC|Ltd|Company|Technologies|Solutions|Labs|Studios|Group|Holdings))\b',
                    entity_type="company",
                    confidence=0.8
                ),
                Pattern(
                    name="tech_company",
                    pattern=r'\b[A-Z][a-zA-Z0-9\s&\-\']+(?:\s+(?:AI|Tech|Systems|Software|Digital|Cyber|Quantum|Robotics))\b',
                    entity_type="company",
                    confidence=0.7
                ),
                Pattern(
                    name="startup_capital",
                    pattern=r'\b[A-Z][a-zA-Z0-9\s&\-\'][A-Z][a-z][a-z][a-z\-\s]*\b',
                    entity_type="company",
                    confidence=0.5
                )
            ],
            'product': [
                Pattern(
                    name="product_with_version",
                    pattern=r'\b[A-Z][a-zA-Z0-9\s+\-\.]+(?:\s+v\d+(?:\.\d+)*)\b',
                    entity_type="product",
                    confidence=0.8
                ),
                Pattern(
                    name="ai_product",
                    pattern=r'\b[A-Z][a-zA-Z0-9\s+\-\']*(?:AI|Chat|Bot|Assistant|Model|Engine|Platform|Service|System)\b',
                    entity_type="product",
                    confidence=0.7
                ),
                Pattern(
                    name="version_number",
                    pattern=r'\bv?\d+(?:\.\d+)*(?:\s*(?:alpha|beta|rc|preview|release))?\b',
                    entity_type="product",
                    confidence=0.6
                )
            ],
            'person': [
                Pattern(
                    name="full_name",
                    pattern=r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b',
                    entity_type="person",
                    confidence=0.7
                ),
                Pattern(
                    name="person_with_title",
                    pattern=r'\b(?:Mr|Mrs|Ms|Dr|Prof|Sir|Madam)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b',
                    entity_type="person",
                    confidence=0.8
                )
            ],
            'technology': [
                Pattern(
                    name="tech_terms",
                    pattern=r'\b(?:machine learning|deep learning|neural network|natural language processing|computer vision|reinforcement learning|transfer learning|generative AI|large language model|transformer|attention mechanism|convolutional neural network|recurrent neural network|gradient descent|backpropagation|overfitting|regularization)\b',
                    entity_type="technology",
                    confidence=0.9
                ),
                Pattern(
                    name="ai_frameworks",
                    pattern=r'\b(?:TensorFlow|PyTorch|Keras|Scikit-learn|Theano|MXNet|Caffe|PaddlePaddle|JAX|FastAI|Hugging Face|spaCy|NLTK|OpenCV)\b',
                    entity_type="technology",
                    confidence=0.9
                ),
                Pattern(
                    name="algorithms",
                    pattern=r'\b(?:GAN|VAE|BERT|GPT|Transformer|CNN|RNN|LSTM|GRU|SVM|Random Forest|Decision Tree|k-NN|Naive Bayes|K-means|DBSCAN|PCA|t-SNE|Word2Vec|GloVe|FastText)\b',
                    entity_type="technology",
                    confidence=0.8
                )
            ],
            'research': [
                Pattern(
                    name="research_papers",
                    pattern=r'\b["\'][A-Z][a-zA-Z0-9\s\-:,.]+(?:Paper|Research|Study|Analysis)["\']\b',
                    entity_type="research",
                    confidence=0.8
                ),
                Pattern(
                    name="conferences",
                    pattern=r'\b(?:NeurIPS|ICML|ICLR|AAAI|IJCAI|ACL|EMNLP|CVPR|ICCV|ECCV|ICASSP|INTERSPEECH|KDD|WSDM|WWW|SIGIR|RecSys)\b',
                    entity_type="research",
                    confidence=0.9
                ),
                Pattern(
                    name="universities",
                    pattern=r'\b(?:MIT|Stanford|Carnegie Mellon|UC Berkeley|Caltech|Harvard|Oxford|Cambridge|ETH Zurich|TU Munich|Tsinghua|Peking University)\b',
                    entity_type="research",
                    confidence=0.8
                )
            ],
            'event': [
                Pattern(
                    name="ai_events",
                    pattern=r'\b(?:AI Summit|Tech Conference|Product Launch|Developer Conference|Keynote|Workshop|Hackathon|Demo Day)\b',
                    entity_type="event",
                    confidence=0.7
                ),
                Pattern(
                    name="named_events",
                    pattern=r'\b[A-Z][a-zA-Z0-9\s]+(?:\d{4}|Conference|Summit|Expo|Fair)\b',
                    entity_type="event",
                    confidence=0.6
                )
            ]
        }
    
    def _initialize_relationship_patterns(self) -> List[Dict[str, Any]]:
        """Initialize patterns for extracting entity relationships."""
        return [
            {
                'pattern': r'([A-Z][a-z]+\s+[A-Z][a-z]+)\s+(?:is|was|as)\s+(?:the\s+)?(?:CEO|chief\s+executive\s+officer)\s+of\s+([A-Z][a-zA-Z\s&]+)',
                'relationship_type': 'CEO_OF',
                'source_index': 1,
                'target_index': 2,
                'confidence': 0.9
            },
            {
                'pattern': r'([A-Z][a-zA-Z\s&]+)\s+(?:acquired|bought|purchased)\s+([A-Z][a-zA-Z\s&]+)',
                'relationship_type': 'ACQUIRED',
                'source_index': 1,
                'target_index': 2,
                'confidence': 0.8
            },
            {
                'pattern': r'([A-Z][a-zA-Z\s&]+)\s+(?:announced|released|launched)\s+([A-Z][a-zA-Z0-9\s]+)',
                'relationship_type': 'RELEASED',
                'source_index': 1,
                'target_index': 2,
                'confidence': 0.7
            },
            {
                'pattern': r'([A-Z][a-zA-Z\s&]+)\s+(?:partnered\s+with|collaborated\s+with)\s+([A-Z][a-zA-Z\s&]+)',
                'relationship_type': 'PARTNERED_WITH',
                'source_index': 1,
                'target_index': 2,
                'confidence': 0.8
            },
            {
                'pattern': r'([A-Z][a-zA-Z\s&]+)\s+(?:invested\s+in|funded)\s+([A-Z][a-zA-Z\s&]+)',
                'relationship_type': 'INVESTED_IN',
                'source_index': 1,
                'target_index': 2,
                'confidence': 0.8
            },
            {
                'pattern': r'([A-Z][a-z]+\s+[A-Z][a-z]+)\s+(?:joined|works\s+for|is\s+at)\s+([A-Z][a-zA-Z\s&]+)',
                'relationship_type': 'WORKS_FOR',
                'source_index': 1,
                'target_index': 2,
                'confidence': 0.7
            },
            {
                'pattern': r'([A-Z][a-zA-Z\s&]+)\s+(?:founded|created|established)\s+([A-Z][a-zA-Z\s&]+)',
                'relationship_type': 'FOUNDED',
                'source_index': 1,
                'target_index': 2,
                'confidence': 0.8
            }
        ]
    
    def extract_pattern_entities(self, text: str, include_context: bool = True, 
                                known_entities_map: Optional[Dict[str, Any]] = None) -> List[ExtractedEntity]:
        """Extract entities using regex patterns."""
        entities = []
        
        # Use patterns from entity manager if available
        if self.entity_manager:
            patterns_to_use = list(self.entity_manager.patterns)
        else:
            # Use built-in validation patterns
            patterns_to_use = []
            for entity_type_patterns in self.validation_patterns.values():
                patterns_to_use.extend(entity_type_patterns)
        
        for pattern in patterns_to_use:
            if not pattern.compiled_pattern:
                continue
            
            try:
                for match in pattern.compiled_pattern.finditer(text):
                    entity_text = match.group().strip()
                    
                    # Skip if matches exclusion patterns
                    if self._is_excluded(entity_text):
                        continue
                    
                    # Additional validation
                    if self.validate_entity_text(entity_text, pattern.entity_type):
                        context = self._get_context(text, match.start(), match.end()) if include_context else ""
                        
                        # Check if matches known entity
                        known_entity = None
                        if known_entities_map:
                            known_entity = known_entities_map.get(entity_text.lower())
                        
                        if known_entity:
                            # Use known entity info
                            entities.append(ExtractedEntity(
                                text=entity_text,
                                entity_type=EntityType(pattern.entity_type),
                                confidence=pattern.confidence,
                                start_char=match.start(),
                                end_char=match.end(),
                                context=context,
                                canonical_name=known_entity.get('name') or entity_text,
                                aliases=known_entity.get('aliases', []),
                                metadata={**pattern.__dict__, **known_entity.get('metadata', {})},
                                extraction_method="pattern_known"
                            ))
                        else:
                            # New entity from pattern
                            entities.append(ExtractedEntity(
                                text=entity_text,
                                entity_type=EntityType(pattern.entity_type),
                                confidence=pattern.confidence * 0.8,  # Lower for unknown
                                start_char=match.start(),
                                end_char=match.end(),
                                context=context,
                                metadata={'pattern_name': pattern.name, 'pattern_confidence': pattern.confidence},
                                extraction_method="pattern_new"
                            ))
            
            except Exception as e:
                logger.error(f"Pattern matching error for {pattern.name}: {e}")
        
        return entities
    
    def extract_relationships(self, text: str, target_entity: Optional[str] = None) -> List[Dict[str, Any]]:
        """Extract entity relationships using patterns."""
        relationships = []
        
        for pattern_info in self.relationship_patterns:
            pattern = pattern_info['pattern']
            relationship_type = pattern_info['relationship_type']
            source_idx = pattern_info['source_index']
            target_idx = pattern_info['target_index']
            confidence = pattern_info['confidence']
            
            try:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    groups = match.groups()
                    if len(groups) >= 2:
                        source_entity = groups[source_idx - 1].strip()
                        target_entity_in_match = groups[target_idx - 1].strip()
                        
                        # Only add relationships if target_entity is specified and matches
                        if target_entity is None or (
                            source_entity.lower() == target_entity.lower() or 
                            target_entity_in_match.lower() == target_entity.lower()
                        ):
                            context = self._get_context(text, match.start(), match.end())
                            
                            relationship = {
                                'source_entity': source_entity,
                                'target_entity': target_entity_in_match,
                                'relationship_type': relationship_type,
                                'confidence': confidence,
                                'context': context,
                                'start_char': match.start(),
                                'end_char': match.end(),
                                'evidence': match.group(0)
                            }
                            
                            relationships.append(relationship)
            except Exception as e:
                logger.error(f"Error in relationship pattern {pattern_info['relationship_type']}: {e}")
        
        return relationships
    
    def is_ai_related_entity(self, text: str, entity_type: EntityType) -> bool:
        """Check if entity is AI-related based on context."""
        text_lower = text.lower()
        
        # Check if entity text contains AI keywords
        has_ai_keyword = any(kw in text_lower for kw in self.ai_keywords)
        
        # For companies and products, be more lenient
        if entity_type in [EntityType.COMPANY, EntityType.PRODUCT]:
            return True
        
        # For other types, require AI keywords
        return has_ai_keyword
    
    def validate_entity_text(self, text: str, entity_type: str) -> bool:
        """Validate entity text."""
        # Length validation
        if len(text.strip()) < 2 or len(text) > 100:
            return False
        
        # Word count validation
        if len(text.split()) > 6:
            return False
        
        # Type-specific validation
        if entity_type == "company" and not any(c.isupper() for c in text):
            return False
        
        # Must contain at least one alphanumeric character
        if not any(c.isalnum() for c in text):
            return False
        
        return True
    
    def _is_excluded(self, text: str) -> bool:
        """Check if text matches exclusion patterns."""
        if text.strip() in self.false_positives:
            return True
        
        # Common exclusion patterns
        exclusion_patterns = [
            r'^\d+$',  # Pure numbers
            r'^[A-Z]{2,}$',  # All caps abbreviations
            r'^[^a-zA-Z0-9]+$',  # No alphanumeric characters
            r'^(?:the|this|that|these|those|and|or|but|for|with|from|to|in|on|at|by)$',  # Common words
        ]
        
        for pattern in exclusion_patterns:
            if re.match(pattern, text.strip(), re.IGNORECASE):
                return True
        
        # Use entity manager exclusion if available
        if self.entity_manager and hasattr(self.entity_manager, '_is_excluded'):
            return self.entity_manager._is_excluded(text)
        
        return False
    
    def _get_context(self, text: str, start: int, end: int, window: int = 100) -> str:
        """Get context around entity."""
        start_context = max(0, start - window)
        end_context = min(len(text), end + window)
        
        context = text[start_context:end_context].strip()
        
        # Clean up context
        context = re.sub(r'\s+', ' ', context)
        
        return context
    
    def get_relationship_from_verb(self, verb_lemma: str) -> Optional[str]:
        """Infer relationship type from verb lemma."""
        verb_relationships = {
            'acquire': 'ACQUIRED',
            'buy': 'ACQUIRED',
            'purchase': 'ACQUIRED',
            'launch': 'RELEASED',
            'release': 'RELEASED',
            'announce': 'RELEASED',
            'partner': 'PARTNERED_WITH',
            'collaborate': 'PARTNERED_WITH',
            'invest': 'INVESTED_IN',
            'fund': 'INVESTED_IN',
            'lead': 'LEADS',
            'run': 'LEADS',
            'manage': 'LEADS',
            'work': 'WORKS_FOR',
            'employ': 'EMPLOYS',
            'found': 'FOUNDED',
            'create': 'CREATED',
            'establish': 'FOUNDED',
            'build': 'CREATED',
            'develop': 'CREATED'
        }
        
        return verb_relationships.get(verb_lemma)
    
    def get_relationship_from_preposition(self, prep_text: str) -> Optional[str]:
        """Infer relationship type from preposition."""
        prep_relationships = {
            'with': 'PARTNERED_WITH',
            'by': 'CREATED_BY',
            'from': 'FROM',
            'in': 'LOCATED_IN',
            'at': 'LOCATED_AT',
            'for': 'WORKS_FOR',
            'of': 'PART_OF'
        }
        
        return prep_relationships.get(prep_text.lower())
    
    def add_custom_pattern(self, pattern: Pattern):
        """Add a custom validation pattern."""
        entity_type = pattern.entity_type
        if entity_type not in self.validation_patterns:
            self.validation_patterns[entity_type] = []
        
        self.validation_patterns[entity_type].append(pattern)
        logger.info(f"Added custom pattern '{pattern.name}' for entity type '{entity_type}'")
    
    def add_relationship_pattern(self, pattern: Dict[str, Any]):
        """Add a custom relationship pattern."""
        required_keys = ['pattern', 'relationship_type', 'source_index', 'target_index', 'confidence']
        if all(key in pattern for key in required_keys):
            self.relationship_patterns.append(pattern)
            logger.info(f"Added custom relationship pattern '{pattern['relationship_type']}'")
        else:
            logger.error(f"Invalid relationship pattern. Missing keys: {required_keys - set(pattern.keys())}")
    
    def get_pattern_statistics(self) -> Dict[str, Any]:
        """Get statistics about loaded patterns."""
        stats = {
            'validation_patterns': {
                entity_type: len(patterns) 
                for entity_type, patterns in self.validation_patterns.items()
            },
            'relationship_patterns': len(self.relationship_patterns),
            'total_validation_patterns': sum(
                len(patterns) for patterns in self.validation_patterns.values()
            )
        }
        
        return stats