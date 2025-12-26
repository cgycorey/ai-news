"""Simplified entity extraction for AI news - business focused.

REUSABILITY IN OTHER FEATURES:
=============================
This module is designed to be easily reused across the codebase:

1. SIMPLE USAGE:
   from ai_news.entity_extractor import EntityExtractor
   from ai_news.text_processor import TextProcessor
   
   text_processor = TextProcessor()
   extractor = EntityExtractor(text_processor, use_spacy=True)
   entities = extractor.extract_entities("Your text here")
   
2. CUSTOM CONFIDENCE THRESHOLD:
   entities = extractor.extract_entities(text, confidence_threshold=0.7)
   
3. ACCESS BY ENTITY TYPE:
   companies = [e for e in entities if e.entity_type == EntityType.COMPANY]
   products = [e for e in entities if e.entity_type == EntityType.PRODUCT]
   
4. GET TOP ENTITIES:
   top_entities = extractor.get_top_entities(entities, limit=10)
   
5. BATCH PROCESSING:
   for article in articles:
       entities = extractor.extract_entities(article['content'])
       # Use entities for tagging, recommendations, etc.

6. FEATURES THAT CAN USE THIS:
   - Article tagging and classification
   - Content recommendations (based on entity overlap)
   - Trending entity detection
   - Feed personalization
   - Search enhancement
   - Duplicate article detection
   - Topic clustering

The extractor is stateless and thread-safe after initialization.
"""

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

        # Load entities from EntityManager (dynamic, not hardcoded)
        self.tech_companies = self._load_entities_from_manager('company')
        self.ai_products = self._load_entities_from_manager('product')
        self.technologies = self._load_entities_from_manager('technology')
    
    def _load_entity_patterns(self) -> Dict[str, List[str]]:
        """Load regex patterns for entity extraction.

        NOTE: Pattern matching is LESS precise than spaCy NER.
        These patterns are conservative and should only match high-confidence patterns.
        """
        return {
            'company': [
                # Only match companies with legal suffixes (very specific)
                r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Inc|LLC|Corp|Ltd|Corporation|Company)\b',
                # All-caps company abbreviations (but not common words)
                r'\b(?:IBM|HP|Dell|AMD|Intel|NVIDIA|BMW|IBM|SAP|GE|GM|Ford)\b'
            ],
            'product': [
                # Only well-known AI products (specific list, not generic patterns)
                r'\b(?:ChatGPT|Claude|Gemini|GPT-4|GPT-4o|Bard|Llama|Mistral|Copilot|Perplexity)\b'
            ],
            'technology': [
                # Only specific tech terms, not generic patterns
                r'\b(?:machine learning|artificial intelligence|deep learning|neural networks|NLP|computer vision)\b',
                r'\b(?:transformers?|large language model|generative AI|LLM|AI ethics|prompt engineering)\b'
            ]
        }

    def _is_stopword_or_common(self, text: str) -> bool:
        """Check if text is a common word that should not be an entity."""
        common_stops = {
            'the', 'this', 'that', 'these', 'those', 'a', 'an', 'and', 'or', 'but',
            'with', 'for', 'from', 'about', 'in', 'on', 'at', 'to', 'by', 'of', 'as',
            'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does',
            'will', 'would', 'could', 'should', 'may', 'might', 'can', 'new', 'first',
            'last', 'other', 'own', 'such', 'so', 'than', 'too', 'very', 'just', 'also',
            'now', 'here', 'there', 'when', 'where', 'how', 'what', 'which', 'who', 'whom'
        }
        return text.lower() in common_stops

    def _should_filter_entity(self, text: str, entity_type: EntityType, spacy_label: Optional[str] = None) -> bool:
        """Check if an entity should be filtered out as a false positive.

        Returns True if the entity should be filtered out.
        """
        text_lower = text.lower().strip()
        text_clean = ' '.join(text.split())  # Normalize whitespace

        # 1. Generic terms that should never be companies
        generic_company_terms = {
            'ai', 'ml', 'llm', 'api', 'app', 'bot', 'ceo', 'cto', 'cfo', 'coo',
            'startup', 'startups', 'tech', 'data', 'cloud', 'digital', 'smart',
            'pro', 'plus', 'premium', 'basic', 'free', 'lite', 'standard',
            'labs', 'lab', 'team', 'group', 'division', 'department', 'unit',
            'the', 'a', 'an', 'and', 'or', 'but', 'with', 'for', 'from', 'about',
            'system', 'platform', 'service', 'tool', 'software', 'solution',
            'research', 'development', 'security', 'monitoring', 'capabilities',
            'uses', 'announced', 'launched', 'released', 'developed', 'testified',
            'based', 'located', 'founded', 'congress', 'us', 'francisco', 'san'
        }

        # 2. Product suffixes that shouldn't be standalone entities
        product_suffixes_only = {
            'pro', 'plus', 'premium', 'basic', 'free', 'lite', 'standard',
            'enterprise', 'business', 'personal', 'home', 'max', 'ultra',
            'mini', 'express', 'advanced', 'professional', 'ultimate'
        }

        # Filter generic company terms
        if entity_type == EntityType.COMPANY and text_lower in generic_company_terms:
            return True

        # Filter standalone product suffixes
        if entity_type in [EntityType.COMPANY, EntityType.PRODUCT] and text_lower in product_suffixes_only:
            # Only filter if it's JUST the suffix (not part of a larger name)
            if len(text_clean.split()) == 1:
                return True

        # 3. Generic tech abbreviations that aren't companies
        if entity_type == EntityType.COMPANY and text_lower in {'ai', 'ml', 'llm', 'api'}:
            # "AI" alone is too generic - must be part of a company name
            return True

        # 4. spaCy-specific misclassifications
        if spacy_label == 'PERSON':
            # Products often misclassified as PERSON - check for product indicators
            # Only filter if the entity IS exactly or starts with a product indicator
            # e.g., "GPT-4", "ChatGPT Pro", not "Sam Altman" (contains no product indicator)
            product_indicators = ['pro', 'plus', 'bot', 'assistant', 'ai']
            # Split entity into words to check if it STARTS with a product indicator
            words = text_lower.split()
            if words and words[0] in product_indicators:
                # First word is a product indicator, likely not a person
                return True
            # Also check for exact matches to product names
            if text_lower in {'gpt', 'chatgpt', 'claude', 'gemini', 'copilot'}:
                return True

        if spacy_label == 'ORG':
            # Common words misclassified as ORG
            if text_lower in {'ai', 'ml', 'llm', 'api', 'pro', 'plus', 'labs', 'team'}:
                return True

        if spacy_label == 'FAC':
            # Version numbers like "Claude 4" misclassified as FAC
            if any(char.isdigit() for char in text):
                # Contains a digit, likely a version, not a facility
                return True

        if spacy_label == 'GPE':
            # Companies misclassified as GPE
            known_companies_not_locations = {'openai', 'anthropic', 'meta', 'google', 'microsoft', 'amazon'}
            if text_lower in known_companies_not_locations:
                return True

        # 5. Very short single-letter or two-letter entities (except known abbreviations)
        if len(text_clean) <= 2 and text_clean.upper() not in {'IBM', 'HP', 'AI', 'GM', 'GE', 'BMW', 'AMD', 'SAP'}:
            return True

        return False
    
    def _load_entities_from_manager(self, entity_type: str) -> Set[str]:
        """Load entities from EntityManager (combines JSON config + database).

        Hybrid approach:
        - Core entities from JSON (rarely changes, manual edits)
        - Discovered entities from DB (auto-learned, no maintenance)

        NO HARDCODED LISTS - all entities loaded dynamically.
        """
        try:
            entities = self.entity_manager.get_entities_by_type(entity_type)
            return {e.name for e in entities}
        except Exception as e:
            logger.warning(f"Failed to load {entity_type} entities from manager: {e}")
            return set()
    
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
        spacy_full_entities = set()  # Track full spaCy entities to avoid partial matches
        if self.use_spacy and self.nlp:
            spacy_entities = self._extract_with_spacy(text)
            entities.extend(spacy_entities)
            # Track multi-word spaCy entities
            spacy_full_entities = {e.text for e in spacy_entities if len(e.text.split()) > 1}

        # Discover and learn new entities (auto-discovery) - do this BEFORE known entity matching
        discovered_entities = self._discover_and_learn_entities(text)
        entities.extend(discovered_entities)
        # Also track multi-word entities from discovery
        for e in discovered_entities:
            if len(e.text.split()) > 1:
                spacy_full_entities.add(e.text)

        # Known entity matching - pass spaCy entities to avoid partial matches
        known_entities = self._extract_known_entities(text, spacy_full_entities)
        entities.extend(known_entities)

        # Remove duplicates and filter by confidence
        unique_entities = self._deduplicate_entities(entities)
        filtered_entities = [e for e in unique_entities if e.confidence >= confidence_threshold]

        # Sort by confidence and position
        filtered_entities.sort(key=lambda e: (-e.confidence, e.start_position))

        return filtered_entities
    
    def _extract_with_patterns(self, text: str, tokens: List[str]) -> List[ExtractedEntity]:
        """Extract entities using regex patterns.

        NOTE: This is conservative - pattern matching is less precise than spaCy.
        Only high-confidence matches are returned.
        """
        entities = []

        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    entity_text = match.group().strip()

                    # Skip very short or very long matches
                    if len(entity_text) < 2 or len(entity_text) > 50:
                        continue

                    # Skip stopwords and common words
                    if self._is_stopword_or_common(entity_text):
                        continue

                    # Calculate confidence based on match quality
                    confidence = self._calculate_pattern_confidence(entity_text, entity_type)

                    # Higher threshold for pattern matching (less precise than spaCy)
                    if confidence >= 0.7:
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
        """Extract entities using spaCy NER.

        Uses the basic spaCy extraction with enhanced confidence scoring.
        """
        entities = []

        try:
            doc = self.nlp(text)

            # Use the same label configs as enhanced discovery for consistency
            spacy_entity_configs = {
                'ORG': {'type': 'company', 'min_confidence': 0.65},
                'PRODUCT': {'type': 'product', 'min_confidence': 0.60},
                'PERSON': {'type': 'person', 'min_confidence': 0.70},
                'GPE': {'type': 'location', 'min_confidence': 0.75},
                'FAC': {'type': 'location', 'min_confidence': 0.70},
                'EVENT': {'type': 'event', 'min_confidence': 0.65},
            }

            for ent in doc.ents:
                label = ent.label_

                # Only process labels we have configs for
                if label not in spacy_entity_configs:
                    continue

                config = spacy_entity_configs[label]

                # Map spaCy labels to our entity types
                entity_type_map = {
                    'company': EntityType.COMPANY,
                    'product': EntityType.PRODUCT,
                    'person': EntityType.PERSON,
                    'location': EntityType.LOCATION,
                    'event': EntityType.EVENT
                }
                entity_type = entity_type_map.get(config['type'])

                if not entity_type:
                    continue

                # Apply entity filtering (remove false positives)
                if self._should_filter_entity(ent.text.strip(), entity_type, label):
                    logger.debug(f"Filtered spaCy entity: {ent.text} ({label})")
                    continue

                # Skip stopwords
                if self._is_stopword_or_common(ent.text.strip()):
                    continue

                # Calculate confidence using the enhanced scoring
                confidence = self._calculate_enhanced_spacy_confidence(ent, doc, config)

                # Filter by minimum confidence
                if confidence < config['min_confidence']:
                    continue

                entity = ExtractedEntity(
                    text=ent.text.strip(),
                    entity_type=entity_type,
                    start_position=ent.start_char,
                    end_position=ent.end_char,
                    confidence=confidence,
                    extraction_method="spacy_ner",
                    metadata={"spacy_label": label, "confidence_breakdown": self._get_confidence_breakdown(ent, doc)}
                )
                entities.append(entity)

        except Exception as e:
            logger.warning(f"spaCy extraction failed: {e}")

        return entities
    
    def _extract_known_entities(self, text: str, spacy_full_entities: Optional[Set[str]] = None) -> List[ExtractedEntity]:
        """Extract entities from known entity lists using word-boundary matching.

        Args:
            text: The input text
            spacy_full_entities: Set of full spaCy entity texts to avoid partial matches
                              e.g., if spaCy found "Sam Altman", don't extract "Sam" or "Altman" separately
        """
        entities = []
        text_lower = text.lower()
        spacy_full_entities = spacy_full_entities or set()

        # Check tech companies
        for company in self.tech_companies:
            # Use word boundary matching to avoid partial matches
            # e.g., "AI" shouldn't match inside "OpenAI"
            pattern = r'\b' + re.escape(company.lower()) + r'\b'
            if re.search(pattern, text_lower):
                match = re.search(pattern, text_lower)
                start_pos = match.start()
                confidence = 0.9  # High confidence for known entities

                # Apply filtering
                if self._should_filter_entity(company, EntityType.COMPANY):
                    continue

                # Skip if this is a substring of a spaCy multi-word entity
                # e.g., don't extract "Sam" if spaCy found "Sam Altman"
                is_partial_of_spacy_entity = False
                for spacy_entity in spacy_full_entities:
                    if company.lower() in spacy_entity.lower() and len(company) < len(spacy_entity):
                        # This is a partial match, skip it
                        is_partial_of_spacy_entity = True
                        break

                if is_partial_of_spacy_entity:
                    continue

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
            pattern = r'\b' + re.escape(product.lower()) + r'\b'
            if re.search(pattern, text_lower):
                match = re.search(pattern, text_lower)
                start_pos = match.start()
                confidence = 0.85

                # Apply filtering
                if self._should_filter_entity(product, EntityType.PRODUCT):
                    continue

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
            pattern = r'\b' + re.escape(tech.lower()) + r'\b'
            if re.search(pattern, text_lower):
                match = re.search(pattern, text_lower)
                start_pos = match.start()
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
    
    def _discover_and_learn_entities(self, text: str) -> List[ExtractedEntity]:
        """Discover new entities from text and learn them for future use.

        Priority:
        1. spaCy NER (most precise, expanded labels) - HIGH PRIORITY
        2. Keyword-based discovery (fallback only, lower priority)
        """
        discovered = []

        try:
            # Get existing entity names to avoid duplicates
            existing_names = set(self.tech_companies) | set(self.ai_products) | set(self.technologies)
            existing_names_lower = {n.lower() for n in existing_names}

            # Method 1: Enhanced spaCy NER with expanded labels and confidence scoring
            if self.use_spacy and self.nlp:
                try:
                    doc = self.nlp(text)

                    # Expanded spaCy label mapping for business entities
                    # Each label has specific validation rules and confidence scoring
                    spacy_entity_configs = {
                        'ORG': {'type': 'company', 'min_confidence': 0.65},
                        'PRODUCT': {'type': 'product', 'min_confidence': 0.60},
                        'PERSON': {'type': 'person', 'min_confidence': 0.70},
                        'GPE': {'type': 'location', 'min_confidence': 0.75},  # Geographical entities
                        'FAC': {'type': 'location', 'min_confidence': 0.70},  # Facilities, buildings
                        'EVENT': {'type': 'event', 'min_confidence': 0.65},
                        'LAW': {'type': 'product', 'min_confidence': 0.60},  # Regulations, laws
                        'WORK_OF_ART': {'type': 'product', 'min_confidence': 0.55},
                    }

                    for ent in doc.ents:
                        label = ent.label_

                        # Skip labels we don't handle
                        if label not in spacy_entity_configs:
                            continue

                        config = spacy_entity_configs[label]
                        entity_name = ent.text.strip()

                        # Skip if already known
                        if entity_name.lower() in existing_names_lower:
                            continue

                        # Skip too short or too long
                        if len(entity_name) < 2 or len(entity_name) > 50:
                            continue

                        # Map to our entity type
                        entity_type_map = {
                            'company': EntityType.COMPANY,
                            'product': EntityType.PRODUCT,
                            'person': EntityType.PERSON,
                            'location': EntityType.LOCATION,
                            'event': EntityType.EVENT
                        }
                        entity_type = entity_type_map.get(config['type'], EntityType.COMPANY)

                        # Apply entity filtering (remove false positives)
                        if self._should_filter_entity(entity_name, entity_type, label):
                            logger.debug(f"Filtered enhanced discovery entity: {entity_name} ({label})")
                            continue

                        # Calculate confidence using sophisticated scoring
                        confidence = self._calculate_enhanced_spacy_confidence(ent, doc, config)

                        # Filter by minimum confidence threshold for this label type
                        if confidence < config['min_confidence']:
                            logger.debug(f"Filtered low-confidence {label} entity: {entity_name} (conf={confidence:.2f})")
                            continue

                        # Additional context validation for companies
                        if label == 'ORG' and not self._validate_company_context(ent, doc):
                            logger.debug(f"Failed context validation for company: {entity_name}")
                            continue

                        # Create new entity with confidence-based scoring
                        new_entity = Entity(
                            name=entity_name,
                            entity_type=config['type'],
                            confidence=confidence,
                            description=f"Auto-discovered {config['type']} via enhanced spaCy NER ({label})",
                            metadata={
                                'source': 'auto_discovery_spacy_enhanced',
                                'spacy_label': label,
                                'confidence_score': confidence,
                                'context_validated': label == 'ORG'
                            }
                        )

                        # Save to database
                        self.entity_manager.add_entity(new_entity)
                        logger.info(f"✅ Learned new {config['type']} via spaCy {label}: {entity_name} (conf={confidence:.2f})")

                        # Convert to ExtractedEntity
                        discovered.append(ExtractedEntity(
                            text=entity_name,
                            entity_type=entity_type,
                            start_position=ent.start_char,
                            end_position=ent.end_char,
                            confidence=confidence,
                            extraction_method="spacy_enhanced_discovery",
                            metadata={"spacy_label": label, "confidence_breakdown": self._get_confidence_breakdown(ent, doc)}
                        ))

                except Exception as e:
                    logger.warning(f"Enhanced spaCy discovery failed: {e}")

            # Method 2: Keyword-based discovery (FALLBACK ONLY, lower priority)
            # Only run if spaCy found fewer than 3 entities OR is disabled
            use_keyword_fallback = (
                not self.use_spacy or
                not self.nlp or
                len(discovered) < 3
            )

            if use_keyword_fallback:
                logger.debug("Using keyword-based discovery as fallback")
                discovered.extend(self._keyword_based_discovery(text, existing_names, existing_names_lower, discovered))

        except Exception as e:
            logger.warning(f"Entity discovery failed: {e}")

        return discovered
    
    def _keyword_based_discovery(self, text: str, existing_names: Set[str],
                                  existing_names_lower: Set[str],
                                  already_discovered: List[ExtractedEntity]) -> List[ExtractedEntity]:
        """Keyword-based discovery as FALLBACK when spaCy finds insufficient entities.

        This is intentionally conservative and lower priority than spaCy.
        """
        discovered = []

        # Expanded common words to filter out (generic terms that slip through)
        common_words = {
            'The', 'This', 'That', 'These', 'Those', 'A', 'An', 'And', 'Or', 'But',
            'With', 'For', 'From', 'About', 'In', 'On', 'At', 'To', 'By', 'Of',
            'As', 'It', 'Its', 'Their', 'Our', 'We', 'You', 'They', 'He', 'She',
            'Be', 'Is', 'Are', 'Was', 'Were', 'Been', 'Being', 'Have', 'Has', 'Had',
            'Do', 'Does', 'Did', 'Will', 'Would', 'Could', 'Should', 'May', 'Might',
            'Can', 'Need', 'Must', 'Shall', 'AI', 'ML', 'LLM', 'API', 'App', 'Bot',
            'Use', 'Used', 'Using', 'Make', 'Made', 'Get', 'Got', 'Know', 'Known',
            'See', 'Seen', 'Find', 'Found', 'Look', 'Looking', 'Say', 'Said', 'Tell',
            'Told', 'Ask', 'Asked', 'Come', 'Came', 'Go', 'Went', 'Take', 'Took',
            'Year', 'Time', 'Way', 'New', 'First', 'Last', 'Long', 'Great', 'Own',
            'Other', 'Old', 'Right', 'Big', 'High', 'Different', 'Small', 'Large',
            'Next', 'Early', 'Young', 'Important', 'Public', 'Bad', 'Same', 'Able'
        }

        # Generic tech terms that are NOT companies
        generic_tech_terms = {
            'Model', 'Models', 'System', 'Systems', 'Tool', 'Tools', 'Platform',
            'Platforms', 'Service', 'Services', 'Solution', 'Solutions', 'Network',
            'Networks', 'Framework', 'Frameworks', 'Algorithm', 'Algorithms',
            'Method', 'Methods', 'Approach', 'Approaches', 'Process', 'Processes',
            'Technology', 'Technologies', 'Research', 'Development', 'Application',
            'Applications', 'Software', 'Hardware', 'Device', 'Devices', 'User',
            'Users', 'Data', 'Information', 'Content', 'Feature', 'Features',
            'Result', 'Results', 'Study', 'Studies', 'Paper', 'Papers', 'Work'
        }

        # Combine all filters
        skip_words = common_words | generic_tech_terms

        # Company suffixes that increase confidence
        company_suffixes = {'AI', 'Labs', 'Tech', 'Technologies', 'Solutions',
                           'Corp', 'Inc', 'LLC', 'Ltd', 'Group', 'Holdings',
                           'Systems', 'Software', 'Dynamics', 'Partners', 'Analytics'}

        words = text.split()
        for i, word in enumerate(words):
            # Clean word of punctuation
            clean_word = word.strip('.,!?;:"()[]{}')

            # Skip if not a potential company name
            if not clean_word or len(clean_word) < 2 or len(clean_word) > 30:
                continue

            # Must start with capital letter or be all caps (but NOT just common words)
            if not (clean_word[0].isupper() or clean_word.isupper()):
                continue

            # Skip if in existing names
            if clean_word in existing_names or clean_word.lower() in existing_names_lower:
                continue

            # Skip common and generic words
            if clean_word in skip_words:
                continue

            # Check for multi-word company names
            potential_name = clean_word
            confidence = 0.4  # Base confidence for keyword fallback

            # If next word is a company suffix, include it and boost confidence
            if i + 1 < len(words):
                next_word = words[i + 1].strip('.,!?;:"()[]{}')
                if next_word in company_suffixes:
                    potential_name = f"{clean_word} {next_word}"
                    confidence = 0.55  # Higher confidence with company suffix

            # Additional validation: must contain at least one capital letter per word
            if not all(part[0].isupper() or part.isupper() for part in potential_name.split()):
                continue

            # Skip if already discovered in this batch
            if potential_name in [d.text for d in already_discovered]:
                continue
            if potential_name in [d.text for d in discovered]:
                continue

            # Create and save entity (only if confidence is reasonable)
            if confidence >= 0.4:
                new_entity = Entity(
                    name=potential_name,
                    entity_type='company',
                    confidence=confidence,
                    description=f"Auto-discovered company via keyword fallback (low priority)",
                    metadata={'source': 'keyword_fallback', 'is_fallback': True}
                )

                try:
                    self.entity_manager.add_entity(new_entity)
                    logger.info(f"⚠️ Learned entity via keyword fallback: {potential_name} (conf={confidence:.2f})")

                    start_pos = text.find(potential_name)
                    if start_pos >= 0:
                        discovered.append(ExtractedEntity(
                            text=potential_name,
                            entity_type=EntityType.COMPANY,
                            start_position=start_pos,
                            end_position=start_pos + len(potential_name),
                            confidence=confidence,
                            extraction_method="keyword_fallback"
                        ))
                except Exception as e:
                    logger.debug(f"Entity {potential_name} already exists or failed to save: {e}")

        return discovered

    def _calculate_enhanced_spacy_confidence(self, ent, doc, config: dict) -> float:
        """Calculate sophisticated confidence score for spaCy entities.

        Uses multiple linguistic features:
        - Dependency parsing context
        - Part-of-speech patterns
        - Entity consistency and shape
        - Surrounding context validation
        """
        base_confidence = 0.65  # Base confidence for spaCy entities
        label = ent.label_

        # 1. Label-specific base adjustments
        label_confidence_boost = {
            'ORG': 0.10,      # Organizations are generally accurate
            'PERSON': 0.12,   # Person names are very accurate
            'PRODUCT': 0.05,  # Products can be less precise
            'GPE': 0.15,      # Geo-political entities are very accurate
            'FAC': 0.05,      # Facilities can vary
            'EVENT': 0.08,
            'LAW': 0.05,
            'WORK_OF_ART': 0.02
        }
        base_confidence += label_confidence_boost.get(label, 0.0)

        # 2. Entity text analysis
        entity_text = ent.text.strip()

        # Capitalization patterns (proper capitalization = higher confidence)
        if entity_text.istitle() or entity_text.isupper():
            base_confidence += 0.05
        elif not entity_text[0].isupper():
            base_confidence -= 0.15  # Penalty for not starting with capital

        # Length considerations
        if 2 <= len(entity_text) <= 4:
            base_confidence -= 0.05  # Very short entities can be ambiguous
        elif len(entity_text) > 40:
            base_confidence -= 0.15  # Very long entities are often over-extractions

        # 3. Dependency parsing context (check if entity is in a valid grammatical role)
        try:
            # Get the head token (what this entity grammatically connects to)
            head_token = None
            for i, token in enumerate(doc):
                if ent.start <= i < ent.end:
                    head_token = token.head
                    break

            if head_token:
                # POS tag of the head can indicate entity quality
                head_pos = head_token.pos_

                # Good indicators: entity is subject/object of business verbs
                if head_pos in ['NOUN', 'PROPN']:
                    base_confidence += 0.03
                elif head_pos == 'VERB':
                    # Check if it's a business-related verb
                    business_verbs = {'announce', 'launch', 'release', 'develop',
                                     'acquire', 'partner', 'invest', 'create',
                                     'build', 'offer', 'provide', 'introduce'}
                    if head_token.lemma_.lower() in business_verbs:
                        base_confidence += 0.08

                # Dependency relation quality
                dep_relation = head_token.dep_ if head_token else ''
                good_relations = {'nsubj', 'nsubjpass', 'dobj', 'pobj', 'compound'}
                if dep_relation in good_relations:
                    base_confidence += 0.04

        except Exception as e:
            logger.debug(f"Dependency parsing failed for confidence: {e}")

        # 4. Context window analysis (check surrounding words)
        try:
            # Get context window (3 tokens before and after)
            start_idx = max(0, ent.start - 3)
            end_idx = min(len(doc), ent.end + 3)
            context_tokens = doc[start_idx:end_idx]

            context_text = ' '.join([t.text.lower() for t in context_tokens])

            # Business context indicators
            business_context_words = {
                'ceo', 'cto', 'founder', 'president', 'executive', 'company',
                'corporation', 'startup', 'firm', 'business', 'enterprise',
                'organization', 'agency', 'department', 'division', 'subsidiary',
                'headquarters', 'based', 'located', 'founded', 'launched'
            }

            # Tech context indicators
            tech_context_words = {
                'ai', 'artificial', 'intelligence', 'machine', 'learning',
                'technology', 'software', 'platform', 'service', 'product',
                'tool', 'application', 'system', 'model', 'algorithm'
            }

            # Check for business/tech context
            if any(word in context_text for word in business_context_words):
                base_confidence += 0.06
            if any(word in context_text for word in tech_context_words):
                base_confidence += 0.04

        except Exception as e:
            logger.debug(f"Context analysis failed: {e}")

        # 5. Company suffix bonuses (for ORG entities)
        if label == 'ORG':
            company_suffixes = ['Inc', 'LLC', 'Corp', 'Ltd', 'Corporation', 'Company',
                               'Labs', 'Technologies', 'Solutions', 'Group', 'Industries',
                               'International', 'Associates', 'Partners', 'Holdings',
                               'Dynamics', 'Systems', 'Software', 'Analytics']
            if any(suffix in entity_text for suffix in company_suffixes):
                base_confidence += 0.12

            # Common suffixes for AI/tech companies
            tech_suffixes = ['AI', 'Tech', 'Data', 'Cloud', 'Digital', 'Smart']
            if entity_text.endswith(tuple(tech_suffixes)):
                base_confidence += 0.08

        return min(max(base_confidence, 0.1), 1.0)

    def _validate_company_context(self, ent, doc) -> bool:
        """Validate that an ORG entity appears in a company-like context.

        Uses dependency parsing and POS tagging to ensure the entity
        is actually being used as a company name, not as a generic noun.

        Returns:
            True if the entity appears in valid company context
            False if the context suggests it's not a company
        """
        try:
            entity_text = ent.text.strip().lower()

            # 1. Blacklist: Words that are commonly mislabeled as ORG but aren't companies
            non_company_blacklist = {
                'team', 'department', 'division', 'group', 'unit', 'branch',
                'section', 'committee', 'council', 'board', 'panel',
                'government', 'administration', 'authority', 'ministry',
                'public', 'private', 'sector', 'industry', 'market',
                'community', 'society', 'association', 'organization',
                'university', 'college', 'school', 'institute', 'academy',
                'hospital', 'clinic', 'center', 'foundation'
            }

            # Check if entity is in blacklist (with or without "the")
            if entity_text in non_company_blacklist:
                return False
            if entity_text.replace('the ', '') in non_company_blacklist:
                return False

            # 2. Dependency parsing: Check grammatical role
            # Companies should appear as subjects/objects, not as adjectives
            for i, token in enumerate(doc):
                if ent.start <= i < ent.end:
                    # Get dependency relation
                    dep = token.dep_

                    # Good: entity is acting as subject or object
                    if dep in ['nsubj', 'nsubjpass', 'dobj', 'pobj', 'compound']:
                        return True

                    # Warning: entity is being used as an adjective (e.g., "Google search")
                    # Still might be valid, continue to next check
                    pass

            # 3. Context window: Check for business verbs around the entity
            start_idx = max(0, ent.start - 5)
            end_idx = min(len(doc), ent.end + 5)
            context_tokens = doc[start_idx:end_idx]

            # Business action verbs that typically precede company names
            company_action_verbs = {
                'announce', 'launch', 'release', 'introduce', 'unveil',
                'acquire', 'buy', 'purchase', 'merge', 'partner', 'collaborate',
                'compete', 'rival', 'challenge', 'join', 'leave', 'found',
                'lead', 'run', 'manage', 'head', 'operate'
            }

            context_lemmas = {t.lemma_.lower() for t in context_tokens}

            # If we see business action verbs, it's likely a company
            if context_lemmas & company_action_verbs:
                return True

            # 4. Pattern-based validation
            # Check for common company name patterns
            # Has legal suffix? Very likely a company
            legal_suffixes = ['inc', 'llc', 'corp', 'ltd', 'corporation']
            if any(entity_text.endswith(s) for s in legal_suffixes):
                return True

            # Has tech/AI suffix? Likely a company in our domain
            tech_suffixes = ['ai', 'labs', 'tech', 'technologies', 'systems']
            if any(entity_text.endswith(s) for s in tech_suffixes):
                return True

            # Multi-word with capitalization? Likely a proper noun (company)
            words = ent.text.strip().split()
            if len(words) >= 2 and all(w[0].isupper() for w in words if w):
                return True

            # Default: if not clearly rejected, accept it
            # (Better to have some false positives than false negatives)
            return True

        except Exception as e:
            logger.debug(f"Company context validation failed: {e}")
            return True  # Default to accepting on error

    def _get_confidence_breakdown(self, ent, doc) -> dict:
        """Get a breakdown of confidence factors for debugging."""
        breakdown = {
            'label': ent.label_,
            'text': ent.text,
            'capitalization': 'proper' if ent.text.istitle() or ent.text.isupper() else 'improper',
            'length': len(ent.text),
            'position': (ent.start_char, ent.end_char)
        }

        try:
            # Add dependency info if available
            for i, token in enumerate(doc):
                if ent.start <= i < ent.end:
                    breakdown['dependency'] = token.dep_
                    breakdown['head'] = token.head.text
                    breakdown['head_pos'] = token.head.pos_
                    break
        except:
            pass

        return breakdown

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
        """Remove duplicate entities, keeping highest confidence ones.

        Deduplication strategy:
        1. Group entities by normalized text (case-insensitive, no extra spaces)
        2. Within each group, keep only the highest confidence entity
        3. For overlapping entities, prefer spaCy methods over pattern/keyword
        4. Remove entities that are substrings of higher-confidence entities
        5. Remove single-word company entities that are parts of spaCy person entities
        """
        if not entities:
            return []

        # Sort by confidence (highest first) - spaCy enhanced discovery will naturally float to top
        entities.sort(key=lambda e: e.confidence, reverse=True)

        # Group by normalized text
        groups = {}
        for entity in entities:
            # Normalize: lowercase, strip extra spaces
            normalized = ' '.join(entity.text.lower().split())
            group_key = (normalized, entity.entity_type.value)

            if group_key not in groups:
                groups[group_key] = []
            groups[group_key].append(entity)

        # For each group, keep only the best entity
        unique_entities = []
        seen_texts = set()

        for group_key, group_entities in groups.items():
            normalized_text, entity_type = group_key

            # Skip if we've already seen a very similar entity
            if normalized_text in seen_texts:
                continue

            # Sort group by confidence, then by extraction method priority
            method_priority = {
                'spacy_enhanced_discovery': 5,
                'spacy_ner': 4,
                'known_entity_matching': 3,
                'pattern_matching': 2,
                'keyword_fallback': 1
            }

            group_entities.sort(
                key=lambda e: (
                    e.confidence,
                    method_priority.get(e.extraction_method, 0)
                ),
                reverse=True
            )

            # Keep the best entity from the group
            best_entity = group_entities[0]

            # Check if this entity is a substring of an existing entity (avoid "AI" vs "OpenAI")
            is_substring = False
            for existing in unique_entities:
                # Same type substring check
                if (best_entity.entity_type == existing.entity_type and
                    best_entity.text.lower() in existing.text.lower() and
                    len(best_entity.text) < len(existing.text)):
                    # This is a substring, skip it
                    is_substring = True
                    break

                # Cross-type check: if this is a single-word company entity and there's a multi-word PERSON entity
                # containing this word, skip the company entity (e.g., "Sundar" company vs "Sundar Pichai" person)
                if (best_entity.entity_type == EntityType.COMPANY and
                    existing.entity_type == EntityType.PERSON and
                    len(best_entity.text.split()) == 1 and
                    len(existing.text.split()) > 1 and
                    best_entity.text.lower() in existing.text.lower()):
                    # This single-word company is part of a person name, skip it
                    is_substring = True
                    break

            if not is_substring:
                unique_entities.append(best_entity)
                seen_texts.add(normalized_text)

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