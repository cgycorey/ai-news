"""Simplified NLP pipeline for AI news processing - business focused."""

import re
from typing import List, Dict, Optional, Tuple, Set, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

from .text_processor import TextProcessor, ProcessedText
from .entity_extractor import EntityExtractor, ExtractedEntity
from .spacy_utils import load_spacy_model, is_model_available
from .entity_types import EntityType, SentimentLabel

logger = logging.getLogger(__name__)


class SentimentResult:
    """Simple sentiment analysis result."""
    
    def __init__(self, label: SentimentLabel, score: float, confidence: float):
        self.label = label
        self.score = score  # -1.0 to 1.0
        self.confidence = confidence  # 0.0 to 1.0
    
    def __repr__(self):
        return f"SentimentResult(label={self.label.value}, score={self.score:.2f}, confidence={self.confidence:.2f})"


class TextClassification:
    """Simple text classification result."""
    
    def __init__(self, ai_relevant: bool, category: str, confidence: float):
        self.ai_relevant = ai_relevant
        self.category = category  # 'technology', 'business', 'research', 'policy', 'ethics'
        self.confidence = confidence
        self.keywords: List[str] = []
    
    def __repr__(self):
        return f"TextClassification(ai_relevant={self.ai_relevant}, category={self.category}, confidence={self.confidence:.2f})"


@dataclass
class NLPResult:
    """Complete NLP processing result."""
    original_text: str
    text_processed: Optional[ProcessedText] = None
    entities: List[ExtractedEntity] = field(default_factory=list)
    sentiment: Optional[SentimentResult] = None
    classification: Optional[TextClassification] = None
    processing_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'original_text': self.original_text,
            'entities': [entity.to_dict() for entity in self.entities],
            'sentiment': {
                'label': self.sentiment.label.value,
                'score': self.sentiment.score,
                'confidence': self.sentiment.confidence
            } if self.sentiment else None,
            'classification': {
                'ai_relevant': self.classification.ai_relevant,
                'category': self.classification.category,
                'confidence': self.classification.confidence,
                'keywords': self.classification.keywords
            } if self.classification else None,
            'processing_time': self.processing_time
        }


class NLPPipeline:
    """Simplified NLP pipeline focused on business intelligence.
    
    Core capabilities:
    1. Text preprocessing and cleaning
    2. Entity extraction (companies, products, technologies, people)
    3. Basic sentiment analysis
    4. Text classification (AI relevance and categorization)
    
    Removed academic features:
    - Topic modeling (LDA, clustering)
    - Complex sentiment analysis
    - Advanced ML classification
    - Named entity disambiguation
    """
    
    def __init__(self, use_spacy: bool = True):
        self.use_spacy = use_spacy and is_model_available()
        
        # Initialize components
        self.text_processor = TextProcessor()
        self.entity_extractor = EntityExtractor(self.text_processor, use_spacy)
        
        # Load spaCy model for sentiment if available
        self.nlp = None
        if self.use_spacy:
            try:
                self.nlp = load_spacy_model()
                logger.info("✅ spaCy model loaded for NLP pipeline")
            except Exception as e:
                logger.warning(f"Failed to load spaCy model: {e}")
                self.use_spacy = False
        
        # AI relevance keywords
        self.ai_keywords = self._load_ai_keywords()
        
        # Category keywords
        self.category_keywords = self._load_category_keywords()
        
        # Sentiment words
        self.positive_words = self._load_positive_words()
        self.negative_words = self._load_negative_words()
    
    def _load_ai_keywords(self) -> Set[str]:
        """Load AI-related keywords for relevance detection."""
        return {
            'artificial intelligence', 'machine learning', 'deep learning', 'neural network',
            'AI', 'ML', 'DL', 'GAN', 'transformer', 'attention mechanism',
            'large language model', 'LLM', 'generative AI', 'GPT', 'ChatGPT',
            'Claude', 'Gemini', 'Bard', 'Llama', 'Mistral', 'computer vision',
            'natural language processing', 'NLP', 'reinforcement learning',
            'robotics', 'automation', 'algorithm', 'data science', 'big data',
            'OpenAI', 'Anthropic', 'Google AI', 'Microsoft AI', 'Meta AI',
            'autonomous vehicle', 'self-driving', 'AI ethics', 'AI safety'
        }
    
    def _load_category_keywords(self) -> Dict[str, Set[str]]:
        """Load category-specific keywords."""
        return {
            'technology': {
                'software', 'hardware', 'infrastructure', 'cloud', 'database',
                'API', 'framework', 'library', 'platform', 'architecture',
                'scalability', 'performance', 'optimization', 'deployment'
            },
            'business': {
                'revenue', 'profit', 'investment', 'funding', 'startup', 'acquisition',
                'merger', 'IPO', 'stock', 'market share', 'competition', 'strategy',
                'customers', 'sales', 'marketing', 'partnership', 'enterprise'
            },
            'research': {
                'research', 'study', 'paper', 'journal', 'conference', 'university',
                'breakthrough', 'innovation', 'discovery', 'experiment', 'results',
                'methodology', 'analysis', 'findings', 'publication', 'peer review'
            },
            'policy': {
                'regulation', 'policy', 'government', 'law', 'legislation', 'compliance',
                'privacy', 'security', 'safety', 'ethics', 'governance', 'standard',
                'guideline', 'framework', 'oversight', 'transparency', 'accountability'
            },
            'ethics': {
                'ethics', 'ethical', 'bias', 'fairness', 'discrimination', 'transparency',
                'accountability', 'privacy', 'consent', 'human rights', 'social impact',
                'responsible AI', 'AI safety', 'beneficial AI', 'alignment'
            }
        }
    
    def _load_positive_words(self) -> Set[str]:
        """Load positive sentiment words."""
        return {
            'good', 'great', 'excellent', 'amazing', 'fantastic', 'wonderful',
            'outstanding', 'impressive', 'innovative', 'breakthrough', 'success',
            'effective', 'efficient', 'powerful', 'useful', 'valuable', 'important',
            'significant', 'positive', 'beneficial', 'helpful', 'promising', 'exciting'
        }
    
    def _load_negative_words(self) -> Set[str]:
        """Load negative sentiment words."""
        return {
            'bad', 'terrible', 'awful', 'horrible', 'disaster', 'failure',
            'ineffective', 'inefficient', 'useless', 'worthless', 'dangerous',
            'harmful', 'negative', 'problem', 'issue', 'concern', 'risk',
            'threat', 'challenge', 'difficult', 'complex', 'controversial', 'controversy'
        }
    
    def process_text(self, text: str, title: str = "") -> NLPResult:
        """Process text through the complete NLP pipeline."""
        import time
        start_time = time.time()
        
        # Combine title and text for processing
        full_text = f"{title} {text}" if title else text
        
        # Text preprocessing
        processed_text = self.text_processor.process_text(full_text)
        
        # Entity extraction
        entities = self.entity_extractor.extract_entities(full_text)
        
        # Sentiment analysis
        sentiment = self._analyze_sentiment(full_text)
        
        # Text classification
        classification = self._classify_text(full_text)
        
        processing_time = time.time() - start_time
        
        return NLPResult(
            original_text=full_text,
            text_processed=processed_text,
            entities=entities,
            sentiment=sentiment,
            classification=classification,
            processing_time=processing_time
        )
    
    def _analyze_sentiment(self, text: str) -> Optional[SentimentResult]:
        """Simple sentiment analysis using keyword matching and spaCy if available."""
        if not text:
            return None
        
        text_lower = text.lower()
        words = text_lower.split()
        
        # Count positive and negative words
        positive_count = sum(1 for word in words if word in self.positive_words)
        negative_count = sum(1 for word in words if word in self.negative_words)
        
        total_sentiment_words = positive_count + negative_count
        
        if total_sentiment_words == 0:
            # Neutral sentiment
            return SentimentResult(SentimentLabel.NEUTRAL, 0.0, 0.5)
        
        # Calculate sentiment score (-1 to 1)
        sentiment_score = (positive_count - negative_count) / total_sentiment_words
        
        # Determine sentiment label
        if sentiment_score > 0.1:
            label = SentimentLabel.POSITIVE
        elif sentiment_score < -0.1:
            label = SentimentLabel.NEGATIVE
        else:
            label = SentimentLabel.NEUTRAL
        
        # Calculate confidence based on how many sentiment words were found
        confidence = min(total_sentiment_words / 10.0, 1.0)  # Cap at 1.0
        
        return SentimentResult(label, sentiment_score, confidence)
    
    def _classify_text(self, text: str) -> Optional[TextClassification]:
        """Classify text for AI relevance and category."""
        if not text:
            return None
        
        text_lower = text.lower()
        
        # Check AI relevance
        ai_keywords_found = [kw for kw in self.ai_keywords if kw in text_lower]
        ai_relevant = len(ai_keywords_found) > 0
        ai_confidence = min(len(ai_keywords_found) / 3.0, 1.0)  # Cap at 1.0
        
        # Categorize text
        category_scores = {}
        for category, keywords in self.category_keywords.items():
            found_keywords = [kw for kw in keywords if kw in text_lower]
            category_scores[category] = len(found_keywords)
        
        # Determine primary category
        if category_scores:
            primary_category = max(category_scores, key=category_scores.get)
            category_confidence = min(category_scores[primary_category] / 3.0, 1.0)
        else:
            primary_category = 'general'
            category_confidence = 0.5
        
        # Overall confidence (average of AI relevance and category confidence)
        overall_confidence = (ai_confidence + category_confidence) / 2.0
        
        classification = TextClassification(
            ai_relevant=ai_relevant,
            category=primary_category,
            confidence=overall_confidence
        )
        
        # Add found keywords
        classification.keywords = ai_keywords_found[:5]  # Top 5 keywords
        
        return classification
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get statistics about the NLP pipeline."""
        return {
            'spacy_available': self.use_spacy,
            'spacy_model_loaded': self.nlp is not None,
            'ai_keywords_count': len(self.ai_keywords),
            'category_count': len(self.category_keywords),
            'positive_words_count': len(self.positive_words),
            'negative_words_count': len(self.negative_words),
            'entity_extractor_stats': self.entity_extractor.get_extraction_stats()
        }


def create_default_pipeline(use_spacy: bool = True) -> NLPPipeline:
    """Create a default NLP pipeline."""
    return NLPPipeline(use_spacy)


def create_simple_pipeline() -> NLPPipeline:
    """Create a simple pipeline without spaCy dependencies."""
    return NLPPipeline(use_spacy=False)