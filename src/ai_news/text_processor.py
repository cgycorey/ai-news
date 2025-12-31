"""Text processing utilities for AI news intelligence.

NLTK, textblob, and langdetect removed - using basic string operations.
Core collection and digest operations use FastEmbed for semantic processing.
"""

import re
import html
import string
from collections import Counter
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse
import logging

logger = logging.getLogger(__name__)

# Import security utilities
try:
    from .security_utils import clean_text_content, sanitize_html
    SECURITY_AVAILABLE = True
except ImportError:
    SECURITY_AVAILABLE = False
    logger.warning("security_utils not available. Using fallback HTML cleaning.")


@dataclass
class ProcessedText:
    """Container for processed text data."""
    original_text: str
    cleaned_text: str
    normalized_text: str
    sentences: List[str]
    tokens: List[str]
    lemmatized_tokens: List[str]
    filtered_tokens: List[str]
    keywords: List[str]
    language: str
    word_count: int
    sentence_count: int
    readability_score: float


@dataclass
class TextStatistics:
    """Text analysis statistics."""
    char_count: int
    word_count: int
    sentence_count: int
    paragraph_count: int
    avg_sentence_length: float
    avg_word_length: float
    readability_score: float
    keyword_density: Dict[str, float]
    language: str
    confidence: float


class TextProcessor:
    """Text processing with basic string operations."""
    
    # Standard English stopwords
    STANDARD_STOPWORDS = {
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
        'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
        'to', 'was', 'will', 'with', 'i', 'you', 'your', 'we', 'our', 'they',
        'their', 'this', 'these', 'those', 'am', 'been', 'being', 'did', 'do',
        'does', 'had', 'have', 'having', 'may', 'might', 'must', 'shall',
        'should', 'would', 'could', 'can', 'cannot', 'might', 'must', 'shall',
        'should', 'will', 'would'
    }
    
    def __init__(self, spacy_model: Optional[str] = None):
        """Initialize the text processor.
        
        Args:
            spacy_model: Ignored, kept for compatibility
        """
        self.nlp = None
        
        # AI/Tech domain specific stopwords
        self.tech_stopwords = {
            'ai', 'artificial', 'intelligence', 'machine', 'learning', 
            'deep', 'neural', 'network', 'algorithm', 'data', 'model',
            'system', 'technology', 'software', 'platform', 'service',
            'company', 'startup', 'tech', 'digital', 'online', 'web'
        }
        
        # Compile regex patterns for performance
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.phone_pattern = re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b')
        self.html_tag_pattern = re.compile(r'<[^>]+>')
        self.whitespace_pattern = re.compile(r'\s+')
        self.punctuation_pattern = re.compile(f'[{re.escape(string.punctuation)}]')
    
    def clean_html(self, text: str) -> str:
        """Clean HTML content and extract meaningful text securely.
        
        Args:
            text: Text that may contain HTML
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Use secure HTML cleaning if available
        if SECURITY_AVAILABLE:
            try:
                return clean_text_content(text)
            except Exception as e:
                logger.warning(f"Secure HTML cleaning failed: {e}")
        
        # Fallback to regex-based HTML removal
        text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
        text = self.html_tag_pattern.sub(' ', text)
        
        # Remove dangerous patterns
        text = re.sub(r'javascript:', '', text, flags=re.IGNORECASE)
        text = re.sub(r'vbscript:', '', text, flags=re.IGNORECASE)
        text = re.sub(r'on\w+\s*=', '', text, flags=re.IGNORECASE)
        
        return text
    
    def normalize_text(self, text: str) -> str:
        """Normalize text by handling encoding, whitespace, and special characters.
        
        Args:
            text: Input text
            
        Returns:
            Normalized text
        """
        if not text:
            return ""
        
        # HTML decode
        text = html.unescape(text)
        
        # Remove URLs, emails, and phone numbers
        text = self.url_pattern.sub(' [URL] ', text)
        text = self.email_pattern.sub(' [EMAIL] ', text)
        text = self.phone_pattern.sub(' [PHONE] ', text)
        
        # Handle special characters and unicode
        text = text.replace('—', ' -- ')
        text = text.replace('–', '-')
        text = text.replace('"', '"')
        text = text.replace(''', "'")
        text = text.replace(''', "'")
        text = text.replace('…', '...')
        
        # Remove excessive whitespace
        text = self.whitespace_pattern.sub(' ', text)
        text = text.strip()
        
        return text
    
    def detect_language(self, text: str) -> Tuple[str, float]:
        """Detect language (simplified - always returns English).
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (language_code, confidence_score)
        """
        return 'en', 0.5
    
    def tokenize_and_lemmatize(self, text: str, language: str = 'en') -> Tuple[List[str], List[str]]:
        """Tokenize text using basic string operations.
        
        Args:
            text: Input text
            language: Language code (ignored)
            
        Returns:
            Tuple of (tokens, lemmatized_tokens)
        """
        if not text:
            return [], []
        
        # Basic word tokenization
        tokens = re.findall(r'\b\w+\b', text)
        lemmatized_tokens = [token.lower() for token in tokens if token.isalpha()]
        
        return tokens, lemmatized_tokens
    
    def filter_stopwords(self, tokens: List[str], custom_stopwords: Optional[Set[str]] = None) -> List[str]:
        """Filter out stopwords and common terms.
        
        Args:
            tokens: List of tokens
            custom_stopwords: Additional stopwords to filter
            
        Returns:
            Filtered tokens
        """
        if not tokens:
            return []
        
        # Start with standard stopwords
        stop_words = self.STANDARD_STOPWORDS.copy()
        
        # Add custom stopwords
        if custom_stopwords:
            stop_words.update(custom_stopwords)
        
        # Add tech domain stopwords
        stop_words.update(self.tech_stopwords)
        
        # Filter tokens
        filtered = [
            token for token in tokens 
            if token.lower() not in stop_words
            and len(token) > 2
            and not token.isdigit()
            and not token.isnumeric()
        ]
        
        return filtered
    
    def extract_keywords(self, text: str, max_keywords: int = 20) -> List[str]:
        """Extract keywords using token frequency.
        
        Args:
            text: Input text
            max_keywords: Maximum number of keywords to extract
            
        Returns:
            List of keywords
        """
        if not text or len(text) < 100:
            return []
        
        try:
            # Simple tokenization
            tokens = re.findall(r'\b\w+\b', text.lower())
            
            # Basic stopword filtering
            filtered_tokens = [
                token for token in tokens 
                if token not in self.STANDARD_STOPWORDS 
                and len(token) > 2
            ]
            
            if filtered_tokens:
                word_freq = Counter(filtered_tokens)
                keywords = [word for word, freq in word_freq.most_common(max_keywords)]
            else:
                keywords = []
            
            # Filter by quality
            quality_keywords = [
                kw for kw in keywords
                if len(kw) > 2
                and not kw.isnumeric()
            ]
            
            return quality_keywords[:max_keywords]
            
        except Exception as e:
            logger.error(f"Keyword extraction failed: {e}")
            return []
    
    def calculate_readability(self, text: str) -> float:
        """Calculate readability score (0-1, higher is more readable).
        
        Args:
            text: Input text
            
        Returns:
            Readability score
        """
        if not text:
            return 0.0
        
        try:
            # Basic sentence splitting
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            words = text.split()
            
            if not sentences or not words:
                return 0.0
            
            # Calculate averages
            avg_sentence_length = len(words) / len(sentences)
            avg_word_length = sum(len(word) for word in words) / len(words)
            
            # Simple readability score
            sentence_score = max(0, 1 - (avg_sentence_length - 15) / 25)
            word_score = max(0, 1 - (avg_word_length - 5) / 10)
            
            readability = (sentence_score + word_score) / 2
            return max(0, min(1, readability))
            
        except Exception as e:
            logger.debug(f"Readability scoring failed: {e}")
            return 0.5
    
    def process_text(self, text: str) -> ProcessedText:
        """Process text through the complete pipeline.
        
        Args:
            text: Input text
            
        Returns:
            ProcessedText object with all processed data
        """
        if not text:
            return ProcessedText(
                original_text="",
                cleaned_text="",
                normalized_text="",
                sentences=[],
                tokens=[],
                lemmatized_tokens=[],
                filtered_tokens=[],
                keywords=[],
                language="en",
                word_count=0,
                sentence_count=0,
                readability_score=0.0
            )
        
        # Step 1: Clean HTML
        cleaned_text = self.clean_html(text)
        
        # Step 2: Normalize text
        normalized_text = self.normalize_text(cleaned_text)
        
        # Step 3: Detect language
        language, lang_confidence = self.detect_language(normalized_text)
        
        # Step 4: Sentence segmentation
        sentences = [s.strip() for s in normalized_text.replace('\n', '. ').split('.') if s.strip()]
        
        # Step 5: Tokenization
        tokens, lemmatized_tokens = self.tokenize_and_lemmatize(normalized_text, language)
        
        # Step 6: Stopword filtering
        filtered_tokens = self.filter_stopwords(lemmatized_tokens)
        
        # Step 7: Keyword extraction
        keywords = self.extract_keywords(normalized_text)
        
        # Step 8: Calculate statistics
        word_count = len(tokens)
        sentence_count = len(sentences)
        readability_score = self.calculate_readability(normalized_text)
        
        return ProcessedText(
            original_text=text,
            cleaned_text=cleaned_text,
            normalized_text=normalized_text,
            sentences=sentences,
            tokens=tokens,
            lemmatized_tokens=lemmatized_tokens,
            filtered_tokens=filtered_tokens,
            keywords=keywords,
            language=language,
            word_count=word_count,
            sentence_count=sentence_count,
            readability_score=readability_score
        )
    
    def get_text_statistics(self, text: str) -> TextStatistics:
        """Get comprehensive text statistics.
        
        Args:
            text: Input text
            
        Returns:
            TextStatistics object
        """
        processed = self.process_text(text)
        
        # Calculate additional statistics
        char_count = len(processed.normalized_text)
        word_count = processed.word_count
        sentence_count = processed.sentence_count
        
        # Count paragraphs
        paragraphs = [p.strip() for p in processed.normalized_text.split('\n\n') if p.strip()]
        paragraph_count = len(paragraphs)
        
        # Calculate averages
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
        avg_word_length = sum(len(word) for word in processed.tokens) / word_count if word_count > 0 else 0
        
        # Calculate keyword density
        keyword_density = {}
        if processed.filtered_tokens:
            token_freq = Counter(processed.filtered_tokens)
            total_tokens = len(processed.filtered_tokens)
            keyword_density = {word: freq / total_tokens for word, freq in token_freq.most_common(20)}
        
        return TextStatistics(
            char_count=char_count,
            word_count=word_count,
            sentence_count=sentence_count,
            paragraph_count=paragraph_count,
            avg_sentence_length=avg_sentence_length,
            avg_word_length=avg_word_length,
            readability_score=processed.readability_score,
            keyword_density=keyword_density,
            language=processed.language,
            confidence=0.8
        )


def get_default_processor() -> TextProcessor:
    """Get a default text processor instance."""
    return TextProcessor()
