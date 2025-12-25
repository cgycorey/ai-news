"""Advanced text processing utilities for AI news intelligence."""

import re
import html
import string
from collections import Counter
from typing import List, Dict, Optional, Tuple, Set
# from bs4 import BeautifulSoup, Tag
BeautifulSoup = None
Tag = None
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse

# Import security utilities
try:
    from .security_utils import clean_text_content, sanitize_html
    SECURITY_AVAILABLE = True
except ImportError:
    SECURITY_AVAILABLE = False
    print("Warning: security_utils not available. Using fallback HTML cleaning.")
# Heavy imports done lazily to improve startup performance

# Import these lazily to avoid startup issues if packages aren't installed
nltk = None
langdetect = None
spacy = None
Language = None
TfidfVectorizer = None

def _import_langdetect():
    """Import langdetect lazily."""
    global langdetect
    try:
        import langdetect
        return langdetect
    except ImportError:
        logger.warning("langdetect not available, using English as default")
        return None

def _import_spacy():
    """Import spacy lazily."""
    global spacy, Language
    try:
        import spacy
        from spacy.language import Language
        return spacy, Language
    except ImportError:
        logger.warning("spaCy not available, using basic processing")
        return None, None

def _import_sklearn():
    """Import sklearn TF-IDF lazily."""
    global TfidfVectorizer
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        return TfidfVectorizer
    except ImportError:
        logger.warning("scikit-learn not available, TF-IDF analysis disabled")
        return None

# Import heavy NLP libraries only when needed
def _import_nltk_components():
    """Import NLTK components lazily."""
    global WordNetLemmatizer
    from nltk.stem import WordNetLemmatizer
    return WordNetLemmatizer

def _import_textblob():
    """Import TextBlob lazily."""
    global TextBlob
    from textblob import TextBlob
    return TextBlob
from collections import Counter
import logging

from .spacy_utils import load_spacy_model

logger = logging.getLogger(__name__)

# Import NLTK utilities for persistent caching
try:
    from .nltk_utils import ensure_nltk_data_lazy
    NLTK_UTILS_AVAILABLE = True
except ImportError:
    NLTK_UTILS_AVAILABLE = False
    logger.warning("NLTK utils not available. NLTK data may be downloaded repeatedly.")
    
    # Fallback function
    def ensure_nltk_data_lazy(package_id: str, resource_path: str) -> bool:
        """Fallback NLTK data checker."""
        try:
            nltk.data.find(resource_path)
            return True
        except LookupError:
            try:
                nltk.download(package_id, quiet=True)
                nltk.data.find(resource_path)
                return True
            except Exception:
                return False

# Configure NLTK data paths to avoid path issues
import os
_NLTK_DATA_PATHS = [
    os.path.expanduser('~/nltk_data'),
    os.path.join(os.path.dirname(__file__), '../../nltk_data'),
    '/usr/share/nltk_data',
    '/usr/local/share/nltk_data'
]

# Add common paths to NLTK data path if they exist
try:
    import nltk
    for path in _NLTK_DATA_PATHS:
        if os.path.exists(path) and path not in nltk.data.path:
            nltk.data.path.append(path)
except ImportError:
    # NLTK not available, skip path configuration
    pass


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
    """Advanced text processing with NLP capabilities."""
    
    def __init__(self, spacy_model: Optional[str] = "en_core_web_sm"):
        """Initialize the text processor.
        
        Args:
            spacy_model: Name of spaCy model to use
        """
        self.spacy_model_name = spacy_model
        self.nlp = None
        # Lazy load NLTK lemmatizer
        try:
            nltk = __import__('nltk')
            from nltk.stem import WordNetLemmatizer
            self.lemmatizer = WordNetLemmatizer()
        except ImportError as e:
            logger.warning(f"Could not load NLTK lemmatizer: {e}. Using basic processing.")
            self.lemmatizer = None
        
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
        
        # Load spaCy model
        self._load_spacy_model()
    
    def _load_spacy_model(self):
        """Load spaCy model for advanced NLP."""
        self.nlp = load_spacy_model(self.spacy_model_name, auto_download=False)
        if self.nlp:
            logger.info(f"Loaded spaCy model: {self.spacy_model_name}")
        else:
            logger.warning(f"spaCy model {self.spacy_model_name} not found. Using basic processing.")
            logger.info(f"Run 'uv run ai-news setup-spacy' to download required models")
            logger.info("spaCy model should be included with the project")
            self.nlp = None
    
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
                # Fall back to BeautifulSoup method
        
        # Fallback to BeautifulSoup with improved security
        try:
            # Remove HTML comments
            text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
            
            # Parse with BeautifulSoup
            soup = BeautifulSoup(text, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Remove common non-content elements
            for element in soup.find_all(['nav', 'header', 'footer', 'aside', 'advertisement']):
                element.decompose()
            
            # Remove potentially dangerous attributes
            for tag in soup.find_all():
                if tag.has_attr('onclick'):
                    del tag['onclick']
                if tag.has_attr('onload'):
                    del tag['onload']
                if tag.has_attr('onerror'):
                    del tag['onerror']
                if tag.has_attr('href') and tag['href'].startswith('javascript:'):
                    del tag['href']
            
            # Extract text with paragraph structure
            text = soup.get_text(separator=' ')
            
        except Exception as e:
            logger.warning(f"HTML parsing failed: {e}")
            # Fallback to regex-based HTML removal
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
        """Detect the language of the text.
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (language_code, confidence_score)
        """
        if not text or len(text) < 50:
            return 'en', 0.5  # Default to English for very short text
        
        try:
            langdetect = _import_langdetect()
            if langdetect is None:
                return 'en', 0.3  # Low confidence fallback
            lang = langdetect.detect(text)
            # Get confidence by checking multiple samples
            samples = text.split('\n')[:5]  # First 5 lines
            if len(samples) > 1:
                detections = [langdetect.detect(sample) for sample in samples if len(sample) > 20]
                if detections:
                    confidence = detections.count(lang) / len(detections)
                    return lang, confidence
            return lang, 0.8
        except Exception as e:
            logger.debug(f"Language detection failed: {e}, falling back to English")
            return 'en', 0.5  # Fallback to English
    
    def tokenize_and_lemmatize(self, text: str, language: str = 'en') -> Tuple[List[str], List[str]]:
        """Tokenize and lemmatize text.
        
        Args:
            text: Input text
            language: Language code
            
        Returns:
            Tuple of (tokens, lemmatized_tokens)
        """
        if not text:
            return [], []
        
        # Use spaCy if available
        if self.nlp and language == 'en':
            doc = self.nlp(text)
            tokens = [token.text for token in doc if not token.is_space]
            lemmatized_tokens = [token.lemma_.lower() for token in doc if not token.is_space and not token.is_punct]
        else:
            # Fallback to NLTK - ensure data is available first
            if not ensure_nltk_data_lazy('punkt', 'tokenizers/punkt'):
                logger.error("NLTK punkt data not available. Please run 'uv run ai-news setup-nltk'")
                # Fallback to basic tokenization
                tokens = text.split()
                return tokens, [token.lower() for token in tokens if token.isalpha()]
            
            if not ensure_nltk_data_lazy('averaged_perceptron_tagger', 'taggers/averaged_perceptron_tagger'):
                logger.error("NLTK POS tagger data not available. Please run 'uv run ai-news setup-nltk'")
                # Basic tokenization without POS tagging
                from nltk.tokenize import word_tokenize
                tokens = word_tokenize(text)
                lemmatized_tokens = [self.lemmatizer.lemmatize(word.lower()) for word in tokens if word.isalpha() and len(word) > 1]
            else:
                # Full NLTK processing
                from nltk.tokenize import word_tokenize
                from nltk.tag import pos_tag
                from nltk.corpus import wordnet
                
                tokens = word_tokenize(text)
                lemmatized_tokens = []
                
                # Get POS tags for better lemmatization
                pos_tags = pos_tag(tokens)
                
                for word, pos in pos_tags:
                    if word.isalpha() and len(word) > 1:
                        # Map POS tag to WordNet format
                        wn_pos = self._get_wordnet_pos(pos)
                        if wn_pos:
                            lemma = self.lemmatizer.lemmatize(word.lower(), pos=wn_pos)
                        else:
                            lemma = self.lemmatizer.lemmatize(word.lower())
                        lemmatized_tokens.append(lemma)
        
        return tokens, lemmatized_tokens
    
    def _get_wordnet_pos(self, treebank_tag: str) -> Optional[str]:
        """Convert Treebank POS tag to WordNet format."""
        # Import wordnet lazily
        if not ensure_nltk_data_lazy('wordnet', 'corpora/wordnet'):
            # Fallback without wordnet
            if treebank_tag.startswith('J'):
                return 'a'  # adjective
            elif treebank_tag.startswith('V'):
                return 'v'  # verb
            elif treebank_tag.startswith('N'):
                return 'n'  # noun
            elif treebank_tag.startswith('R'):
                return 'r'  # adverb
            return None
        
        from nltk.corpus import wordnet
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        return None
    
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
        
        # Get NLTK stopwords - ensure data is available
        if ensure_nltk_data_lazy('stopwords', 'corpora/stopwords'):
            from nltk.corpus import stopwords
            standard_stopwords = set(stopwords.words('english'))
        else:
            logger.warning("NLTK stopwords not available. Using basic stopword list.")
            # Basic fallback stopword list
            standard_stopwords = {
                'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
                'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
                'to', 'was', 'will', 'with', 'i', 'you', 'your', 'we', 'our', 'they',
                'their', 'this', 'these', 'those', 'am', 'been', 'being', 'did', 'do',
                'does', 'had', 'have', 'having', 'may', 'might', 'must', 'shall',
                'should', 'would', 'could', 'can', 'cannot', 'could', 'might',
                'must', 'shall', 'should', 'will', 'would'
            }
        
        # Add custom stopwords
        stop_words = standard_stopwords.copy()
        if custom_stopwords:
            stop_words.update(custom_stopwords)
        
        # Add tech domain stopwords
        stop_words.update(self.tech_stopwords)
        
        # Filter tokens
        filtered = [
            token for token in tokens 
            if token.lower() not in stop_words
            and len(token) > 2  # Remove very short tokens
            and not token.isdigit()  # Remove pure numbers
            and not token.isnumeric()  # Remove numeric strings
        ]
        
        return filtered
    
    def extract_keywords(self, text: str, max_keywords: int = 20) -> List[str]:
        """Extract keywords using TF-IDF and TextBlob.
        
        Args:
            text: Input text
            max_keywords: Maximum number of keywords to extract
            
        Returns:
            List of keywords
        """
        if not text or len(text) < 100:
            return []
        
        try:
            # Method 1: TF-IDF based keywords
            try:
                TfidfVectorizer = _import_sklearn()
                if TfidfVectorizer is None:
                    tfidf_keywords = []
                else:
                    vectorizer = TfidfVectorizer(
                        max_features=max_keywords * 2,
                        stop_words='english',
                        ngram_range=(1, 2),
                        min_df=1,
                        max_df=0.8
                    )
                    tfidf_matrix = vectorizer.fit_transform([text])
                    feature_names = vectorizer.get_feature_names_out()
                    tfidf_scores = tfidf_matrix.toarray()[0]
                    
                    # Get top TF-IDF keywords
                    tfidf_keywords = [
                        feature_names[i] for i in tfidf_scores.argsort()[-max_keywords:][::-1]
                    ]
            except Exception as e:
                logger.debug(f"TF-IDF keyword extraction failed: {e}")
                tfidf_keywords = []
            
            # Method 2: TextBlob noun phrases
            try:
                TextBlob = _import_textblob()
                blob = TextBlob(text)
                noun_phrases = [phrase.lower() for phrase in blob.noun_phrases[:max_keywords]]
            except Exception as e:
                logger.debug(f"TextBlob noun phrase extraction failed: {e}")
                noun_phrases = []
            
            # Method 3: Simple token frequency without recursion
            try:
                # Simple tokenization without going through full process_text to avoid recursion
                tokens = re.findall(r'\b\w+\b', text.lower())
                # Basic stopword filtering
                basic_stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
                filtered_tokens = [token for token in tokens if token not in basic_stopwords and len(token) > 2]
                
                if filtered_tokens:
                    word_freq = Counter(filtered_tokens)
                    freq_keywords = [word for word, freq in word_freq.most_common(max_keywords)]
                else:
                    freq_keywords = []
            except Exception as e:
                logger.debug(f"Token frequency keyword extraction failed: {e}")
                freq_keywords = []
            
            # Combine and deduplicate keywords
            all_keywords = list(set(tfidf_keywords + noun_phrases + freq_keywords))
            
            # Filter and rank by length and quality
            quality_keywords = [
                kw for kw in all_keywords
                if len(kw.split()) <= 3  # Max 3 words
                and len(kw) > 2  # Min 3 characters
                and not kw.isnumeric()  # Not just numbers
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
            # Ensure NLTK data is available for tokenization
            if ensure_nltk_data_lazy('punkt', 'tokenizers/punkt'):
                from nltk.tokenize import sent_tokenize, word_tokenize
                sentences = sent_tokenize(text)
                words = word_tokenize(text)
            else:
                # Fallback to basic splitting
                sentences = [s.strip() for s in text.split('.') if s.strip()]
                words = text.split()
            
            if not sentences or not words:
                return 0.0
            
            # Calculate average sentence length and word length
            avg_sentence_length = len(words) / len(sentences)
            avg_word_length = sum(len(word) for word in words) / len(words)
            
            # Simple readability score based on sentence and word complexity
            # Lower scores for very long sentences and very long words
            sentence_score = max(0, 1 - (avg_sentence_length - 15) / 25)  # Optimal around 15 words
            word_score = max(0, 1 - (avg_word_length - 5) / 10)  # Optimal around 5 characters
            
            readability = (sentence_score + word_score) / 2
            return max(0, min(1, readability))
            
        except Exception as e:
            logger.debug(f"Readability scoring failed: {e}, returning default readability")
            return 0.5  # Default to medium readability
    
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
        if ensure_nltk_data_lazy('punkt', 'tokenizers/punkt'):
            try:
                from nltk.tokenize import sent_tokenize
                sentences = sent_tokenize(normalized_text)
            except Exception as e:
                logger.warning(f"NLTK sentence tokenization failed: {e}")
                # Fallback: split by periods and newlines
                sentences = [s.strip() for s in normalized_text.replace('\n', '. ').split('.') if s.strip()]
        else:
            # Fallback: split by periods and newlines
            sentences = [s.strip() for s in normalized_text.replace('\n', '. ').split('.') if s.strip()]
        
        # Step 5: Tokenization and lemmatization
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
            confidence=0.8  # Default confidence
        )
    
    def batch_process(self, texts: List[str], show_progress: bool = True) -> List[ProcessedText]:
        """Process multiple texts efficiently.
        
        Args:
            texts: List of texts to process
            show_progress: Whether to show progress bar
            
        Returns:
            List of ProcessedText objects
        """
        from tqdm import tqdm
        
        if show_progress:
            texts = tqdm(texts, desc="Processing texts")
        
        results = []
        for text in texts:
            try:
                processed = self.process_text(text)
                results.append(processed)
            except Exception as e:
                logger.error(f"Error processing text: {e}")
                # Add empty processed text for failed processing
                results.append(ProcessedText(
                    original_text=text,
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
                ))
        
        return results


class TextCache:
    """Cache for processed text to avoid reprocessing."""
    
    def __init__(self, cache_dir: str = "text_cache", max_size: int = 10000):
        """Initialize text cache.
        
        Args:
            cache_dir: Directory for cache storage
            max_size: Maximum number of cached items
        """
        import diskcache as dc
        self.cache = dc.Cache(cache_dir, size_limit=max_size)
    
    def get(self, text_hash: str) -> Optional[ProcessedText]:
        """Get cached processed text.
        
        Args:
            text_hash: Hash of the original text
            
        Returns:
            Cached ProcessedText or None
        """
        return self.cache.get(text_hash)
    
    def set(self, text_hash: str, processed_text: ProcessedText) -> None:
        """Cache processed text.
        
        Args:
            text_hash: Hash of the original text
            processed_text: ProcessedText to cache
        """
        self.cache.set(text_hash, processed_text)
    
    def clear(self) -> None:
        """Clear the cache."""
        self.cache.clear()
    
    def size(self) -> int:
        """Get cache size."""
        return len(self.cache)


def create_text_hash(text: str) -> str:
    """Create a hash for text to use as cache key.
    
    Args:
        text: Text to hash
        
    Returns:
        Hash string
    """
    import hashlib
    return hashlib.md5(text.encode('utf-8')).hexdigest()[:16]


def get_default_processor() -> TextProcessor:
    """Get a default text processor instance."""
    return TextProcessor()