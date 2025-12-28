"""Performance tracking for NLP operations."""

import time
import threading
from typing import List, Dict, Any
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Track NLP performance metrics."""
    
    # Timing
    total_collection_time: float = 0.0
    entity_extraction_times: List[float] = field(default_factory=list)
    
    # Counts
    articles_processed: int = 0
    spaCy_calls: int = 0
    pattern_only_calls: int = 0
    hybrid_calls: int = 0
    
    # Timestamps
    start_time: float = field(default_factory=time.time)
    
    # Thread safety
    _lock = threading.Lock()
    
    def record_entity_extraction(self, duration: float, method: str):
        """Record a single entity extraction operation."""
        with self._lock:
            self.entity_extraction_times.append(duration)
            self.articles_processed += 1
            
            if method == "spacy":
                self.spaCy_calls += 1
            elif method == "pattern":
                self.pattern_only_calls += 1
            elif method == "hybrid":
                self.hybrid_calls += 1
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics."""
        with self._lock:
            if not self.entity_extraction_times:
                return {
                    "articles_processed": 0,
                    "average_time": 0.0,
                    "spaCy_calls": 0,
                    "pattern_only_calls": 0
                }
            
            import statistics
            return {
                "articles_processed": self.articles_processed,
                "average_time": statistics.mean(self.entity_extraction_times),
                "median_time": statistics.median(self.entity_extraction_times),
                "total_time": sum(self.entity_extraction_times),
                "spaCy_calls": self.spaCy_calls,
                "pattern_only_calls": self.pattern_only_calls,
                "hybrid_calls": self.hybrid_calls,
                "spaCy_usage_percent": (self.spaCy_calls / self.articles_processed * 100) if self.articles_processed > 0 else 0
            }
    
    def log_summary(self):
        """Log performance summary."""
        summary = self.get_summary()
        logger.info(
            f"Performance Metrics: "
            f"{summary['articles_processed']} articles, "
            f"avg {summary['average_time']:.3f}s/article, "
            f"{summary['spaCy_calls']} spaCy calls ({summary['spaCy_usage_percent']:.1f}%)"
        )


# Global metrics instance
_metrics = PerformanceMetrics()

def get_metrics() -> PerformanceMetrics:
    """Get global performance metrics instance."""
    return _metrics
