"""Simple configuration management for AI News."""

from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class ScheduleConfig:
    """Configuration for news collection scheduling."""
    enabled: bool = False
    interval: str = "daily"  # hourly, daily, weekly
    last_collection: Optional[str] = None
    next_collection: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ScheduleConfig':
        return cls(
            enabled=data.get('enabled', False),
            interval=data.get('interval', 'daily'),
            last_collection=data.get('last_collection'),
            next_collection=data.get('next_collection')
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'enabled': self.enabled,
            'interval': self.interval,
            'last_collection': self.last_collection,
            'next_collection': self.next_collection
        }


class FeedConfig:
    """Configuration for a single RSS feed."""
    
    def __init__(self, name: str, url: str, category: str = "general", 
                 enabled: bool = True, ai_keywords: List[str] | None = None):
        # Validate inputs
        if not name or not name.strip():
            raise ValueError("Feed name cannot be empty")
        if not url or not url.strip():
            raise ValueError("Feed URL cannot be empty")
        if not url.startswith(('http://', 'https://')):
            raise ValueError(f"Invalid URL format: {url}")
        
        self.name = name.strip()
        self.url = url.strip()
        self.category = category.strip().lower() if category else "general"
        self.enabled = enabled
        self.ai_keywords = ai_keywords or [
            "artificial intelligence", "machine learning", "deep learning",
            "neural network", "LLM", "GPT", "ChatGPT", "OpenAI", "Anthropic",
            "AI", "ML", "AGI", "transformer", "BERT", "NLP", "computer vision"
        ]
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "url": self.url,
            "category": self.category,
            "enabled": self.enabled,
            "ai_keywords": self.ai_keywords
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "FeedConfig":
        try:
            return cls(**data)
        except (TypeError, ValueError) as e:
            logger.error(f"Invalid feed configuration: {e}")
            raise ValueError(f"Invalid feed configuration: {e}")


@dataclass
class RegionConfig:
    """Configuration for a regional feed group."""
    name: str
    feeds: List[FeedConfig] = field(default_factory=list)
    enabled: bool = True
    collection_priority: int = 1  # 1=high, 2=medium, 3=low
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RegionConfig':
        feeds = [FeedConfig.from_dict(feed_data) for feed_data in data.get('feeds', [])]
        return cls(
            name=data.get('name', ''),
            feeds=feeds,
            enabled=data.get('enabled', True),
            collection_priority=data.get('collection_priority', 1)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'feeds': [feed.to_dict() for feed in self.feeds],
            'enabled': self.enabled,
            'collection_priority': self.collection_priority
        }


@dataclass
class Config:
    """Main application configuration."""
    database_path: str = "data/production/ai_news.db"
    regions: Dict[str, RegionConfig] = field(default_factory=dict)
    feeds: List[FeedConfig] = field(default_factory=list)  # Backward compatibility
    max_articles_per_feed: int = 50
    collection_interval_hours: int = 6
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    config_path: Optional[Path] = field(default=None, init=False)
    
    def __post_init__(self):
        # Initialize default regions if empty
        if not self.regions:
            self.regions = {
                'us': RegionConfig(name='United States'),
                'uk': RegionConfig(name='United Kingdom'), 
                'eu': RegionConfig(name='European Union'),
                'apac': RegionConfig(name='Asia-Pacific')
            }
        
        # If feeds provided via old interface, migrate to global region
        if self.feeds:
            self.regions['global'] = RegionConfig(name='Global', feeds=self.feeds)
        else:
            # Sync feeds field with regions
            self.feeds = self.get_all_feeds()
        
        # Validate inputs
        if self.max_articles_per_feed <= 0:
            raise ValueError("max_articles_per_feed must be positive")
        if self.collection_interval_hours <= 0:
            raise ValueError("collection_interval_hours must be positive")
    
    @property
    def all_feeds(self) -> List[FeedConfig]:
        """Get all feeds from all regions (backward compatibility)."""
        return self.get_all_feeds()
    
    def get_all_feeds(self) -> List[FeedConfig]:
        """Get all feeds from all regions."""
        all_feeds = []
        for region in self.regions.values():
            all_feeds.extend(region.feeds)
        return all_feeds
    
    def get_feeds_by_region(self, region: str) -> List[FeedConfig]:
        """Get feeds for specific region."""
        if region.lower() in self.regions:
            return self.regions[region.lower()].feeds
        return []
    
    def get_enabled_regions(self) -> List[str]:
        """Get list of enabled region codes."""
        return [code for code, region in self.regions.items() if region.enabled]

    def to_dict(self) -> Dict[str, Any]:
        """Convert Config to dictionary."""
        result = {
            "database_path": self.database_path,
            "regions": {code: region.to_dict() for code, region in self.regions.items()},
            "max_articles_per_feed": self.max_articles_per_feed,
            "collection_interval_hours": self.collection_interval_hours,
        }
        
        # Include feeds for backward compatibility
        all_feeds = self.get_all_feeds()
        if all_feeds:
            result["feeds"] = [feed.to_dict() for feed in all_feeds]
        
        if self.schedule:
            result['schedule'] = self.schedule.to_dict()
        return result

    @classmethod
    def load(cls, config_path: Path) -> 'Config':
        """Load configuration from JSON file."""
        if not config_path.exists():
            # Create default config
            default_config = cls._create_default()
            default_config.save(config_path)
            return default_config
        
        try:
            with open(config_path, 'r') as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Error loading config from {config_path}: {e}")
            logger.info("Creating default configuration instead")
            default_config = cls._create_default()
            default_config.save(config_path)
            return default_config
        
        # Handle backward compatibility
        regions_data = data.pop('regions', {})
        
        # Migrate old feeds structure to 'global' region
        if 'feeds' in data:
            regions_data['global'] = {
                'name': 'Global',
                'feeds': data.pop('feeds'),
                'enabled': True
            }
        
        # Load regions
        regions = {}
        for region_code, region_data in regions_data.items():
            regions[region_code] = RegionConfig.from_dict(region_data)
        
        schedule_data = data.get('schedule', {})
        schedule = ScheduleConfig.from_dict(schedule_data)
        
        config = cls(
            database_path=data.get('database_path', 'data/production/ai_news.db'),
            regions=regions,
            max_articles_per_feed=data.get('max_articles_per_feed', 50),
            collection_interval_hours=data.get('collection_interval_hours', 6),
            schedule=schedule
        )
        config.config_path = config_path
        
        # Update feeds field for backward compatibility
        config.feeds = config.get_all_feeds()
        
        return config

    def save(self, config_path: Optional[Path] = None):
        """Save configuration to JSON file."""
        path = config_path or self.config_path
        if not path:
            raise ValueError("No config path specified")
        
        config_data = self.to_dict()
        
        # Create directory if it doesn't exist
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(config_data, f, indent=2, default=str)

    @staticmethod
    def _create_default() -> 'Config':
        """Create default configuration with sample feeds."""
        default_feeds = [
            FeedConfig(
                name="TechCrunch AI",
                url="https://techcrunch.com/category/artificial-intelligence/feed/",
                category="tech"
            ),
            FeedConfig(
                name="MIT Technology Review AI",
                url="https://www.technologyreview.com/topic/artificial-intelligence/feed/",
                category="research"
            ),
            FeedConfig(
                name="Ars Technica AI",
                url="https://arstechnica.com/tag/artificial-intelligence/feed/",
                category="tech"
            ),
            FeedConfig(
                name="Towards Data Science",
                url="https://towardsdatascience.com/feed",
                category="tutorial"
            ),
            FeedConfig(
                name="arXiv Computer Science - Machine Learning",
                url="http://rss.arxiv.org/rss/cs.LG",
                category="research",
                ai_keywords=["machine learning", "neural network", "deep learning", "algorithm"]
            ),
            FeedConfig(
                name="Hacker News",
                url="https://hnrss.org/frontpage",
                category="general",
                ai_keywords=["AI", "artificial intelligence", "machine learning", "OpenAI", "GPT"]
            ),
        ]
        
        # Create regions with default feeds in global region
        regions = {
            'global': RegionConfig(name='Global', feeds=default_feeds),
            'us': RegionConfig(name='United States'),
            'uk': RegionConfig(name='United Kingdom'), 
            'eu': RegionConfig(name='European Union'),
            'apac': RegionConfig(name='Asia-Pacific')
        }
        
        config = Config(regions=regions)
        config.feeds = config.get_all_feeds()
        return config