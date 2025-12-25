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
class TopicConfig:
    """Configuration for a single topic with dynamic discovery."""
    name: str
    keywords: List[str] = field(default_factory=list)
    auto_discover: bool = True
    min_confidence: float = 0.3
    max_keywords: int = 50
    discovered_keywords: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "keywords": self.keywords,
            "auto_discover": self.auto_discover,
            "min_confidence": self.min_confidence,
            "max_keywords": self.max_keywords,
            "discovered_keywords": self.discovered_keywords
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TopicConfig':
        return cls(
            name=data.get('name', ''),
            keywords=data.get('keywords', []),
            auto_discover=data.get('auto_discover', True),
            min_confidence=data.get('min_confidence', 0.3),
            max_keywords=data.get('max_keywords', 50),
            discovered_keywords=data.get('discovered_keywords', [])
        )


@dataclass
class DiscoveryConfig:
    """Configuration for topic discovery system."""
    enabled: bool = True
    min_occurrence: int = 3
    prune_days: int = 30
    min_confidence: float = 0.2

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "min_occurrence": self.min_occurrence,
            "prune_days": self.prune_days,
            "min_confidence": self.min_confidence
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DiscoveryConfig':
        return cls(
            enabled=data.get('enabled', True),
            min_occurrence=data.get('min_occurrence', 3),
            prune_days=data.get('prune_days', 30),
            min_confidence=data.get('min_confidence', 0.2)
        )


@dataclass
class TopicCombinationConfig:
    """Configuration for topic intersections/combinations."""
    name: str
    topics: List[str] = field(default_factory=list)
    min_confidence: float = 0.3
    region: str = "global"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "topics": self.topics,
            "min_confidence": self.min_confidence,
            "region": self.region
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TopicCombinationConfig':
        return cls(
            name=data.get('name', ''),
            topics=data.get('topics', []),
            min_confidence=data.get('min_confidence', 0.3),
            region=data.get('region', 'global')
        )


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

    # New topic-related fields
    topics: Dict[str, TopicConfig] = field(default_factory=dict)
    topic_combinations: Dict[str, TopicCombinationConfig] = field(default_factory=dict)
    discovery: DiscoveryConfig = field(default_factory=DiscoveryConfig)
    
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
    
    def add_feed(self, region: str, name: str, url: str, category: str = "general", 
                 ai_keywords: List[str] | None = None) -> bool:
        """Add a new feed to a specific region."""
        try:
            # Validate region
            region_code = region.lower()
            if region_code not in self.regions:
                # Create region if it doesn't exist
                self.regions[region_code] = RegionConfig(name=region.title())
            
            # Check if feed already exists
            existing_feeds = self.get_feeds_by_region(region_code)
            for existing_feed in existing_feeds:
                if existing_feed.url == url or existing_feed.name == name:
                    return False  # Feed already exists
            
            # Create new feed
            new_feed = FeedConfig(
                name=name,
                url=url,
                category=category,
                enabled=True,
                ai_keywords=ai_keywords or []
            )
            
            # Add to region
            self.regions[region_code].feeds.append(new_feed)
            
            # Sync backward compatibility field
            self.feeds = self.get_all_feeds()
            
            # Save configuration
            if self.config_path:
                self.save(self.config_path)
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding feed {name}: {e}")
            return False
    
    def get_enabled_regions(self) -> List[str]:
        """Get list of enabled region codes."""
        return [code for code, region in self.regions.items() if region.enabled]

    def get_topic_keywords(self, topic_name: str, discovered_keywords: Optional[List[str]] = None) -> List[str]:
        """
        Get base keywords for a topic, optionally combined with discovered keywords.

        Args:
            topic_name: Name of the topic
            discovered_keywords: List of discovered keywords from database (optional)

        Returns:
            List of base + discovered keywords (deduplicated)
        """
        if topic_name not in self.topics:
            return []

        topic = self.topics[topic_name]
        base_keywords = topic.keywords.copy()

        # Add discovered keywords if provided
        if discovered_keywords:
            # Combine and deduplicate
            seen = {kw.lower() for kw in base_keywords}
            for kw in discovered_keywords:
                if kw.lower() not in seen:
                    base_keywords.append(kw)
                    seen.add(kw.lower())

        return base_keywords[:topic.max_keywords]

    def add_topic(self, name: str, keywords: List[str], auto_discover: bool = True) -> TopicConfig:
        """Add a new topic or update existing."""
        if name in self.topics:
            # Update existing
            self.topics[name].keywords.extend(keywords)
            self.topics[name].keywords = list(set(self.topics[name].keywords))
        else:
            # Create new
            self.topics[name] = TopicConfig(
                name=name,
                keywords=keywords,
                auto_discover=auto_discover
            )

        # Save if config path set
        if self.config_path:
            self.save(self.config_path)

        return self.topics[name]

    def remove_topic(self, name: str) -> bool:
        """Remove a topic."""
        if name in self.topics:
            del self.topics[name]

            # Also remove any combinations that use this topic
            to_remove = [
                combo_name for combo_name, combo in self.topic_combinations.items()
                if name in combo.topics
            ]
            for combo_name in to_remove:
                del self.topic_combinations[combo_name]

            # Save if config path set
            if self.config_path:
                self.save(self.config_path)

            return True
        return False

    def list_topics(self) -> List[str]:
        """Get list of all topic names."""
        return list(self.topics.keys())

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

        # Include topics configuration
        if self.topics:
            result["topics"] = {name: topic.to_dict() for name, topic in self.topics.items()}

        if self.topic_combinations:
            result["topic_combinations"] = {name: combo.to_dict() for name, combo in self.topic_combinations.items()}

        if self.discovery:
            result["discovery"] = self.discovery.to_dict()

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

        # Load topics
        topics_data = data.get('topics', {})
        topics = {}
        for topic_name, topic_data in topics_data.items():
            topics[topic_name] = TopicConfig.from_dict(topic_data)

        # Load topic combinations
        combinations_data = data.get('topic_combinations', {})
        combinations = {}
        for combo_name, combo_data in combinations_data.items():
            combinations[combo_name] = TopicCombinationConfig.from_dict(combo_data)

        # Load discovery config
        discovery_data = data.get('discovery', {})
        discovery = DiscoveryConfig.from_dict(discovery_data)

        config = cls(
            database_path=data.get('database_path', 'data/production/ai_news.db'),
            regions=regions,
            max_articles_per_feed=data.get('max_articles_per_feed', 50),
            collection_interval_hours=data.get('collection_interval_hours', 6),
            schedule=schedule,
            topics=topics,
            topic_combinations=combinations,
            discovery=discovery
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
        """Create default configuration with sample feeds and topics."""
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

        # Create default topics
        topics = {
            'AI': TopicConfig(
                name='AI',
                keywords=['AI', 'artificial intelligence', 'machine learning', 'deep learning', 'LLM', 'GPT'],
                auto_discover=True
            ),
            'Healthcare': TopicConfig(
                name='Healthcare',
                keywords=['healthcare', 'medical', 'clinical', 'hospital', 'patient care'],
                auto_discover=True
            ),
            'Finance': TopicConfig(
                name='Finance',
                keywords=['finance', 'banking', 'fintech', 'trading', 'investment'],
                auto_discover=True
            ),
            'Insurance': TopicConfig(
                name='Insurance',
                keywords=['insurance', 'insurtech', 'underwriting', 'claims', 'risk'],
                auto_discover=True
            )
        }

        # Create default topic combinations
        combinations = {
            'ai_insurance': TopicCombinationConfig(
                name='AI+Insurance',
                topics=['AI', 'Insurance'],
                min_confidence=0.3,
                region='global'
            ),
            'ai_healthcare': TopicCombinationConfig(
                name='AI+Healthcare',
                topics=['AI', 'Healthcare'],
                min_confidence=0.3,
                region='global'
            )
        }

        config = Config(
            regions=regions,
            topics=topics,
            topic_combinations=combinations,
            discovery=DiscoveryConfig()
        )
        config.feeds = config.get_all_feeds()
        return config