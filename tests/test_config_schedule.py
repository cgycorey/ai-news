"""Tests for schedule configuration functionality."""

import pytest
from dataclasses import dataclass, asdict
from typing import Optional
from pathlib import Path
import tempfile
import json

from src.ai_news.config import Config, FeedConfig


class TestScheduleConfig:
    """Test cases for ScheduleConfig dataclass."""

    def test_schedule_config_dataclass_exists(self):
        """Test that ScheduleConfig dataclass exists and can be imported."""
        # This test should fail initially because ScheduleConfig doesn't exist yet
        from ai_news.config import ScheduleConfig
        
        # Test that it's a dataclass
        assert hasattr(ScheduleConfig, '__dataclass_fields__')
        
        # Test default values
        schedule = ScheduleConfig()
        assert schedule.enabled is False
        assert schedule.interval == "daily"
        assert schedule.last_collection is None
        assert schedule.next_collection is None

    def test_schedule_config_custom_values(self):
        """Test ScheduleConfig with custom values."""
        from ai_news.config import ScheduleConfig
        
        schedule = ScheduleConfig(
            enabled=True,
            interval="hourly",
            last_collection="2025-12-13T10:30:00Z",
            next_collection="2025-12-13T11:30:00Z"
        )
        
        assert schedule.enabled is True
        assert schedule.interval == "hourly"
        assert schedule.last_collection == "2025-12-13T10:30:00Z"
        assert schedule.next_collection == "2025-12-13T11:30:00Z"

    def test_schedule_config_to_dict(self):
        """Test ScheduleConfig to_dict method."""
        from ai_news.config import ScheduleConfig
        
        schedule = ScheduleConfig(
            enabled=True,
            interval="weekly",
            last_collection="2025-12-12T10:30:00Z",
            next_collection="2025-12-19T10:30:00Z"
        )
        
        result = schedule.to_dict()
        expected = {
            "enabled": True,
            "interval": "weekly",
            "last_collection": "2025-12-12T10:30:00Z",
            "next_collection": "2025-12-19T10:30:00Z"
        }
        
        assert result == expected

    def test_schedule_config_from_dict(self):
        """Test ScheduleConfig from_dict class method."""
        from ai_news.config import ScheduleConfig
        
        data = {
            "enabled": False,
            "interval": "daily",
            "last_collection": "2025-12-13T09:00:00Z",
            "next_collection": "2025-12-14T09:00:00Z"
        }
        
        schedule = ScheduleConfig.from_dict(data)
        
        assert schedule.enabled is False
        assert schedule.interval == "daily"
        assert schedule.last_collection == "2025-12-13T09:00:00Z"
        assert schedule.next_collection == "2025-12-14T09:00:00Z"


class TestConfigWithSchedule:
    """Test cases for Config class with schedule support."""

    def test_config_has_schedule_field(self):
        """Test that Config class has schedule field."""
        # Create config with schedule
        from src.ai_news.config import ScheduleConfig
        
        schedule = ScheduleConfig(enabled=True, interval="hourly")
        config = Config(schedule=schedule)
        
        assert hasattr(config, 'schedule')
        assert config.schedule is schedule
        assert config.schedule.enabled is True
        assert config.schedule.interval == "hourly"

    def test_config_default_schedule(self):
        """Test Config gets default schedule when none provided."""
        config = Config()
        
        assert hasattr(config, 'schedule')
        assert config.schedule is not None
        assert config.schedule.enabled is False
        assert config.schedule.interval == "daily"

    def test_config_to_dict_includes_schedule(self):
        """Test Config.to_dict includes schedule information."""
        from src.ai_news.config import ScheduleConfig
        
        schedule = ScheduleConfig(enabled=True, interval="weekly")
        config = Config(
            database_path="test.db",
            max_articles_per_feed=25,
            collection_interval_hours=3,
            schedule=schedule
        )
        
        result = config.to_dict()
        
        assert "schedule" in result
        assert result["schedule"]["enabled"] is True
        assert result["schedule"]["interval"] == "weekly"

    def test_config_load_includes_schedule(self):
        """Test Config.load includes schedule from JSON."""
        config_data = {
            "database_path": "test.db",
            "feeds": [],
            "max_articles_per_feed": 30,
            "collection_interval_hours": 12,
            "schedule": {
                "enabled": True,
                "interval": "hourly",
                "last_collection": "2025-12-13T08:00:00Z",
                "next_collection": "2025-12-13T09:00:00Z"
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_path = Path(f.name)
        
        try:
            config = Config.load(config_path)
            
            assert hasattr(config, 'schedule')
            assert config.schedule.enabled is True
            assert config.schedule.interval == "hourly"
            assert config.schedule.last_collection == "2025-12-13T08:00:00Z"
            assert config.schedule.next_collection == "2025-12-13T09:00:00Z"
        finally:
            config_path.unlink()

    def test_config_save_includes_schedule(self):
        """Test Config.save includes schedule in JSON."""
        from src.ai_news.config import ScheduleConfig
        
        schedule = ScheduleConfig(enabled=True, interval="daily", last_collection="2025-12-13T10:00:00Z")
        config = Config(schedule=schedule)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_path = Path(f.name)
        
        try:
            config.save(config_path)
            
            # Load and verify the saved file
            with open(config_path, 'r') as f:
                saved_data = json.load(f)
            
            assert "schedule" in saved_data
            assert saved_data["schedule"]["enabled"] is True
            assert saved_data["schedule"]["interval"] == "daily"
            assert saved_data["schedule"]["last_collection"] == "2025-12-13T10:00:00Z"
        finally:
            config_path.unlink()

    def test_config_load_missing_schedule_uses_default(self):
        """Test Config.load creates default schedule when missing from JSON."""
        config_data = {
            "database_path": "test.db",
            "feeds": [],
            "max_articles_per_feed": 20,
            "collection_interval_hours": 6
            # No schedule field
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_path = Path(f.name)
        
        try:
            config = Config.load(config_path)
            
            assert hasattr(config, 'schedule')
            assert config.schedule is not None
            assert config.schedule.enabled is False  # Default value
            assert config.schedule.interval == "daily"  # Default value
        finally:
            config_path.unlink()