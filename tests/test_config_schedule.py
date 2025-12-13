"""Tests for schedule configuration functionality."""

import pytest
from dataclasses import dataclass, asdict
from typing import Optional
from pathlib import Path
import tempfile
import json

from ai_news.config import Config, FeedConfig


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
        assert schedule.enabled is True
        assert schedule.interval_minutes == 360
        assert schedule.timezone == "UTC"
        assert schedule.max_retries == 3
        assert schedule.retry_delay_minutes == 5

    def test_schedule_config_custom_values(self):
        """Test ScheduleConfig with custom values."""
        from ai_news.config import ScheduleConfig
        
        schedule = ScheduleConfig(
            enabled=False,
            interval_minutes=720,
            timezone="America/New_York",
            max_retries=5,
            retry_delay_minutes=10
        )
        
        assert schedule.enabled is False
        assert schedule.interval_minutes == 720
        assert schedule.timezone == "America/New_York"
        assert schedule.max_retries == 5
        assert schedule.retry_delay_minutes == 10

    def test_schedule_config_to_dict(self):
        """Test ScheduleConfig to_dict method."""
        from ai_news.config import ScheduleConfig
        
        schedule = ScheduleConfig(
            enabled=False,
            interval_minutes=120,
            timezone="Europe/London"
        )
        
        result = schedule.to_dict()
        expected = {
            "enabled": False,
            "interval_minutes": 120,
            "timezone": "Europe/London",
            "max_retries": 3,
            "retry_delay_minutes": 5
        }
        
        assert result == expected

    def test_schedule_config_from_dict(self):
        """Test ScheduleConfig from_dict class method."""
        from ai_news.config import ScheduleConfig
        
        data = {
            "enabled": True,
            "interval_minutes": 180,
            "timezone": "Asia/Tokyo",
            "max_retries": 2,
            "retry_delay_minutes": 15
        }
        
        schedule = ScheduleConfig.from_dict(data)
        
        assert schedule.enabled is True
        assert schedule.interval_minutes == 180
        assert schedule.timezone == "Asia/Tokyo"
        assert schedule.max_retries == 2
        assert schedule.retry_delay_minutes == 15


class TestConfigWithSchedule:
    """Test cases for Config class with schedule support."""

    def test_config_has_schedule_field(self):
        """Test that Config class has schedule field."""
        # Create config with schedule
        from ai_news.config import ScheduleConfig
        
        schedule = ScheduleConfig(enabled=False, interval_minutes=120)
        config = Config(schedule=schedule)
        
        assert hasattr(config, 'schedule')
        assert config.schedule is schedule
        assert config.schedule.enabled is False
        assert config.schedule.interval_minutes == 120

    def test_config_default_schedule(self):
        """Test Config gets default schedule when none provided."""
        config = Config()
        
        assert hasattr(config, 'schedule')
        assert config.schedule is not None
        assert config.schedule.enabled is True
        assert config.schedule.interval_minutes == 360

    def test_config_to_dict_includes_schedule(self):
        """Test Config.to_dict includes schedule information."""
        from ai_news.config import ScheduleConfig
        
        schedule = ScheduleConfig(enabled=False, interval_minutes=240)
        config = Config(
            database_path="test.db",
            max_articles_per_feed=25,
            collection_interval_hours=3,
            schedule=schedule
        )
        
        result = config.to_dict()
        
        assert "schedule" in result
        assert result["schedule"]["enabled"] is False
        assert result["schedule"]["interval_minutes"] == 240

    def test_config_load_includes_schedule(self):
        """Test Config.load includes schedule from JSON."""
        config_data = {
            "database_path": "test.db",
            "feeds": [],
            "max_articles_per_feed": 30,
            "collection_interval_hours": 12,
            "schedule": {
                "enabled": False,
                "interval_minutes": 480,
                "timezone": "Europe/Paris",
                "max_retries": 5,
                "retry_delay_minutes": 20
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_path = Path(f.name)
        
        try:
            config = Config.load(config_path)
            
            assert hasattr(config, 'schedule')
            assert config.schedule.enabled is False
            assert config.schedule.interval_minutes == 480
            assert config.schedule.timezone == "Europe/Paris"
            assert config.schedule.max_retries == 5
            assert config.schedule.retry_delay_minutes == 20
        finally:
            config_path.unlink()

    def test_config_save_includes_schedule(self):
        """Test Config.save includes schedule in JSON."""
        from ai_news.config import ScheduleConfig
        
        schedule = ScheduleConfig(enabled=True, interval_minutes=60, timezone="UTC")
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
            assert saved_data["schedule"]["interval_minutes"] == 60
            assert saved_data["schedule"]["timezone"] == "UTC"
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
            assert config.schedule.enabled is True  # Default value
            assert config.schedule.interval_minutes == 360  # Default value
        finally:
            config_path.unlink()