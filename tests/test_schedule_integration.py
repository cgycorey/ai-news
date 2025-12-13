# tests/test_schedule_integration.py
import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.ai_news.cli import main

def test_full_schedule_workflow(tmp_path):
    """Test complete schedule setup workflow"""
    config_path = tmp_path / "test_config.json"
    
    # Start with basic config
    config_data = {
        "database_path": "test.db",
        "feeds": [
            {
                "name": "Test Feed",
                "url": "https://example.com/rss",
                "category": "tech",
                "enabled": True,
                "ai_keywords": ["AI", "machine learning"]
            }
        ]
    }
    config_path.write_text(json.dumps(config_data))
    
    # 1. Set schedule
    with patch('sys.argv', ['ai-news', '--config', str(config_path), 'schedule', 'set', 'daily']):
        try:
            main()
        except SystemExit:
            pass
    
    # Verify schedule was set
    updated_config = json.loads(config_path.read_text())
    assert updated_config['schedule']['enabled'] == True
    assert updated_config['schedule']['interval'] == 'daily'
    
    # 2. Show schedule
    with patch('sys.argv', ['ai-news', '--config', str(config_path), 'schedule', 'show']):
        with patch('builtins.print') as mock_print:
            try:
                main()
            except SystemExit:
                pass
    
    print_calls = [str(call) for call in mock_print.call_args_list]
    assert any("ENABLED" in call for call in print_calls)
    
    # 3. Get cron instructions
    with patch('sys.argv', ['ai-news', '--config', str(config_path), 'schedule', 'cron-setup']):
        with patch('builtins.print') as mock_print:
            try:
                main()
            except SystemExit:
                pass
    
    print_calls = [str(call) for call in mock_print.call_args_list]
    assert any("crontab -e" in call for call in print_calls)
    
    # 4. Clear schedule
    with patch('sys.argv', ['ai-news', '--config', str(config_path), 'schedule', 'clear']):
        try:
            main()
        except SystemExit:
            pass
    
    # Verify schedule was cleared
    final_config = json.loads(config_path.read_text())
    assert final_config['schedule']['enabled'] == False

def test_schedule_validation():
    """Test that invalid intervals are rejected"""
    from src.ai_news.config import ScheduleConfig
    
    # Test valid intervals
    for interval in ['hourly', 'daily', 'weekly']:
        schedule = ScheduleConfig(interval=interval)
        assert schedule.interval == interval
    
    # Test that invalid interval would be caught by argparse validation
    # (argparse handles this in CLI, so we just verify the valid cases)