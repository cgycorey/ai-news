#!/usr/bin/env python3
"""
Migration script to convert flat feeds structure to regional structure.
"""

import json
import sys
from pathlib import Path


def migrate_config(config_path: Path):
    """Migrate flat feeds to regional structure."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    if 'regions' in config:
        print("✅ Configuration already has regional structure")
        return
    
    if 'feeds' not in config:
        print("❌ No feeds found in configuration")
        return
    
    # Backup original config FIRST
    backup_path = config_path.with_suffix('.json.backup')
    with open(backup_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Create regional structure
    feeds_list = config.pop('feeds', [])
    regions = {
        'global': {
            'name': 'Global',
            'feeds': feeds_list,
            'enabled': True
        }
    }
    
    # Add default empty regions
    for region_code in ['us', 'uk', 'eu', 'apac']:
        if region_code not in regions:
            regions[region_code] = {
                'name': {'us': 'United States', 'uk': 'United Kingdom', 'eu': 'European Union', 'apac': 'Asia-Pacific'}[region_code],
                'feeds': [],
                'enabled': True
            }
    
    config['regions'] = regions
    
    # Write migrated config
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Migration complete")
    print(f"📁 Original config backed up to: {backup_path}")
    print(f"🌍 Migrated {len(feeds_list)} feeds to 'global' region")


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python migrate_to_regions.py <config_path>")
        sys.exit(1)
    
    config_path = Path(sys.argv[1])
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        sys.exit(1)
    
    migrate_config(config_path)