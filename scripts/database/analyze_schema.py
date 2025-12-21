#!/usr/bin/env python3
"""
Analyze the database schema to understand the structure for diagnostics.
"""

import sqlite3
import json
from typing import Dict, List, Any

def analyze_database_schema(db_path: str = "ai_news.db") -> Dict[str, Any]:
    """Analyze database schema and content structure."""
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    analysis = {
        "tables": {},
        "sample_data": {},
        "content_stats": {}
    }
    
    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    
    print(f"Found tables: {tables}")
    
    for table in tables:
        cursor.execute(f"PRAGMA table_info({table})")
        columns = cursor.fetchall()
        analysis["tables"][table] = {
            "columns": [(col[1], col[2], col[3], col[5]) for col in columns],
            "column_count": len(columns)
        }
        
        # Get row count
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        row_count = cursor.fetchone()[0]
        analysis["tables"][table]["row_count"] = row_count
        
        print(f"\n{table}: {row_count} rows")
        for col in columns:
            print(f"  - {col[1]} ({col[2]})")
        
        # Get sample data
        if row_count > 0:
            cursor.execute(f"SELECT * FROM {table} LIMIT 3")
            sample_rows = cursor.fetchall()
            analysis["sample_data"][table] = sample_rows
            
            # Print sample data for key tables
            if table in ['articles', 'keywords', 'regions']:
                print(f"  Sample data:")
                for row in sample_rows:
                    print(f"    {row[:3]}..." if len(row) > 3 else f"    {row}")
    
    # Analyze content statistics
    if 'articles' in tables:
        # Basic stats
        cursor.execute("SELECT COUNT(*) FROM articles")
        total_articles = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT region) FROM articles")
        unique_regions = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT source_name) FROM articles")
        unique_sources = cursor.fetchone()[0]
        
        analysis["content_stats"] = {
            "total_articles": total_articles,
            "unique_regions": unique_regions,
            "unique_sources": unique_sources
        }
        
        print(f"\n📊 Content Statistics:")
        print(f"  Total articles: {total_articles}")
        print(f"  Unique regions: {unique_regions}")
        print(f"  Unique sources: {unique_sources}")
        
        # Region distribution
        cursor.execute("SELECT region, COUNT(*) FROM articles GROUP BY region")
        regions = cursor.fetchall()
        print(f"\n  Region distribution:")
        for region, count in regions:
            print(f"    {region}: {count}")
    
    conn.close()
    return analysis

if __name__ == "__main__":
    analysis = analyze_database_schema()
    
    # Save analysis to JSON for reference
    with open("db_schema_analysis.json", "w") as f:
        json.dump(analysis, f, indent=2, default=str)
    
    print(f"\n✅ Schema analysis saved to db_schema_analysis.json")