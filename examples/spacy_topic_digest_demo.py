#!/usr/bin/env python3
"""
Demo script showing the new spaCy-powered topic digest generation.

This demonstrates the enhanced MarkdownGenerator with relevance grouping.
"""

import sys
from datetime import datetime
from unittest.mock import Mock

sys.path.insert(0, 'src')

from ai_news.markdown_generator import MarkdownGenerator
from ai_news.spacy_digest_analyzer import ScoredArticle


def main():
    """Demonstrate spaCy-powered digest generation."""

    print("=" * 70)
    print("spaCy-Powered Topic Digest Generation Demo")
    print("=" * 70)
    print()

    # Create a mock database (not used for this demo)
    mock_db = Mock()
    generator = MarkdownGenerator(mock_db)

    # Create example scored articles with varying confidence levels
    topics = ["AI", "education"]
    days = 7

    scored_articles = [
        # Strong Match examples
        ScoredArticle(
            article={
                "id": 1,
                "title": "How AI is Transforming Education Systems Worldwide",
                "url": "https://example.com/ai-education",
                "content": "Artificial intelligence is revolutionizing classrooms by providing personalized learning experiences...",
                "summary": "AI brings personalized learning to education",
                "source_name": "Tech Education Daily",
                "author": "Dr. Sarah Chen",
                "published_at": datetime(2025, 12, 25, 10, 30),
                "category": "education"
            },
            confidence=0.92,
            matched_entities={"AI", "education", "personalized learning"}
        ),
        ScoredArticle(
            article={
                "id": 2,
                "title": "Machine Learning Tools for K-12 Classrooms",
                "url": "https://example.com/ml-k12",
                "content": "New ML tools help teachers create adaptive curriculum...",
                "summary": "ML tools adapt to student needs",
                "source_name": "EduTech Review",
                "author": "Mark Johnson",
                "published_at": datetime(2025, 12, 24, 14, 20),
                "category": "education"
            },
            confidence=0.88,
            matched_entities={"machine learning", "classrooms", "education"}
        ),

        # Moderate Match examples
        ScoredArticle(
            article={
                "id": 3,
                "title": "Technology Trends in Modern Schools",
                "url": "https://example.com/tech-schools",
                "content": "Schools are adopting various technologies for administration...",
                "summary": "Technology adoption in education administration",
                "source_name": "School News Network",
                "author": "Lisa Park",
                "published_at": datetime(2025, 12, 23, 9, 15),
                "category": "education"
            },
            confidence=0.78,
            matched_entities={"technology", "schools"}
        ),
        ScoredArticle(
            article={
                "id": 4,
                "title": "Digital Transformation in Higher Education",
                "url": "https://example.com/digital-edu",
                "content": "Universities are undergoing digital transformation...",
                "summary": "Digital tools in universities",
                "source_name": "University News",
                "author": "Prof. James Wilson",
                "published_at": datetime(2025, 12, 22, 16, 45),
                "category": "education"
            },
            confidence=0.72,
            matched_entities={"digital", "education"}
        ),

        # Related examples
        ScoredArticle(
            article={
                "id": 5,
                "title": "EdTech Startup Raises $10M Funding",
                "url": "https://example.com/edtech-funding",
                "content": "An educational technology company announced funding...",
                "summary": "EdTech company secures investment",
                "source_name": "TechCrunch",
                "author": "Mike Smith",
                "published_at": datetime(2025, 12, 21, 11, 0),
                "category": "business"
            },
            confidence=0.65,
            matched_entities={"EdTech", "funding"}
        ),
    ]

    # Generate the digest
    print("Generating spaCy-powered digest...")
    print(f"Topics: {', '.join(topics)}")
    print(f"Days: {days}")
    print(f"Total Articles: {len(scored_articles)}")
    print()

    digest = generator.generate_spacy_topic_digest(topics, scored_articles, days)

    print("=" * 70)
    print("GENERATED DIGEST:")
    print("=" * 70)
    print()
    print(digest)

    print()
    print("=" * 70)
    print("Key Features Demonstrated:")
    print("=" * 70)
    print("✓ Articles grouped by confidence levels (Strong/Moderate/Related)")
    print("✓ Chronological sorting within each group (newest first)")
    print("✓ Confidence scores displayed as percentages")
    print("✓ Matched entities shown for each article")
    print("✓ Clean, readable markdown format")
    print("✓ Statistics section with processing time")
    print()


if __name__ == "__main__":
    main()
