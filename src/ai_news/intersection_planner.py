"""
Intersection Planning Module for Multi-Topic Web Search

This module generates search plans for individual topics and their intersections,
enabling automatic collection of overlapping topic articles.
"""

from typing import List, Dict, Set
from itertools import combinations


def plan_topic_searches(
    topics: List[str],
    max_intersection_size: int = 3,
    min_intersection_size: int = 2
) -> List[Dict]:
    """
    Plan search queries for individual topics and their intersections.

    Args:
        topics: List of topic keywords (e.g., ['healthcare', 'finance', 'robotics'])
        max_intersection_size: Maximum number of topics in an intersection
        min_intersection_size: Minimum topics for intersection (default 2)

    Returns:
        List of search plans with:
        - search_type: 'individual' or 'intersection'
        - topics: List of topics in this search
        - query: Search query string
        - tags: List of tags for articles (e.g., ['AI', 'healthcare'])

    Examples:
        Input: ['healthcare', 'finance']
        Output: [
            {'search_type': 'individual', 'topics': ['healthcare'], 
             'query': 'AI healthcare', 'tags': ['AI', 'healthcare']},
            {'search_type': 'individual', 'topics': ['finance'], 
             'query': 'AI finance', 'tags': ['AI', 'finance']},
            {'search_type': 'intersection', 'topics': ['healthcare', 'finance'], 
             'query': 'AI healthcare AND finance', 'tags': ['AI', 'healthcare', 'finance']}
        ]
    """
    if not topics:
        return []

    search_plans = []

    # 1. Individual topic searches
    for topic in topics:
        search_plans.append({
            'search_type': 'individual',
            'topics': [topic],
            'query': f'AI {topic}',
            'tags': ['AI', topic]
        })

    # 2. Intersection combinations (pairwise, triple, etc.)
    # Only generate intersections if we have more than one topic
    if len(topics) >= min_intersection_size:
        for r in range(min_intersection_size, min(max_intersection_size + 1, len(topics) + 1)):
            for combo in combinations(topics, r):
                search_plans.append({
                    'search_type': 'intersection',
                    'topics': list(combo),
                    'query': f'AI {" AND ".join(combo)}',
                    'tags': ['AI'] + list(combo)
                })

    return search_plans


def format_search_summary(search_plans: List[Dict]) -> str:
    """
    Generate a human-readable summary of the search plan.

    Args:
        search_plans: List of search plans from plan_topic_searches()

    Returns:
        Formatted summary string
    """
    if not search_plans:
        return "No search plans generated."

    individual = [p for p in search_plans if p['search_type'] == 'individual']
    intersections = [p for p in search_plans if p['search_type'] == 'intersection']

    lines = [
        "═════════════════════════════════════════════════════",
        "              SEARCH PLAN SUMMARY",
        "═════════════════════════════════════════════════════",
        f"Total search plans: {len(search_plans)}",
        f"Individual topics: {len(individual)}",
        f"Intersection combinations: {len(intersections)}",
        "",
        "Individual topics:",
    ]

    for plan in individual:
        tags_str = ', '.join(plan['tags'])
        lines.append(f"  • [{tags_str}]")

    if intersections:
        lines.append("")
        lines.append("Intersection combinations:")
        for plan in intersections:
            tags_str = ', '.join(plan['tags'])
            lines.append(f"  • [{tags_str}]")

    lines.append("═════════════════════════════════════════════════════")

    return "\n".join(lines)


def get_unique_topics_from_plans(search_plans: List[Dict]) -> Set[str]:
    """
    Extract all unique topics from search plans.

    Args:
        search_plans: List of search plans

    Returns:
        Set of unique topic strings
    """
    unique_topics = set()
    for plan in search_plans:
        unique_topics.update(plan['topics'])
    return unique_topics


def validate_search_plan_coverage(topics: List[str], search_plans: List[Dict]) -> bool:
    """
    Validate that the search plans cover all requested topics.

    Args:
        topics: Original list of topics requested
        search_plans: Generated search plans

    Returns:
        True if all topics are covered
    """
    topic_set = set(topics)
    covered_topics = get_unique_topics_from_plans(search_plans)
    return topic_set.issubset(covered_topics)


def estimate_total_searches(topic_count: int, max_intersection_size: int = 3) -> Dict[str, int]:
    """
    Estimate the number of searches that will be performed.

    Args:
        topic_count: Number of topics to search
        max_intersection_size: Maximum intersection size

    Returns:
        Dictionary with breakdown of search counts
    """
    individual = topic_count
    intersections = 0

    if topic_count >= 2:
        # Calculate combinations for sizes 2 to max_intersection_size
        for r in range(2, min(max_intersection_size + 1, topic_count + 1)):
            from math import comb
            intersections += comb(topic_count, r)

    return {
        'individual': individual,
        'intersections': intersections,
        'total': individual + intersections
    }
