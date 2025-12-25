#!/bin/bash
# Websearch CLI Examples - Quick Reference
# Make sure to run from the ai_news directory

# Example 1: Single Topic - Healthcare AI
echo "Example 1: Single Topic - Healthcare AI"
uv run python -m ai_news.cli websearch "healthcare" --limit 10 --save

# Example 2: Two Topics - Finance + Blockchain
echo "Example 2: Two Topics - Finance + Blockchain"
uv run python -m ai_news.cli websearch "finance" "blockchain" --limit 10 --save

# Example 3: Three Topics - Edge Computing Ecosystem
echo "Example 3: Three Topics - Edge Computing Ecosystem"
uv run python -m ai_news.cli websearch "edge" "computing" "iot" --limit 5 --save

# Example 4: Research Topic - AI in Healthcare
echo "Example 4: Research Topic - AI in Healthcare"
uv run python -m ai_news.cli websearch "diagnosis" "treatment" "medical" --limit 5 --save

# Example 5: Industry Analysis - Manufacturing Automation
echo "Example 5: Industry Analysis - Manufacturing Automation"
uv run python -m ai_news.cli websearch "robotics" "automation" "manufacturing" --limit 5 --save

# Example 6: Technology Trends - Future Tech
echo "Example 6: Technology Trends - Future Tech"
uv run python -m ai_news.cli websearch "quantum" "computing" --limit 10 --save

# Example 7: Without Intersections (Faster)
echo "Example 7: Without Intersections (Faster)"
uv run python -m ai_news.cli websearch "security" "cloud" --limit 10 --no-intersections --save

# Example 8: Custom Confidence Threshold
echo "Example 8: Custom Confidence Threshold (Lower = More Results)"
uv run python -m ai_news.cli websearch "retail" "ecommerce" --limit 10 --min-confidence 0.15 --save

# Example 9: Max Intersection Size (Control Combination Depth)
echo "Example 9: Max Intersection Size = 2 (Pairwise Only)"
uv run python -m ai_news.cli websearch "ai" "ml" "dl" "data" --limit 5 --max-intersection-size 2 --save

# Example 10: Interactive Mode (Prompts Before Saving)
echo "Example 10: Interactive Mode (No --save Flag)"
uv run python -m ai_news.cli websearch "gaming" "vr" --limit 5
