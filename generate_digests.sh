#!/bin/bash
# Automatic news collection and digest generation

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Create digests directory if it doesn't exist
mkdir -p digests

echo "=== AI News Collector - Automatic Digest Generation ==="
echo "Starting at $(date)"

# Step 1: Collect news
echo "Step 1: Collecting news from RSS feeds..."
uv run python -m ai_news.cli collect

if [ $? -eq 0 ]; then
    echo "✅ News collection completed successfully"
else
    echo "❌ News collection failed"
    exit 1
fi

# Step 2: Generate daily AI digest
echo "Step 2: Generating daily AI digest..."
uv run python -m ai_news.cli digest --type daily --ai-only --save --output digests

if [ $? -eq 0 ]; then
    echo "✅ Daily digest generated successfully"
else
    echo "❌ Daily digest generation failed"
fi

# Step 3: Generate full daily digest
echo "Step 3: Generating full daily digest..."
uv run python -m ai_news.cli digest --type daily --save --output digests

if [ $? -eq 0 ]; then
    echo "✅ Full daily digest generated successfully"
else
    echo "❌ Full daily digest generation failed"
fi

# Step 4: Generate topic analysis for trending AI topics
echo "Step 4: Generating topic analyses..."

# Common AI topics to analyze
TOPICS=("OpenAI" "Anthropic" "Claude" "ChatGPT" "GPT" "machine learning" "AI" "LLM")

for topic in "${TOPICS[@]}"; do
    echo "  Analyzing topic: $topic"
    uv run python -m ai_news.cli digest --type topic --topic "$topic" --days 7 --save --output digests
done

echo "✅ Topic analyses completed"

# Step 5: Display statistics
echo "Step 5: Database statistics..."
uv run python -m ai_news.cli stats

# Step 6: List generated files
echo "Step 6: Generated files:"
ls -la digests/*.md | tail -10

echo ""
echo "=== Digest Generation Complete ==="
echo "Finished at $(date)"
echo ""
echo "To view the latest AI digest:"
echo "  cat digests/daily_digest_*.md | head -30"
echo ""
echo "To search for specific topics:"
echo "  uv run python -m ai_news.cli search \"your query\""