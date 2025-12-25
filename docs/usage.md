# AI News Usage Guide

This guide covers all the major features and commands of the AI News Intelligence System.

## Table of Contents

- [Quick Start](#quick-start)
- [Collecting News](#collecting-news)
- [Generating Digests](#generating-digestes)
- [Topic Discovery with spaCy](#topic-discovery-with-spacy)
- [Searching Articles](#searching-articles)
- [Managing Feeds](#managing-feeds)
- [System Statistics](#system-statistics)

## Quick Start

```bash
# Install dependencies
uv sync

# Download required NLP models
uv run ai-news setup-nltk
uv run ai-news setup-spacy

# Collect today's news
uv run ai-news collect --hours 24

# Generate a digest
uv run ai-news digest --topic "machine learning" --days 7
```

## Collecting News

### Basic Collection

```bash
# Collect last 24 hours
uv run ai-news collect --hours 24

# Collect last 7 days
uv run ai-news collect --days 7

# Collect from specific region
uv run ai-news collect --region uk
```

### Regional Collection

```bash
# Collect from UK only
uv run ai-news collect --region uk

# Collect from multiple regions
uv run ai-news collect --regions uk,eu

# Collect from all regions (default)
uv run ai-news collect
```

Supported regions: US, UK, EU, APAC, Global

## Generating Digests

### Topic-Based Digests

```bash
# Basic topic digest
uv run ai-news digest --topic "machine learning"

# With custom time range
uv run ai-news digest --topic "deep learning" --days 30

# Save to file
uv run ai-news digest --topic "computer vision" --save

# Custom filename
uv run ai-news digest --topic "neural networks" --output my_digest.md
```

### Time-Based Digests

```bash
# Daily digest
uv run ai-news digest --type daily

# Weekly digest
uv run ai-news digest --type weekly

# Custom date
uv run ai-news digest --type weekly --date 2024-12-10
```

## Topic Discovery with spaCy

The topic discovery system uses spaCy's Named Entity Recognition to extract high-quality technical terms from articles.

### Usage

```bash
# Discover keywords using spaCy (default)
uv run ai-news topics discover AI

# Use basic extraction (faster, less accurate)
uv run ai-news topics discover AI --no-spacy

# Set minimum relevance threshold
uv run ai-news topics discover AI --min-relevance 0.5

# Set minimum occurrence
uv run ai-news topics discover AI --min-occurrence 3
```

### What Gets Discovered

- **Entities:** Companies (OpenAI, Google, Microsoft), Products (GPT, BERT, Claude)
- **Technical Phrases:** neural networks, deep learning, computer vision, natural language processing
- **Domain Terms:** backpropagation, gradient descent, tokenization, transformers

### Key Improvements Over Basic Extraction

- ✅ **Zero common words** - "the", "and", "for" are completely filtered out
- ✅ **Named entities** - Companies and products are properly identified
- ✅ **Technical terms** - Multi-word technical concepts preserved
- ✅ **Domain relevance** - Terms scored by AI/tech specificity

### Setup

One-time spaCy model installation:

```bash
uv run python -m spacy download en_core_web_sm
```

Or use the built-in command:

```bash
uv run ai-news setup-spacy
```

### Discovery Examples

```bash
# Discover topics for machine learning
uv run ai-news topics discover "machine learning"

# View discovery statistics
uv run ai-news topics stats "machine learning"

# List discovered terms
uv run ai-news topics list "machine learning" --limit 20

# Export discoveries
uv run ai-news topics export "machine learning" --output ml_terms.json
```

### Topic Discovery Options

- `--use-spacy` - Enable spaCy extraction (default: True)
- `--no-spacy` - Disable spaCy, use basic extraction
- `--min-relevance` - Minimum domain relevance score (default: 0.3)
- `--min-occurrence` - Minimum times term must appear (default: 3)

## Searching Articles

```bash
# Search by keyword
uv run ai-news search "machine learning"

# Search with filters
uv run ai-news search "GPT" --days 7 --source techcrunch

# Recent articles
uv run ai-news search --recent 20

# By region
uv run ai-news search "fintech" --region uk
```

## Managing Feeds

```bash
# List all feeds
uv run ai-news feeds list

# List feeds by region
uv run ai-news feeds list --region uk

# Add new feed
uv run ai-news feeds add --name "AI Blog" --url "https://example.com/rss" --category ai --region global

# Remove feed
uv run ai-news feeds remove "AI Blog" --region global
```

## System Statistics

```bash
# Overall statistics
uv run ai-news stats

# Statistics by region
uv run ai-news stats --all-regions

# Database info
uv run ai-news stats --database
```

## NLP Setup

### NLTK Setup

```bash
# Download NLTK data
uv run ai-news setup-nltk

# Check NLTK status
uv run ai-news setup-nltk --check
```

### spaCy Setup

```bash
# Download spaCy models
uv run ai-news setup-spacy

# Verify installation
uv run python -c "import spacy; spacy.load('en_core_web_sm'); print('✅ Model loaded')"
```

## Scheduling

### Set Up Schedule

```bash
# Set daily collection
uv run ai-news schedule set daily

# Set hourly collection
uv run ai-news schedule set hourly

# Set weekly collection
uv run ai-news schedule set weekly

# Get cron instructions
uv run ai-news schedule cron-setup
```

### Manage Schedule

```bash
# Show current schedule
uv run ai-news schedule show

# Clear schedule
uv run ai-news schedule clear
```

## Regional News

### Collect by Region

```bash
# UK news
uv run ai-news collect --region uk

# Multiple regions
uv run ai-news collect --regions uk,eu,us

# All regions
uv run ai-news collect
```

### Filter by Region

```bash
# List UK articles
uv run ai-news list --region uk

# Search within region
uv run ai-news search "AI" --region uk --days 7

# Stats by region
uv run ai-news stats --all-regions
```

## Advanced Workflows

### Research Workflow

```bash
# 1. Collect latest research
uv run ai-news collect --source arxiv --days 7

# 2. Discover research topics
uv run ai-news topics discover "machine learning research"

# 3. Generate focused digest
uv run ai-news digest --topic "machine learning research" --days 7 --save research_digest.md
```

### Competitive Intelligence

```bash
# Monitor competitors
uv run ai-news digest --topic "OpenAI" --days 7 --save openai_intel.md
uv run ai-news digest --topic "Google" --days 7 --save google_intel.md
uv run ai-news digest --topic "Microsoft" --days 7 --save microsoft_intel.md
```

### Trend Analysis

```bash
# Daily digests for trend tracking
uv run ai-news digest --type daily --save daily_$(date +%Y%m%d).md

# Weekly deep dive
uv run ai-news digest --topic "your-topic" --days 7 --save weekly_$(date +%Y%m%d).md

# Monthly analysis
uv run ai-news digest --type weekly --days 30 --save monthly_trends.md
```

## Troubleshooting

### No articles found

```bash
# 1. Check collection status
uv run ai-news stats

# 2. Force fresh collection
uv run ai-news collect --hours 24

# 3. Try broader topic
uv run ai-news digest --topic "artificial intelligence" --days 30
```

### spaCy model not found

```bash
# Reinstall spaCy model
uv run python -m spacy download en_core_web_sm

# Or use built-in command
uv run ai-news setup-spacy
```

### Feed issues

```bash
# List feeds to check status
uv run ai-news feeds list

# Remove problematic feed
uv run ai-news feeds remove "Feed Name" --region global

# Add feed again
uv run ai-news feeds add --name "Feed Name" --url "https://..."
```

## Tips for Best Results

1. **Use specific topics** - "large language models" vs "AI"
2. **Adjust time windows** - Some topics need 30+ days
3. **Enable spaCy** - Get better topic discovery results
4. **Save regularly** - Keep digests for trend analysis
5. **Check stats** - Understand your database coverage
6. **Use regions** - Focus on geographic areas of interest

## More Information

- [README.md](../README.md) - Project overview and quick start
- [Architecture](../docs/architecture/) - System design details
- [Examples](../examples/) - Sample scripts and demos
