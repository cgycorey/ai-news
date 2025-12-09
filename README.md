# AI News Collector

A simple, free RSS-based AI news collector that aggregates AI/LLM-related news from multiple sources without requiring API keys.

## Features

- **RSS Feed Collection**: Aggregates news from multiple AI-focused RSS feeds
- **AI Relevance Detection**: Automatically identifies AI-related content using keyword matching
- **SQLite Storage**: Local database for persistent storage
- **Simple CLI**: Command-line interface for viewing and searching articles
- **No API Keys Required**: Uses only free RSS feeds and standard library

## Quick Start

1. **Install dependencies**:
   ```bash
   uv sync
   ```

2. **🚀 Get today's AI news (default behavior)**:
   ```bash
   uv run python -m ai_news.cli
   # or simply:
   ./ai-news
   ```

3. **Collect news first**:
   ```bash
   uv run python -m ai_news.cli collect
   ```

4. **List recent articles**:
   ```bash
   uv run python -m ai_news.cli list --limit 10
   ```

5. **Search articles**:
   ```bash
   uv run python -m ai_news.cli search "OpenAI"
   ```

## CLI Commands

### `default behavior` 🚀
**Generate today's AI news digest** (default when no command specified):
```bash
uv run python -m ai_news.cli
# or simply:
./ai-news
```

### `collect`
Collect news from all configured RSS feeds:
```bash
uv run python -m ai_news.cli collect
```

### `list`
List recent articles:
```bash
uv run python -m ai_news.cli list --limit 20 --ai-only
```
- `--limit`: Number of articles to show (default: 20)
- `--ai-only`: Show only AI-relevant articles

### `search`
Search articles by content:
```bash
uv run python -m ai_news.cli search "query" --limit 10
```

### `stats`
Show database statistics:
```bash
uv run python -m ai_news.cli stats
```

### `config`
Show current configuration:
```bash
uv run python -m ai_news.cli config
```

### `show`
Show full article details:
```bash
uv run python -m ai_news.cli show <article_id>
```

## Default News Sources

The collector comes pre-configured with these free RSS feeds:

1. **TechCrunch AI** - Latest AI news and startups
2. **MIT Technology Review AI** - AI research and analysis
3. **Ars Technica AI** - AI tech coverage
4. **Towards Data Science** - Data science and ML tutorials
5. **arXiv CS Machine Learning** - Latest research papers
6. **Hacker News** - Tech community discussions

## Configuration

Configuration is stored in `config.json`. You can modify:

- Feed URLs and categories
- AI relevance keywords
- Database location
- Collection settings

## AI Relevance Detection

Articles are automatically flagged as AI-relevant based on keywords:
- artificial intelligence, machine learning, deep learning
- neural network, LLM, GPT, ChatGPT, OpenAI, Anthropic
- AI, ML, AGI, transformer, BERT, NLP, computer vision

## Database Schema

- **Articles table**: Stores article content, metadata, and AI relevance
- **Indexes**: URL, publication date, and AI relevance for fast queries
- **SQLite**: Simple, file-based database

## Development

### Project Structure
```
src/ai_news/
├── __init__.py
├── cli.py          # Command-line interface
├── config.py       # Configuration management
├── collector.py    # RSS feed collection
└── database.py     # SQLite database operations
```

### Adding New RSS Feeds

1. Edit `config.json` or modify the `Config._create_default()` method
2. Add feed with name, URL, category, and custom AI keywords
3. Run `collect` to fetch from new sources

### Dependencies

- **Standard library only** for core functionality
- **Optional dependencies** for enhanced features:
  - `rich` for better CLI formatting
  - `click` for advanced CLI options
  - `feedparser` for robust RSS parsing
  - `httpx` for better HTTP handling
  - `beautifulsoup4` for HTML cleaning

## Future Enhancements

- [ ] Daily/weekly automated collection via cron
- [ ] Product idea generation using collected news
- [ ] Competitive analysis features
- [ ] Web interface for browsing articles
- [ ] Export functionality (JSON, CSV)
- [ ] Advanced NLP for better AI relevance detection

## License

MIT License - feel free to use and modify for your own AI news collection needs.