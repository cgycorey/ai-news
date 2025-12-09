# AI News Project - Agent Guidelines

## Build/Test Commands

```bash
# Install dependencies
uv sync

# Run all tests
uv run pytest

# Run single test file
uv run pytest tests/test_specific_file.py

# Run specific test function
uv run pytest tests/test_file.py::test_function_name

# Run with verbose output
uv run pytest -v

# Run with coverage
uv run pytest --cov=src --cov-report=html

# Lint code
uv run flake8 src/ tests/
uv run black src/ tests/
uv run isort src/ tests/

# Type checking
uv run mypy src/

# Build package
uv run python -m build
```

## Code Style Guidelines

### Python Style
- Use Python 3.10+ syntax and type hints
- Follow PEP 8 and PEP 257 for formatting and docstrings
- Maximum line length: 88 characters (Black default)
- Use f-strings for string formatting
- Import order: standard library, third-party, local imports (each separated by blank line)

### Type Hints
- Always use type hints for function signatures and class attributes
- Use `from typing import` for complex types
- Prefer specific types over generic `Any` when possible

### Naming Conventions
- Classes: `PascalCase`
- Functions and variables: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Private methods: `_leading_underscore`
- Files: `snake_case.py`

### Error Handling
- Use specific exception types, avoid bare `except:`
- Include meaningful error messages with context
- Use context managers (`with` statements) for resource management
- Log errors appropriately before re-raising if needed

### Documentation
- All modules, classes, and public functions need docstrings
- Use Google-style or NumPy-style docstring format
- Include examples in docstrings for complex functions
- Keep comments concise and focused on "why", not "what"

### Testing
- Write tests for all new functionality
- Use descriptive test names that explain what is being tested
- Follow Arrange-Act-Assert pattern
- Use fixtures for common test setup
- Mock external dependencies
- Test both success and failure cases

## Project Structure
```
src/
├── ai_news/
│   ├── __init__.py
│   ├── core.py
│   ├── config.py
│   └── utils.py
tests/
├── test_core.py
├── test_config.py
└── conftest.py
pyproject.toml
README.md
.env.example
```

## Environment Setup
- Copy `.env.example` to `.env` and configure API keys
- Use virtual environments or conda for dependency management
- Keep development dependencies separate from production dependencies

## Git Workflow
- Create feature branches from `main`
- Write clear, descriptive commit messages
- Ensure all tests pass before committing
- Use conventional commits format (feat:, fix:, docs:, etc.)