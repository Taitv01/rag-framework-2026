# Contributing to Ultimate RAG Framework

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing.

## How to Contribute

### 1. Fork the Repository

```bash
# Fork on GitHub, then clone
git clone https://github.com/yourusername/ultimate-rag.git
cd ultimate-rag
```

### 2. Create a Branch

```bash
# Create a feature branch
git checkout -b feature/your-feature-name

# Or a bug fix branch
git checkout -b fix/your-bug-fix
```

### 3. Make Changes

- Write clean, documented code
- Follow the existing code style
- Add tests for new features
- Update documentation if needed

### 4. Test Your Changes

```bash
# Run tests
pytest

# Run linting
black src/ tests/
isort src/ tests/
mypy src/
```

### 5. Commit Your Changes

```bash
# Stage changes
git add .

# Commit with a descriptive message
git commit -m "Add: your feature description"
```

### 6. Push and Create PR

```bash
# Push to your fork
git push origin feature/your-feature-name

# Create Pull Request on GitHub
```

## Code Style

### Python

- Follow PEP 8
- Use type hints
- Write docstrings for all public functions
- Keep functions small and focused

### Documentation

- Update README.md if needed
- Add docstrings to new functions
- Include examples in docstrings

### Testing

- Write unit tests for new features
- Ensure all tests pass
- Aim for good test coverage

## Types of Contributions

### Bug Fixes

- Fix bugs in existing code
- Improve error handling
- Fix documentation errors

### Features

- Add new RAG patterns
- Add new document loaders
- Add new vector store backends
- Add new evaluation metrics

### Documentation

- Improve existing documentation
- Add examples
- Fix typos

### Testing

- Add unit tests
- Add integration tests
- Improve test coverage

## Reporting Issues

### Bug Reports

Include:
- Steps to reproduce
- Expected behavior
- Actual behavior
- Environment details

### Feature Requests

Include:
- Use case description
- Proposed solution
- Alternatives considered

## Code of Conduct

- Be respectful
- Be constructive
- Be collaborative

## Questions?

Open an issue or contact the maintainers.

Thank you for contributing! 🎉
