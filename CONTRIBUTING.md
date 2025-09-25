# Contributing to EV Charging Demand Forecasting System

Thank you for your interest in contributing to this project! This document provides guidelines and information for contributors.

## 🚀 Getting Started

### Prerequisites

- Python 3.10
- Git
- Docker (optional, for containerized development)

### Development Setup

1. **Fork and Clone the Repository**
   ```bash
   git clone https://github.com/adityagit94/EV-Charging-Demand-Forecasting-and-Simulator.git
   cd EV-Charging-Demand-Forecasting-and-Simulator
   ```

2. **Create a Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Development Dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -e .  # Install package in development mode
   ```

4. **Set Up Pre-commit Hooks**
   ```bash
   pre-commit install
   ```

5. **Run Tests to Verify Setup**
   ```bash
   pytest tests/ -v
   ```

## 📝 Development Workflow

### Branch Naming Convention

- `feature/feature-name` - New features
- `bugfix/issue-description` - Bug fixes
- `hotfix/critical-issue` - Critical fixes
- `docs/documentation-update` - Documentation updates
- `refactor/code-improvement` - Code refactoring

### Commit Message Format

Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
type(scope): description

[optional body]

[optional footer]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**
```
feat(api): add batch prediction endpoint
fix(dashboard): resolve chart rendering issue
docs(readme): update installation instructions
test(pipeline): add unit tests for data validation
```

### Pull Request Process

1. **Create a Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Your Changes**
   - Write clean, well-documented code
   - Add tests for new functionality
   - Update documentation as needed

3. **Run Quality Checks**
   ```bash
   # Code formatting
   black src/ tests/
   
   # Import sorting
   isort src/ tests/
   
   # Linting
   flake8 src/ tests/
   
   # Type checking
   mypy src/
   
   # Run tests
   pytest tests/ -v --cov=src
   ```

4. **Commit Your Changes**
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```

5. **Push to Your Fork**
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Create a Pull Request**
   - Use a clear, descriptive title
   - Fill out the PR template completely
   - Link any related issues
   - Request review from maintainers

## 🧪 Testing Guidelines

### Test Structure

```
tests/
├── unit/                 # Unit tests
│   ├── test_data_pipeline.py
│   ├── test_features.py
│   └── test_models.py
├── integration/          # Integration tests
│   ├── test_api_integration.py
│   └── test_training_pipeline.py
└── conftest.py          # Test configuration
```

### Writing Tests

- **Unit Tests**: Test individual functions/classes in isolation
- **Integration Tests**: Test component interactions
- **Use Fixtures**: Leverage pytest fixtures for test data
- **Mock External Dependencies**: Use `unittest.mock` for external services
- **Test Edge Cases**: Include tests for error conditions and edge cases

### Test Coverage

- Maintain minimum 80% test coverage
- Focus on critical business logic
- Test both happy path and error scenarios

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/unit/test_data_pipeline.py

# Run tests with specific marker
pytest -m unit
```

## 📊 Code Quality Standards

### Code Style

- Follow [PEP 8](https://pep8.org/) Python style guide
- Use [Black](https://black.readthedocs.io/) for code formatting
- Use [isort](https://pycqa.github.io/isort/) for import sorting
- Maximum line length: 88 characters

### Documentation

- **Docstrings**: Use Google-style docstrings for all public functions/classes
- **Type Hints**: Include type hints for function parameters and returns
- **Comments**: Explain complex logic, not obvious code
- **README Updates**: Update documentation for new features

### Example Function Documentation

```python
def process_charging_data(
    data: pd.DataFrame, 
    site_id: int,
    start_date: Optional[datetime] = None
) -> pd.DataFrame:
    """Process charging session data for a specific site.
    
    Args:
        data: Raw charging session data
        site_id: ID of the charging site to process
        start_date: Optional start date for filtering data
        
    Returns:
        Processed charging data with engineered features
        
    Raises:
        ValueError: If site_id is not found in data
        DataValidationError: If data fails validation checks
        
    Example:
        >>> data = load_charging_data("sessions.csv")
        >>> processed = process_charging_data(data, site_id=1)
        >>> print(processed.columns)
    """
```

## 🏗️ Architecture Guidelines

### Project Structure

Follow the established project structure:

```
src/
├── api/                 # FastAPI application
├── data_pipeline.py     # Data processing
├── features.py          # Feature engineering
├── models/              # ML models
├── training.py          # Training orchestration
└── utils/               # Utilities
```

### Design Principles

- **Single Responsibility**: Each class/function should have one clear purpose
- **Dependency Injection**: Use dependency injection for testability
- **Error Handling**: Implement comprehensive error handling
- **Logging**: Use structured logging throughout the application
- **Configuration**: Use configuration files for all settings

### Adding New Features

1. **API Endpoints**: Add new endpoints to `src/api/app.py`
2. **Models**: Create new model classes inheriting from `BaseModel`
3. **Features**: Add feature engineering functions to `src/features.py`
4. **Tests**: Include comprehensive tests for all new functionality

## 🐛 Bug Reports

### Before Submitting

1. Check existing issues to avoid duplicates
2. Verify the bug exists in the latest version
3. Try to reproduce with minimal example

### Bug Report Template

```markdown
**Bug Description**
A clear description of the bug.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. See error

**Expected Behavior**
What you expected to happen.

**Screenshots**
If applicable, add screenshots.

**Environment**
- OS: [e.g., Ubuntu 20.04]
- Python version: [e.g., 3.10.6]
- Package version: [e.g., 1.0.0]

**Additional Context**
Any other context about the problem.
```

## 💡 Feature Requests

### Feature Request Template

```markdown
**Feature Description**
A clear description of the feature you'd like to see.

**Problem Statement**
What problem does this feature solve?

**Proposed Solution**
Describe your proposed solution.

**Alternatives Considered**
Other solutions you've considered.

**Additional Context**
Any other context or screenshots.
```

## 🔄 Release Process

### Version Numbering

We follow [Semantic Versioning](https://semver.org/):
- **MAJOR**: Incompatible API changes
- **MINOR**: New functionality (backwards compatible)
- **PATCH**: Bug fixes (backwards compatible)

### Release Checklist

- [ ] Update version in `pyproject.toml`
- [ ] Update `CHANGELOG.md`
- [ ] Run full test suite
- [ ] Update documentation
- [ ] Create release PR
- [ ] Tag release after merge
- [ ] Update deployment environments

## 📞 Getting Help

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and discussions
- **Email**: aditya_2312res46@iitp.ac.in

### Code Review Process

1. All changes require code review
2. At least one approval from a maintainer
3. All CI checks must pass
4. No merge conflicts

### Recognition

Contributors will be:
- Listed in the project README
- Mentioned in release notes
- Invited to join the contributors team (for regular contributors)

## 📜 Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/). Please read and follow these guidelines to ensure a welcoming environment for all contributors.

## 🎉 Thank You!

Thank you for contributing to the EV Charging Demand Forecasting System! Your contributions help make sustainable transportation infrastructure more intelligent and efficient.

---

*This contributing guide is a living document. Please suggest improvements through pull requests or issues.*
