# restaurant-visitor-eda

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

exploratory data analysis for the restaurant visitors data set for epam mentoring task

## Getting Started

This project uses `make` to automate common tasks and `uv` for lightning-fast dependency management.

### Prerequisites
Make sure you have [uv](https://github.com/astral-sh/uv) installed on your system.

### 1. Set up the Environment
To create a new virtual environment with Python 3.10, run:
```bash
make create_environment
```
*After creation, activate it using the instructions printed in your terminal (e.g., `source ./.venv/bin/activate` for Unix/macOS or `.\.venv\Scripts\activate` for Windows).*

### 2. Install Dependencies
Once the virtual environment is active, install all required packages:
```bash
make requirements
```

### 3. Run Exploratory Data Analysis
All EDA processes and plots are executed inside Jupyter Notebooks located in the `notebooks/` directory. If you want to run the raw data processing script, use:
```bash
make data
```

### Code Quality & Linting
This project enforces code quality using `ruff`.

* **Format code:** Auto-fix errors and format files. Run this before committing your code:
  ```bash
  make format
  ```
* **Lint code:** Check for style guide violations without modifying the files:
  ```bash
  make lint
  ```

### Utility Commands
* **Clean up:** Delete all compiled Python files (`.pyc` and `__pycache__`):
  ```bash
  make clean
  ```
* **Help:** List all available make commands:
  ```bash
  make help
  ```

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         restaurant_visitor_eda and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── restaurant_visitor_eda   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes restaurant_visitor_eda a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

