# Makefile for CHESTXRAYPROJECT

# Install dependencies
install:
	pip install -r requirements.txt

# Run Jupyter Notebook
notebook:
	jupyter notebook main.ipynb


# Run tests with pytest
test:
	pytest -v

# Clean cache files
clean:
	rm -rf __pycache__ .pytest_cache tests/__pycache__
