.PHONY: install run test lint clean

install:
	pip install -r requirements.txt

run:
	streamlit run app.py

test:
	python -m pytest tests/ -v

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -f data/tarifas-raw.csv
