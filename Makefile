.PHONY: run install lint format test

run:
	streamlit run app.py

install:
	pip install -r requirements.txt

lint:
	ruff check .

format:
	ruff format .

test:
	pytest tests/
