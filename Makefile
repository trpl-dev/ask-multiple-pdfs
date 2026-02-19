.PHONY: run install lint format test docker-build docker-up docker-down

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

docker-build:
	docker build -t ask-multiple-pdfs .

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down
