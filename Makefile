.PHONY: dev backend frontend test up down

backend:
	cd backend && pip install -r requirements.txt && uvicorn app.main:app --reload

frontend:
	cd frontend && npm install && npm run dev

test:
	cd backend && python -m pytest -q

up:
	docker compose up --build

down:
	docker compose down
