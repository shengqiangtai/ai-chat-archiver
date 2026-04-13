.PHONY: setup run format-check lint-check clean

setup:
	cd server && python -m venv .venv && . .venv/bin/activate && python -m pip install --upgrade pip && pip install -r requirements.txt

run:
	cd server && . .venv/bin/activate && python main.py

format-check:
	@echo "No formatter configured yet."

lint-check:
	cd server && . .venv/bin/activate && python -m py_compile main.py models.py storage.py config.py

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
