.PHONY: run check-camera benchmark-fps docker clean

run:
	python -m server.main

check-camera:
	python scripts/check_camera.py

benchmark-fps:
	python scripts/benchmark_fps.py

docker:
	docker compose up --build

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

