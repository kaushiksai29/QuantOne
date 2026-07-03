.PHONY: test smoke manifest

test:
	pytest -q

# End-to-end CPU smoke: 1 model x 2 quants x 20 prompts x 1 seed (Phase 2).
smoke:
	python runner/run.py --model smoke --quant Q8_0 --seeds 1 --limit 20 --tasks tasks/tasks.jsonl
	python runner/run.py --model smoke --quant Q4_K_M --seeds 1 --limit 20 --tasks tasks/tasks.jsonl
	python manifest.py --model smoke --quants Q8_0,Q4_K_M --seeds 1 --limit 20

manifest:
	python manifest.py
