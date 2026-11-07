.PHONY: train serve export import pretrain handoff

#Self-play + training, resumes from the latest checkpoint
train:
	cd training && XLA_PYTHON_CLIENT_PREALLOCATE=false uv run python -m selfplay.selfplay

#Move API on :8000 (restart after training to load new weights)
serve:
	cd training && uv run fastapi run server.py

#Latest checkpoint -> chess.onnx
export:
	cd training && uv run python onnx_io.py export

#make import ONNX=path/to/file.onnx -> new checkpoint, then make train continues from it
import:
	@test -n "$(ONNX)" || (echo "usage: make import ONNX=path/to/file.onnx" && exit 1)
	cd training && uv run python onnx_io.py import $(abspath $(ONNX))

#make pretrain EVALS=path/to/lichess_db_eval.jsonl.zst -> checkpoints/supervised_<k>, resumes if run again
pretrain:
	@test -n "$(EVALS)" || (echo "usage: make pretrain EVALS=path/to/lichess_db_eval.jsonl.zst" && exit 1)
	cd training && XLA_PYTHON_CLIENT_PREALLOCATE=false uv run python -m supervised.train $(abspath $(EVALS))

#Latest supervised checkpoint -> next self-play checkpoint, then make train continues from it
handoff:
	cd training && uv run python -m supervised.handoff
