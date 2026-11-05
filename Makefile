.PHONY: train serve export import

#Self-play + training, resumes from the latest checkpoint
train:
	cd src && XLA_PYTHON_CLIENT_PREALLOCATE=false uv run python -m selfplay.selfplay

#Move API on :8000 (restart after training to load new weights)
serve:
	cd src && uv run fastapi run server.py

#Latest checkpoint -> chess.onnx
export:
	cd src && uv run python onnx_io.py export

#make import ONNX=path/to/file.onnx -> new checkpoint, then make train continues from it
import:
	@test -n "$(ONNX)" || (echo "usage: make import ONNX=path/to/file.onnx" && exit 1)
	cd src && uv run python onnx_io.py import $(abspath $(ONNX))
