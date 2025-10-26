HAS_CUDA := $(shell command -v nvcc 2> /dev/null && echo 1 || echo 0)

.PHONY: install
install: ## Setup the project
	@if [ $(HAS_CUDA) -eq 1 ]; then \
		echo "=== Install with GPU support ==="; \
		uv sync --all-groups --extra=gpu; \
	else \
		echo "=== Install with CPU support ==="; \
		uv sync --all-groups --extra=cpu; \
	fi

