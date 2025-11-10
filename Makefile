HAS_CUDA := $(shell if command -v nvcc >/dev/null 2>&1; then echo 1; else echo 0; fi)

.PHONY: install
install: ## Setup the project
	@if [ $(HAS_CUDA) -eq 1 ]; then \
		echo "=== Install with GPU support ==="; \
		uv sync --all-groups --extra=gpu; \
	else \
		echo "=== Install with CPU support ==="; \
		uv sync --all-groups --extra=cpu; \
	fi

