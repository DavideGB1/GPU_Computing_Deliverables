# Use:  make <path/matrix.mtx>
# Example:  make matrices/thermomech_dM.mtx

SHELL := /bin/bash
BUILD := build
BIN   := $(BUILD)/SpMV

.DEFAULT_GOAL := help
.PHONY: help clean FORCE

help:
	@echo "Use : make <path/matrix.mtx>"
	@echo "Example:  make matrices/thermomech_dM.mtx"
	@echo "      make clean  – remove directory build/"

clean:
	rm -rf $(BUILD)

FORCE:

%: FORCE
	module load CUDA/12.5.0 && \
	module load GCC/13.3.0  && \
	rm -rf $(BUILD)         && \
	cmake -B $(BUILD)       && \
	cmake --build $(BUILD)  && \
	./$(BIN) "$@"
