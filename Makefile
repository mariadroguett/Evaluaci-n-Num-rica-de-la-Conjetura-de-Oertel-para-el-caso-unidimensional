PY := $(shell if [ -x .venv/bin/python ]; then echo .venv/bin/python; else command -v python3 || command -v python; fi)

# Defaults (override like: make run-ortel D=3 Z="-1,0,1" PAR=1)
D ?= 2
Z ?= 0,1
N_PER_Z ?= 20
N_CP ?= 20
N_HIP ?= 50
N ?= 100000
INTO ?= 3000
PAR ?= 1
CHUNK ?= $(N)
WORKERS ?= 4
# Leave empty to disable; set to an integer to enable
MAXDRAWS ?=

RUN_FLAGS := --d $(D) --z_vals $(Z) --n_per_z $(N_PER_Z) --N_cp $(N_CP) --N_hip $(N_HIP) --N $(N) --into_baul $(INTO)
ifeq ($(PAR),1)
RUN_FLAGS += --use_parallel --chunk_N $(CHUNK) --max_workers $(WORKERS)
endif
ifneq ($(strip $(MAXDRAWS)),)
RUN_FLAGS += --max_draws $(MAXDRAWS)
endif

.PHONY: help venv deps run-ortel clean-venv

help:
	@echo "Targets:"
	@echo "  venv        - Create .venv virtualenv"
	@echo "  deps        - Install requirements into .venv"
	@echo "  run-ortel   - Run main.py with preset flags (override vars)"
	@echo "Variables (override like VAR=value):"
	@echo "  D, Z, N_PER_Z, N_CP, N_HIP, N, INTO, PAR, CHUNK, WORKERS, MAXDRAWS"
	@echo "Example parallel run: make deps run-ortel PAR=1 CHUNK=20000 WORKERS=8 INTO=5000"

venv:
	python3 -m venv .venv

deps: venv
	. .venv/bin/activate; pip install -U pip; pip install -r requirements.txt

run-ortel:
	$(PY) main.py $(RUN_FLAGS)

clean-venv:
	rm -rf .venv

