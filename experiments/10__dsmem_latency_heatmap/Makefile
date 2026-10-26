EXP  := 10__dsmem_latency_heatmap
BINS ?= heatmap
include ../harness/Makefile.inc

# heatmap reuses 09's common09.hpp (placement records) and reads the 03 map (src/common10.hpp).
$(addprefix $(BIN_DIR)/,$(BINS)): src/common10.hpp ../09__dsmem_latency_vs_cluster_size/src/common09.hpp

# The dumps keep every instruction but drop the hex encodings (the trailing /* 0x… */ comments and encoding-only lines).
all: sass-slim
.PHONY: sass-slim quick
sass-slim: sass
	sed -i -E '/^[[:space:]]*\/\* 0x[0-9a-f]{16} \*\/[[:space:]]*$$/d; s#[[:space:]]*/\* 0x[0-9a-f]{16} \*/##' $(addprefix $(SASS_DIR)/,$(addsuffix .sass,$(BINS)))

# Full run (inside: tools/run --profile --lock-sm-mhz 1635 -- make -C experiments/10__dsmem_latency_heatmap run);
# a subset of the steps of src/run.sh with: make ... run STEPS="steer A ncu"
run: all
	bash src/run.sh $(STEPS)

# Smoke test: reduced schedules, appends with its own run_ids (cmdline contains --quick)
quick: all
	bash src/run.sh --quick $(STEPS)
