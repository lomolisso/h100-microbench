EXP  := 11__dsmem_bandwidth_vs_cluster_size
BINS ?= dsmem_bw dsmem_l2mix
include ../harness/Makefile.inc

# both binaries reuse 09's common09.hpp (placement records, Grid09); dsmem_l2mix also 10's common10.hpp (SM map, TPC index).
$(addprefix $(BIN_DIR)/,$(BINS)): src/common11.hpp ../09__dsmem_latency_vs_cluster_size/src/common09.hpp
$(BIN_DIR)/dsmem_l2mix: ../10__dsmem_latency_heatmap/src/common10.hpp

# The dumps keep every instruction but drop the hex encodings (the trailing /* 0x… */ comments and encoding-only lines).
all: sass-slim
.PHONY: sass-slim quick
sass-slim: sass
	sed -i -E '/^[[:space:]]*\/\* 0x[0-9a-f]{16} \*\/[[:space:]]*$$/d; s#[[:space:]]*/\* 0x[0-9a-f]{16} \*/##' $(addprefix $(SASS_DIR)/,$(addsuffix .sass,$(BINS)))

# Full run (inside: tools/run --profile --lock-sm-mhz 1635 -- make -C experiments/11__dsmem_bandwidth_vs_cluster_size run);
# a subset of the steps of src/run.sh with: make ... run STEPS="A S ncu"
run: all
	bash src/run.sh $(STEPS)

# Smoke test: reduced sweeps, appends with its own run_ids (cmdline contains --quick)
quick: all
	bash src/run.sh --quick $(STEPS)
