EXP  := 09__dsmem_latency_vs_cluster_size
BINS ?= dsmem_lat dsmem_vis
include ../harness/Makefile.inc

# Both binaries share src/common09.hpp and read the 03 smid→GPC map through 04's gpc_map.hpp; dsmem_lat reuses 04's chain.
$(addprefix $(BIN_DIR)/,$(BINS)): src/common09.hpp ../04__memory_hierarchy_latency/src/gpc_map.hpp ../04__memory_hierarchy_latency/src/pchase_chain.hpp

# The dumps keep every instruction but drop the hex encodings (the trailing /* 0x… */ comments and encoding-only lines).
all: sass-slim
.PHONY: sass-slim quick
sass-slim: sass
	sed -i -E '/^[[:space:]]*\/\* 0x[0-9a-f]{16} \*\/[[:space:]]*$$/d; s#[[:space:]]*/\* 0x[0-9a-f]{16} \*/##' $(addprefix $(SASS_DIR)/,$(addsuffix .sass,$(BINS)))

# Full run (inside: h100 run --profile --lock-sm-mhz 1635 -- make -C experiments/09__dsmem_latency_vs_cluster_size run);
# a subset of the steps of src/run.sh with: make ... run STEPS="k1 ncu"
run: all
	bash src/run.sh $(STEPS)

# Smoke test: reduced sweeps, appends with its own run_ids (cmdline contains --quick)
quick: all
	bash src/run.sh --quick $(STEPS)
