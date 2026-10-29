EXP  := 12__dsmem_atomics_and_async_stores
BINS ?= atom_lat atom_tput dsmem_async
include ../harness/Makefile.inc

# every binary uses common12.hpp (which pulls in 04's chain, 05's CleanL2/BlockRec and 09's placement helpers) and the
# templated atomics of harness/atomics.cuh
$(addprefix $(BIN_DIR)/,$(BINS)): src/common12.hpp ../harness/atomics.cuh ../09__dsmem_latency_vs_cluster_size/src/common09.hpp \
	../05__memory_hierarchy_bandwidth/src/bw_common.hpp

# The dumps keep every instruction but drop the hex encodings (the trailing /* 0x… */ comments and encoding-only lines).
all: sass-slim
.PHONY: sass-slim quick
sass-slim: sass
	sed -i -E '/^[[:space:]]*\/\* 0x[0-9a-f]{16} \*\/[[:space:]]*$$/d; s#[[:space:]]*/\* 0x[0-9a-f]{16} \*/##' $(addprefix $(SASS_DIR)/,$(addsuffix .sass,$(BINS)))

# Full run (inside: h100 run --profile --lock-sm-mhz 1635 -- make -C experiments/12__dsmem_atomics_and_async_stores run);
# a subset of the steps of src/run.sh with: make ... run STEPS="k0a smem ncu"
run: all
	bash src/run.sh $(STEPS)

# Smoke test: reduced sweeps, appends with its own run_ids (cmdline contains --quick)
quick: all
	bash src/run.sh --quick $(STEPS)
