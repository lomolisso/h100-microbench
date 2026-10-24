EXP  := 07__synchronization_primitives
BINS ?= sync_block sync_cluster sync_mbarrier sync_flags sync_grid
include ../harness/Makefile.inc

# Every binary shares src/sync_common.hpp and reads the 03 smid→GPC map through 04's gpc_map.hpp.
$(addprefix $(BIN_DIR)/,$(BINS)): src/sync_common.hpp ../04__memory_hierarchy_latency/src/gpc_map.hpp

# The dumps keep every instruction but drop the hex encodings (the trailing /* 0x… */ comments and encoding-only lines).
all: sass-slim
.PHONY: sass-slim
sass-slim: sass
	sed -i -E '/^[[:space:]]*\/\* 0x[0-9a-f]{16} \*\/[[:space:]]*$$/d; s#[[:space:]]*/\* 0x[0-9a-f]{16} \*/##' $(addprefix $(SASS_DIR)/,$(addsuffix .sass,$(BINS)))

# Full run (inside: tools/run --profile --lock-sm-mhz 1635 -- make -C experiments/07__synchronization_primitives run);
# a subset of the steps of src/run.sh with: make ... run STEPS="k1 ncu"
run: all
	bash src/run.sh $(STEPS)

# Smoke test: reduced sweeps, appends with its own run_ids (cmdline contains --quick)
quick: all
	bash src/run.sh --quick $(STEPS)
