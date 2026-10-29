EXP  := 08__l2_partitions_and_first_touch
BINS := l2_probe l2_capacity
include ../harness/Makefile.inc

# Both binaries use 05's CleanL2 / partition prior (bw_common.hpp, which reads the 03 map through 04's gpc_map.hpp);
# l2_capacity also uses 04's chain generator.
$(addprefix $(BIN_DIR)/,$(BINS)): ../05__memory_hierarchy_bandwidth/src/bw_common.hpp ../04__memory_hierarchy_latency/src/gpc_map.hpp \
	../04__memory_hierarchy_latency/src/pchase_chain.hpp

# The dumps keep every instruction but drop the hex encodings (the trailing /* 0x… */ comments and encoding-only lines).
all: sass-slim
.PHONY: sass-slim quick
sass-slim: sass
	sed -i -E '/^[[:space:]]*\/\* 0x[0-9a-f]{16} \*\/[[:space:]]*$$/d; s#[[:space:]]*/\* 0x[0-9a-f]{16} \*/##' $(addprefix $(SASS_DIR)/,$(addsuffix .sass,$(BINS)))

# Full run (inside: h100 run --profile --lock-sm-mhz 1635 -- make -C experiments/08__l2_partitions_and_first_touch run);
# a subset of the steps of src/run.sh with: make ... run STEPS="k2 k5"
run: all
	bash src/run.sh $(STEPS)

# Smoke test: reduced sweeps, appends with its own run_ids (cmdline contains --quick)
quick: all
	bash src/run.sh --quick $(STEPS)
