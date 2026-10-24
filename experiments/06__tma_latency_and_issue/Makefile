EXP  := 06__tma_latency_and_issue
BINS := tma_chase tma_completion tma_issue tma_boxes
include ../harness/Makefile.inc

# The chase reuses 04's chain generator and the 03 smid→GPC map reader; HBM-source runs use 05's CleanL2.
$(addprefix $(BIN_DIR)/,$(BINS)): src/tma_common.hpp ../04__memory_hierarchy_latency/src/pchase_chain.hpp \
	../04__memory_hierarchy_latency/src/gpc_map.hpp ../05__memory_hierarchy_bandwidth/src/bw_common.hpp

# tma_issue holds one fully unrolled kernel per batch size and configuration (~200 kernels): its dump keeps every
# instruction but drops the hex encodings (the trailing /* 0x… */ comments and the encoding-only lines).
all: sass-slim
.PHONY: sass-slim
sass-slim: sass
	sed -i -E '/^[[:space:]]*\/\* 0x[0-9a-f]{16} \*\/[[:space:]]*$$/d; s#[[:space:]]*/\* 0x[0-9a-f]{16} \*/##' $(SASS_DIR)/tma_issue.sass

# Full run (inside: tools/run --profile --lock-sm-mhz 1635 -- make -C experiments/06__tma_latency_and_issue run);
# a subset of the steps of src/run.sh with: make ... run STEPS="chase ncu"
run: all
	bash src/run.sh $(STEPS)

# Smoke test: reduced sweeps, appends with its own run_ids (cmdline contains --quick)
quick: all
	bash src/run.sh --quick
