EXP  := 04__memory_hierarchy_latency
BINS := lat_smem lat_footprint lat_fine
include ../harness/Makefile.inc

$(addprefix $(BIN_DIR)/,$(BINS)): src/pchase_chain.hpp src/gpc_map.hpp

# Full run (inside: tools/run --profile --lock-sm-mhz 1635 -- make -C experiments/04__memory_hierarchy_latency run)
run: all
	bash src/run.sh

# Smoke test: a few footprints, 2 probe SMs, appends with its own run_ids (cmdline contains --quick)
quick: all
	bash src/run.sh --quick
