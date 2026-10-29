EXP  := 05__memory_hierarchy_bandwidth
BINS := bw_smem_l1 bw_l2 bw_dram
include ../harness/Makefile.inc

# bw_common.hpp reads the 03 smid→GPC map through 04's reader.
$(addprefix $(BIN_DIR)/,$(BINS)): ../04__memory_hierarchy_latency/src/gpc_map.hpp src/bw_common.hpp

# Full run (inside: h100 run --profile --lock-sm-mhz 1635 -- make -C experiments/05__memory_hierarchy_bandwidth run);
# a subset of the steps of src/run.sh with: make ... run STEPS="k12 link ncul2"
run: all
	bash src/run.sh $(STEPS)

# Smoke test: reduced sweeps, appends with its own run_ids (cmdline contains --quick)
quick: all
	bash src/run.sh --quick
