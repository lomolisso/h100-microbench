EXP  := 03__sm_gpc_topology
BINS := gpc_probe
include ../harness/Makefile.inc

# Full run (inside: tools/run --profile --lock-sm-mhz 1635 -- make -C experiments/03__sm_gpc_topology run)
run: all
	bash src/run.sh
