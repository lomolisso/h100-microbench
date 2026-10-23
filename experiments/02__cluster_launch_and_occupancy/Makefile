EXP  := 02__cluster_launch_and_occupancy
BINS := cluster_launch
include ../harness/Makefile.inc

# Full run (inside: tools/run --profile --lock-sm-mhz 1635 -- make -C experiments/02__cluster_launch_and_occupancy run)
run: all
	bash src/run.sh
