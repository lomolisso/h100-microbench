EXP  := 01__timers_and_harness
BINS := timers harness_selftest
include ../harness/Makefile.inc

# Full run (inside: tools/run --profile --lock-sm-mhz 1635 -- make -C experiments/01__timers_and_harness run)
run: all
	bash src/run.sh
