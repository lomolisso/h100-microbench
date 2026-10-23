# Top-level driver: builds every experiment that has a Makefile, in dependency order.
# Each experiments/<NN__name>/Makefile must provide: all, run, clean (see experiments/harness/Makefile.inc).
EXPS := $(sort $(dir $(wildcard experiments/*/Makefile)))

.PHONY: all run clean $(EXPS)
all:   ; @for e in $(EXPS); do $(MAKE) -C $$e all   || exit 1; done
run:   ; @for e in $(EXPS); do $(MAKE) -C $$e run   || exit 1; done
clean: ; @for e in $(EXPS); do $(MAKE) -C $$e clean || exit 1; done
$(EXPS): ; $(MAKE) -C $@ all
