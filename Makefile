# Top-level driver: builds every experiment that has a Makefile, in dependency order.
# Each experiments/<NN__name>/Makefile must provide: all, run, clean (see experiments/harness/Makefile.inc).
EXPS := $(sort $(dir $(wildcard experiments/*/Makefile)))

.PHONY: all run clean $(EXPS) env-info env-build env-verify gpu-status gpu-check shell
all:   ; @for e in $(EXPS); do $(MAKE) -C $$e all   || exit 1; done
run:   ; @for e in $(EXPS); do $(MAKE) -C $$e run   || exit 1; done
clean: ; @for e in $(EXPS); do $(MAKE) -C $$e clean || exit 1; done
$(EXPS): ; $(MAKE) -C $@ all

# Environment (see docs/environment/README.md). GPU work always goes through tools/run.
env-info:   ; @tools/env info
env-build:  ; tools/env build
env-verify: ; tools/env verify $(if $(GPU),--gpu $(GPU))
gpu-status: ; @tools/gpu status
gpu-check:  ; @tools/gpu check $(if $(GPU),--gpu $(GPU))
shell:      ; tools/run $(if $(GPU),--gpu $(GPU),--no-gpu)
