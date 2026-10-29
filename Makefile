# Top-level driver: builds every experiment that has a Makefile, in dependency order.
# Each experiments/<NN__name>/Makefile must provide: all, run, clean (see experiments/harness/Makefile.inc).
EXPS := $(sort $(dir $(wildcard experiments/*/Makefile)))

.PHONY: all run clean $(EXPS) env-info env-build env-verify gpu-status gpu-check shell
all:   ; @for e in $(EXPS); do $(MAKE) -C $$e all   || exit 1; done
run:   ; @for e in $(EXPS); do $(MAKE) -C $$e run   || exit 1; done
clean: ; @for e in $(EXPS); do $(MAKE) -C $$e clean || exit 1; done
$(EXPS): ; $(MAKE) -C $@ all

# Environment: ~/h100/H100.md (repo notes: docs/environment/README.md). GPU work always goes through the
# per-user queue (`h100 run`); the target GPU is the user's choice (`h100-target`).
env-info:   ; @h100 image info
env-build:  ; h100 image build
env-verify: ; h100 image verify $(if $(GPU),--gpu $(GPU))
gpu-status: ; @h100 status
gpu-check:  ; @h100 check $(GPU)
shell:      ; h100 shell $(if $(GPU),--gpu $(GPU))
