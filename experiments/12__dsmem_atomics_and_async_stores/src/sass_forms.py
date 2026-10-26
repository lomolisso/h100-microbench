"""SASS form of every atomic op × state space, read from the committed dump results/sass/atom_lat.sass (the K1 kernels
k1_smem<OP, SEM, KIND, SP> and k1_glob<OP, SEM, KIND, FINE>, default semantics, kinds atom and red+ld).
forms() → DataFrame: op, space (shared::cta | shared::cluster | global), kind (atom | red), sass (the distinct atomic
opcodes of the kernel, ' + ' joined; the global kernels' own atomicCAS of the SM claim is left out), cas_loop (a compare-and-swap
loop: CAS / CAST.SPIN opcodes for an op that is not cas). A .shared::cluster op compiles to a run-time branch on the address:
the own CTA's window (ATOMS), a peer's smem and the generic global path (ATOM.E) each get their own form."""
import re

import pandas as pd

OPS = ["add.u32", "add.u64", "inc.u32", "min.s32", "max.u32", "and.b32", "exch.b32", "cas.b32", "cas.b64", "add.f32", "add.f64", "add.f16x2",
       "add.bf16x2", "add.v2.f32", "add.v4.f32", "add.v2.f16x2", "add.v4.f16x2"]
KIND = {0: "atom", 1: "red"}
SP = {0: "shared::cta", 1: "shared::cluster"}
PAT = re.compile(r"\b((?:ATOMS|ATOMG|ATOM|REDG|RED|REDAS|STAS)(?![A-Z0-9])(?:\.[A-Z0-9_]+)*)")


def forms(path="results/sass/atom_lat.sass"):
    funcs, cur = {}, None
    for line in open(path):
        m = re.search(r"Function : (\S+)", line)
        if m:
            cur = m.group(1)
            funcs[cur] = set()
            continue
        if cur:
            for op in PAT.findall(line):
                funcs[cur].add(op)
    rows = []
    for name, ops in funcs.items():
        m = re.match(r"_Z7k1_smemILi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)EEv3K1S", name)
        g = re.match(r"_Z7k1_globILi(\d+)ELi(\d+)ELi(\d+)ELi0EEv3K1G", name)
        if m:
            op, sem, kind, sp = map(int, m.groups())
            space = SP[sp]
        elif g:
            op, sem, kind = map(int, g.groups())
            space = "global"
        else:
            continue
        if sem != 0 or kind not in KIND:
            continue
        ops = sorted(o for o in ops if not o.startswith(("STAS", "REDAS")))
        if space == "global" and OPS[op] != "cas.b32":  # the SM claim (32-bit atomicCAS on a global word) is not the op
            ops = [o for o in ops if o != "ATOMG.E.CAS.STRONG.GPU"]
        loop = any(".CAS" in o or "CAST" in o for o in ops) and not OPS[op].startswith("cas")
        rows.append(dict(op=OPS[op], space=space, kind=KIND[kind], sass=" + ".join(ops), cas_loop=loop))
    return pd.DataFrame(rows).sort_values(["op", "kind", "space"]).reset_index(drop=True)


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 120)
    print(forms().to_string())
