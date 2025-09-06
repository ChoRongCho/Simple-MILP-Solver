# strips_milp.py
import re
import itertools
import pulp as pl
from typing import List, Dict, Any, Tuple, Set

from scripts.utils import flatten_and, sexpr_to_literal, atom_to_key
from scripts.domain import Domain, ActionSchema
from scripts.problem import Problem

# ---------- Grounding ----------

def ground_action(schema: ActionSchema, objects: List[str]) -> List[Dict[str, Any]]:
    """모든 파라미터 조합으로 스키마 인스턴스 생성 (untyped)."""
    varlist = schema.parameters
    envs: List[Dict[str, str]] = [dict()] if not varlist else [
        dict(zip(varlist, tup)) for tup in itertools.product(objects, repeat=len(varlist))
    ]

    def subst(expr, env):
        if isinstance(expr, str):
            if expr.startswith("?") and expr in env:
                return env[expr]
            return expr
        elif isinstance(expr, list):
            return [subst(x, env) for x in expr]
        else:
            return expr

    grounded = []
    for env in envs:
        pre = flatten_and(schema.precond)
        eff = flatten_and(schema.effect)
        pre_lits = [sexpr_to_literal(subst(p, env)) for p in pre]
        eff_lits = [sexpr_to_literal(subst(e, env)) for e in eff]
        grounded.append({
            "name": f"{schema.name}(" + ",".join(env.get(v, v) for v in varlist) + ")",
            "pre": pre_lits,
            "eff": eff_lits
        })
    return grounded

def collect_fluents(domain: Domain, problem: Problem, grounded_actions: List[Dict[str, Any]]) -> List[str]:
    """상태변수로 쓸 '양의 원자' 문자열 키 집합 수집."""
    atoms: Set[Tuple[str, tuple]] = set()

    # init/goal/effects 에 등장하는 모든 양의 원자 수집
    for lit in problem.init:
        if lit[0] != "not":
            atoms.add((lit[0], lit[1]))

    goal_lits = [sexpr_to_literal(g) for g in flatten_and(problem.goal)]
    for lit in goal_lits:
        if lit[0] != "not":
            atoms.add((lit[0], lit[1]))

    for ga in grounded_actions:
        for lit in ga["eff"]:
            if lit[0] != "not":
                atoms.add((lit[0], lit[1]))

    def atom_str(a): return f"{a[0]}(" + ",".join(a[1]) + ")"
    return sorted(list(set(atom_str(a) for a in atoms)))

def lits_to_add_del(eff_lits):
    add, dele = [], []
    for lit in eff_lits:
        if lit[0] == "not":
            dele.append(lit[1])  # (pred, args)
        else:
            add.append(lit)
    return add, dele

def litlist_to_pos(lits):
    """('p',args) / ('not',(p,args)) 리스트에서 양의 리터럴만."""
    pos = []
    for l in lits:
        if l[0] != "not":
            pos.append(l)
    return pos

# ---------- MILP Builder ----------

def build_and_solve(domain: Domain, problem: Problem, horizon=3, verbose=True) -> pl.LpProblem:
    # Ground actions
    grounded: List[Dict[str, Any]] = []
    for a in domain.actions:
        grounded += ground_action(a, problem.objects)

    # Fluents (양의 원자만 상태변수)
    P: List[str] = collect_fluents(domain, problem, grounded)

    # Ground actions -> ADD/DEL/Pre 맵
    GA: List[Dict[str, Any]] = []
    for g in grounded:
        add, dele = lits_to_add_del(g["eff"])
        GA.append({
            "name": g["name"],
            "pre_pos": litlist_to_pos(g["pre"]),
            "add": [atom_to_key(a) for a in add],
            "del": [atom_to_key(d) for d in dele],
        })

    # 초기/목표 집합
    I_true: Set[str] = set()
    I_false: Set[str] = set()
    for lit in problem.init:
        if lit[0] == "not":
            I_false.add(atom_to_key(lit[1]))
        else:
            I_true.add(atom_to_key(lit))

    goal_lits = [sexpr_to_literal(g) for g in flatten_and(problem.goal)]
    G_true: Set[str] = set(atom_to_key(l) for l in goal_lits if l[0] != "not")
    G_false: Set[str] = set(atom_to_key(l[1]) for l in goal_lits if l[0] == "not")

    # ---------- MILP ----------
    H = horizon
    m = pl.LpProblem("PDDL_STRIPS_to_MILP", pl.LpMinimize)

    # 상태변수 s[p,t], 액션변수 x[a,t]
    s = {(p, t): pl.LpVariable(f"s_{t}_{re.sub(r'[^a-zA-Z0-9_]', '_', p)}", 0, 1, cat="Binary")
         for p in P for t in range(H+1)}
    x = {(a["name"], t): pl.LpVariable(f"x_{t}_{re.sub(r'[^a-zA-Z0-9_]', '_', a['name'])}", 0, 1, cat="Binary")
         for a in GA for t in range(H)}

    # 초기 상태
    for p in P:
        if p in I_true:
            m += s[(p, 0)] == 1
        elif p in I_false:
            m += s[(p, 0)] == 0
        else:
            m += s[(p, 0)] == 0  # STRIPS 기본 0

    # 목표
    for p in G_true:
        if p in P:
            m += s[(p, H)] == 1
    for p in G_false:
        if p in P:
            m += s[(p, H)] == 0

    # 직렬 계획: 한 시점 하나만 실행
    for t in range(H):
        m += pl.lpSum(x[(a["name"], t)] for a in GA) <= 1

    # 전제: x[a,t] ≤ s[p,t]
    for t in range(H):
        for a in GA:
            for (pred, args) in a["pre_pos"]:
                key = atom_to_key((pred, args))
                if key in P:
                    m += x[(a["name"], t)] <= s[(key, t)]

    # ADD/DEL 인덱스
    ADD: Dict[str, List[str]] = {p: [] for p in P}
    DEL: Dict[str, List[str]] = {p: [] for p in P}
    for a in GA:
        for p in a["add"]:
            if p in ADD:
                ADD[p].append(a["name"])
        for p in a["del"]:
            if p in DEL:
                DEL[p].append(a["name"])

    # 프레임(간단 4부등식)
    for t in range(H):
        for p in P:
            add_sum = pl.lpSum(x[(an, t)] for an in ADD[p]) if ADD[p] else 0
            del_sum = pl.lpSum(x[(an, t)] for an in DEL[p]) if DEL[p] else 0

            m += s[(p, t+1)] >= s[(p, t)] - del_sum
            m += s[(p, t+1)] <= s[(p, t)] + add_sum

            if ADD[p]:
                m += s[(p, t+1)] >= add_sum
            if DEL[p]:
                m += s[(p, t+1)] <= 1 - del_sum

    # 목적: 총 액션 수 최소화(= 최단 계획)
    m += pl.lpSum(x.values())

    status = m.solve(pl.PULP_CBC_CMD(msg=False))

    if verbose:
        from pulp import LpStatus
        print("Status:", LpStatus[status])
        print("Objective (total actions):", pl.value(m.objective))
        print("\nPlan:")
        for t in range(H):
            for a in GA:
                if pl.value(x[(a["name"], t)]) > 0.5:
                    print(f"  t={t}: {a['name']}")
        print("\nState trajectory:")
        for t in range(H+1):
            truths = [p for p in P if pl.value(s[(p, t)]) > 0.5]
            print(f"  t={t}: {sorted(truths)}")

    return m
