# strips_milp.py
import re
import itertools
import pulp as pl
from typing import List, Dict, Any, Tuple, Set

from scripts.utils import flatten_and, sexpr_to_literal, atom_to_key
from scripts.domain import Domain, ActionSchema, parse_domain
from scripts.problem import Problem, parse_problem

# strips_milp.py
class StripsMILPPlanner:
    """
    PDDL(STRIPS) -> MILP(PuLP) 플래너
    - 직렬 계획 (한 시점 최대 1 액션)
    - 양의 원자만 상태변수로 모델링, (not g)는 최종 상태=0 제약으로 처리
    - 목적: 총 액션 수 최소화 (최단 계획)
    """
    def __init__(self, domain_text: str, problem_text: str):
        self.domain: Domain = parse_domain(domain_text)
        self.problem: Problem = parse_problem(problem_text)

        # Grounding/모델 구성에 쓰일 내부 상태
        self.grounded_actions: List[Dict[str, Any]] = []
        self.P: List[str] = []                    # 상태변수(양의 원자) 키
        self.GA: List[Dict[str, Any]] = []       # grounded action map (pre/add/del)

        # MILP 관련
        self.H: int | None = None
        self.model: pl.LpProblem | None = None
        self.s = {}   # state vars: (p,t) -> var
        self.x = {}   # action vars: (a_name,t) -> var

    # ---------- Grounding ----------
    @staticmethod
    def _ground_action(schema: ActionSchema, objects: List[str]) -> List[Dict[str, Any]]:
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

    @staticmethod
    def _collect_fluents(problem: Problem, grounded_actions: List[Dict[str, Any]]) -> List[str]:
        atoms: Set[Tuple[str, tuple]] = set()
        # init
        for lit in problem.init:
            if lit[0] != "not":
                atoms.add((lit[0], lit[1]))
        # goal
        goal_lits = [sexpr_to_literal(g) for g in flatten_and(problem.goal)]
        for lit in goal_lits:
            if lit[0] != "not":
                atoms.add((lit[0], lit[1]))
        # effects
        for ga in grounded_actions:
            for lit in ga["eff"]:
                if lit[0] != "not":
                    atoms.add((lit[0], lit[1]))

        def atom_str(a): return f"{a[0]}(" + ",".join(a[1]) + ")"
        return sorted(list(set(atom_str(a) for a in atoms)))

    @staticmethod
    def _lits_to_add_del(eff_lits):
        add, dele = [], []
        for lit in eff_lits:
            if lit[0] == "not":
                dele.append(lit[1])  # (pred,args)
            else:
                add.append(lit)
        return add, dele

    @staticmethod
    def _litlist_to_pos(lits):
        return [l for l in lits if l[0] != "not"]

    # ---------- Public API ----------
    def build_model(self, horizon: int):
        """수평선 H로 MILP 모델 생성."""
        self.H = int(horizon)

        # 1) Ground all actions
        self.grounded_actions = []
        for a in self.domain.actions:
            self.grounded_actions += self._ground_action(a, self.problem.objects)

        # 2) Collect positive fluents (state variables)
        self.P = self._collect_fluents(self.problem, self.grounded_actions)

        # 3) Build GA
        self.GA = []
        for g in self.grounded_actions:
            add, dele = self._lits_to_add_del(g["eff"])
            self.GA.append({
                "name": g["name"],
                "pre_pos": self._litlist_to_pos(g["pre"]),
                "add": [atom_to_key(a) for a in add],
                "del": [atom_to_key(d) for d in dele],
            })

        # 4) Init/Goal sets
        I_true: Set[str] = set()
        I_false: Set[str] = set()
        for lit in self.problem.init:
            if lit[0] == "not":
                I_false.add(atom_to_key(lit[1]))
            else:
                I_true.add(atom_to_key(lit))
        goal_lits = [sexpr_to_literal(g) for g in flatten_and(self.problem.goal)]
        G_true: Set[str] = set(atom_to_key(l) for l in goal_lits if l[0] != "not")
        G_false: Set[str] = set(atom_to_key(l[1]) for l in goal_lits if l[0] == "not")

        # 5) MILP
        H = self.H
        m = pl.LpProblem("PDDL_STRIPS_to_MILP", pl.LpMinimize)

        # state/action vars
        self.s = {(p, t): pl.LpVariable(f"s_{t}_{re.sub(r'[^a-zA-Z0-9_]', '_', p)}", 0, 1, cat="Binary")
                  for p in self.P for t in range(H+1)}
        self.x = {(a["name"], t): pl.LpVariable(f"x_{t}_{re.sub(r'[^a-zA-Z0-9_]', '_', a['name'])}", 0, 1, cat="Binary")
                  for a in self.GA for t in range(H)}

        # Init
        for p in self.P:
            if p in I_true:
                m += self.s[(p, 0)] == 1
            elif p in I_false:
                m += self.s[(p, 0)] == 0
            else:
                m += self.s[(p, 0)] == 0

        # Goal
        for p in G_true:
            if p in self.P:
                m += self.s[(p, H)] == 1
        for p in G_false:
            if p in self.P:
                m += self.s[(p, H)] == 0

        # Serial plan
        for t in range(H):
            m += pl.lpSum(self.x[(a["name"], t)] for a in self.GA) <= 1

        # Preconditions: x[a,t] <= s[p,t]
        for t in range(H):
            for a in self.GA:
                for (pred, args) in a["pre_pos"]:
                    key = atom_to_key((pred, args))
                    if key in self.P:
                        m += self.x[(a["name"], t)] <= self.s[(key, t)]

        # Frame constraints (ADD/DEL 4-ineq)
        ADD: Dict[str, List[str]] = {p: [] for p in self.P}
        DEL: Dict[str, List[str]] = {p: [] for p in self.P}
        for a in self.GA:
            for p in a["add"]:
                if p in ADD:
                    ADD[p].append(a["name"])
            for p in a["del"]:
                if p in DEL:
                    DEL[p].append(a["name"])

        for t in range(H):
            for p in self.P:
                add_sum = pl.lpSum(self.x[(an, t)] for an in ADD[p]) if ADD[p] else 0
                del_sum = pl.lpSum(self.x[(an, t)] for an in DEL[p]) if DEL[p] else 0

                m += self.s[(p, t+1)] >= self.s[(p, t)] - del_sum
                m += self.s[(p, t+1)] <= self.s[(p, t)] + add_sum

                if ADD[p]:
                    m += self.s[(p, t+1)] >= add_sum
                if DEL[p]:
                    m += self.s[(p, t+1)] <= 1 - del_sum

        # Objective: minimize total actions
        m += pl.lpSum(self.x.values())

        self.model = m
        return m

    def solve(self, verbose: bool = True) -> bool:
        """현재 모델(MILP)을 풂. True=feasible/optimal."""
        assert self.model is not None, "call build_model(horizon) first."
        status = self.model.solve(pl.PULP_CBC_CMD(msg=False))
        from pulp import LpStatus
        feasible = (LpStatus[status] in ("Optimal", "Not Solved", "Feasible", "Infeasible") and
                    LpStatus[status] != "Infeasible")

        if verbose:
            print("Status:", LpStatus[status])
            if self.model.objective is not None:
                try:
                    import pulp as pl
                    print("Objective (total actions):", pl.value(self.model.objective))
                except Exception:
                    pass
            self.print_plan_and_states()

        return feasible

    def print_plan_and_states(self):
        """해 찾은 뒤 플랜/상태 궤적 출력."""
        if self.model is None or self.H is None:
            return
        H = self.H
        print("\nPlan:")
        for t in range(H):
            for a in self.GA:
                var = self.x[(a["name"], t)]
                try:
                    if var.value() is not None and var.value() > 0.5:
                        print(f"  t={t}: {a['name']}")
                except Exception:
                    pass

        print("\nState trajectory:")
        for t in range(H+1):
            truths = []
            for p in self.P:
                v = self.s[(p, t)].value()
                if v is not None and v > 0.5:
                    truths.append(p)
            print(f"  t={t}: {sorted(truths)}")
