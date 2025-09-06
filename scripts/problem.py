# pddl_problem.py
from dataclasses import dataclass
from typing import List, Any, Tuple
from scripts.utils import tokenize, parse_sexpr

Literal = Tuple[str, tuple] | Tuple[str, Tuple[str, tuple]]  # ('p',args) or ('not',(p,args))


@dataclass
class Problem:
    name: str
    domain_name: str
    objects: List[str]
    init: List[Literal]     # [('p', ('a',...)), ('not', ('q', ('b',...)))]
    goal: Any               # S-expr (and ...)


def parse_problem(problem_text: str) -> Problem:
    toks = tokenize(problem_text)
    sx = parse_sexpr(toks)
    assert sx[0] == "define"

    name, domname = None, None
    objects: List[str] = []
    init_lits: List[Literal] = []
    goal_sx: Any = ["and"]

    for item in sx[1:]:
        if isinstance(item, list) and item and item[0] == "problem":
            name = item[1]
        elif isinstance(item, list) and item and item[0] == ":domain":
            domname = item[1]
        elif isinstance(item, list) and item and item[0] == ":objects":
            objects = [o for o in item[1:] if isinstance(o, str)]
        elif isinstance(item, list) and item and item[0] == ":init":
            for lit in item[1:]:
                if isinstance(lit, list) and lit and lit[0] == "not":
                    pred = lit[1][0]
                    args = tuple(lit[1][1:])
                    init_lits.append(("not", (pred, args)))
                elif isinstance(lit, list):
                    pred = lit[0]
                    args = tuple(lit[1:])
                    init_lits.append((pred, args))
        elif isinstance(item, list) and item and item[0] == ":goal":
            goal_sx = item[1]

    return Problem(name=name, domain_name=domname, objects=objects, init=init_lits, goal=goal_sx)
