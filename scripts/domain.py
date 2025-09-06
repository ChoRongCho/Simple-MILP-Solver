# pddl_domain.py
from dataclasses import dataclass
from typing import List, Dict, Any
from scripts.utils import tokenize, parse_sexpr

@dataclass
class ActionSchema:
    name: str
    parameters: List[str]   # list of var names (untyped)
    precond: Any            # S-expr (and ...)
    effect: Any             # S-expr (and ...)

    def __repr__(self):
        return f"ActionSchema({self.name}, params={self.parameters})"


@dataclass
class Domain:
    name: str
    predicates: Dict[str, int]  # pred -> arity
    actions: List[ActionSchema]


def parse_domain(domain_text: str) -> Domain:
    toks = tokenize(domain_text)
    sx = parse_sexpr(toks)
    assert sx[0] == "define"

    name = None
    predicates: Dict[str, int] = {}
    actions: List[ActionSchema] = []

    for item in sx[1:]:
        # (domain <name>)
        if isinstance(item, list) and len(item) >= 2 and item[0] == "domain":
            name = item[1]

        # (:predicates (p ?x ?y) (q ?x))
        elif isinstance(item, list) and item and item[0] == ":predicates":
            for pred in item[1:]:
                if isinstance(pred, list) and len(pred) >= 1:
                    pname = pred[0]
                    arity = len([a for a in pred[1:] if isinstance(a, str) and a.startswith("?")])
                    predicates[pname] = arity

        # (:action ...)
        elif isinstance(item, list) and item and item[0] == ":action":
            aname = item[1]
            params, precond, effect = [], ["and"], ["and"]
            j = 2
            while j < len(item):
                if item[j] == ":parameters":
                    params = [v for v in item[j+1] if isinstance(v, str) and v.startswith("?")]
                    j += 2
                elif item[j] == ":precondition":
                    precond = item[j+1]
                    j += 2
                elif item[j] == ":effect":
                    effect = item[j+1]
                    j += 2
                else:
                    j += 1
            actions.append(ActionSchema(aname, params, precond, effect))

    return Domain(name=name, predicates=predicates, actions=actions)
