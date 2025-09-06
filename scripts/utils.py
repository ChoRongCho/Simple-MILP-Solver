# pddl_utils.py
import re

# ---------- S-expression tokenizer & parser ----------

def tokenize(s: str):
    """PDDL 주석 제거 + 괄호 기반 토크나이즈"""
    s = re.sub(r";;.*", "", s)   # ';;' 주석 제거 (줄 끝까지)
    s = re.sub(r";.*", "", s)    # ';' 주석도 제거
    s = s.replace("\t", " ")
    s = re.sub(r"\(", " ( ", s)
    s = re.sub(r"\)", " ) ", s)
    toks = [t for t in s.split() if t]
    return toks

def parse_sexpr(tokens):
    """토큰 리스트를 중첩 리스트(S-식)로"""
    def _parse(idx):
        if tokens[idx] == "(":
            lst = []
            idx += 1
            while tokens[idx] != ")":
                node, idx = _parse(idx)
                lst.append(node)
            return lst, idx + 1
        else:
            return tokens[idx].lower(), idx + 1
    sexpr, _ = _parse(0)
    return sexpr

# ---------- Logical helpers ----------

def flatten_and(sx):
    """(and a b c) 구조를 평탄화. 단일식이면 [sx]로 반환."""
    if isinstance(sx, list) and sx and sx[0] == "and":
        out = []
        for e in sx[1:]:
            if isinstance(e, list) and e and e[0] == "and":
                out.extend(flatten_and(e))
            else:
                out.append(e)
        return out
    else:
        return [sx]

def sexpr_to_literal(e):
    """
    (p ?x ?y) -> ('p', ('?x','?y'))
    (not (p ?x)) -> ('not', ('p', ('?x',)))
    """
    if isinstance(e, list) and e and e[0] == "not":
        pe = e[1]
        return ("not", (pe[0], tuple(pe[1:])))
    else:
        return (e[0], tuple(e[1:]))

def atom_to_key(atom_tuple):
    """('pred', ('a','b')) -> 'pred(a,b)'"""
    pred, args = atom_tuple
    return f"{pred}(" + ",".join(args) + ")"
