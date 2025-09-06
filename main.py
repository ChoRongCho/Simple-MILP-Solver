# main.py
from scripts.domain import parse_domain
from scripts.problem import parse_problem
from scripts.solver import build_and_solve

domain_str = r"""
(define (domain pickplace)
  (:predicates
    (at ?r ?room)
    (belong_in ?cup ?room)
    (handempty ?r)
    (holding ?r)
  )

  (:action pick
    :parameters (?r ?room ?cup)
    :precondition (and (at ?r ?room) (handempty ?r) (belong_in ?cup ?room))
    :effect (and (holding ?r) (not (handempty ?r)) (not (belong_in ?cup ?room)))
  )

  (:action place
    :parameters (?r ?room ?cup)
    :precondition (and (at ?r ?room) (holding ?r))
    :effect (and (belong_in ?cup ?room) (handempty ?r) (not (holding ?r)))
  )

  (:action move
    :parameters (?r ?from ?to)
    :precondition (and (at ?r ?from))
    :effect (and (at ?r ?to) (not (at ?r ?from)))
  )
)
"""

problem_str = r"""
(define (problem pp1)
  (:domain pickplace)
  (:objects robot room1 room2 cup1)
  (:init
    (at robot room1)
    (belong_in cup1 room1)
    (handempty robot)
    (not (holding robot))
  )
  (:goal (and
    (belong_in cup1 room2)
  ))
)
"""

if __name__ == "__main__":
    dom = parse_domain(domain_str)
    prob = parse_problem(problem_str)
    # 수평선 H=3 (pick -> move -> place)
    build_and_solve(dom, prob, horizon=3, verbose=True)
