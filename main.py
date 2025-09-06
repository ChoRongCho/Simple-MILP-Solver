# main.py
import argparse

from scripts.solver import StripsMILPPlanner

def read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def main():
    parser = argparse.ArgumentParser(description="STRIPS PDDL → MILP Planner (PuLP)")
    # 요청대로 --o (domain), --i (problem) 지원
    parser.add_argument("--o", "--domain", dest="domain_path", required=True, help="PDDL domain file path")
    parser.add_argument("--i", "--problem", dest="problem_path", required=True, help="PDDL problem file path")
    parser.add_argument("--H", "--horizon", dest="horizon", type=int, default=3, help="Plan horizon (time steps)")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    args = parser.parse_args()

    domain_text = read_file(args.domain_path)
    problem_text = read_file(args.problem_path)

    planner = StripsMILPPlanner(domain_text, problem_text)
    planner.build_model(horizon=args.horizon)
    planner.solve(verbose=not args.quiet)

if __name__ == "__main__":
    main()
