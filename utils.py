import argparse
from pathlib import Path
import inspect

import numpy as np

_funcs = {}


def load_dataset(filename, *keys, **kwargs):
    fn = Path("..", "data", filename)
    if not fn.exists() and fn.suffix != ".npz":
        fn = fn.with_suffix(fn.suffix + ".npz")
    data = np.load(fn, allow_pickle=True)
    if keys:
        return [data[k] for k in keys]
    else:
        return dict(**data)


def handle(number):
    def register(func):
        _funcs[number] = func
        return func

    return register


def run(question):
    if question not in _funcs:
        raise ValueError(f"unknown question {question}")
    return _funcs[question]()


def run_with_args(question: str, args: argparse.Namespace):
    if question not in _funcs:
        raise ValueError(f"unknown question {question}")
    fn = _funcs[question]
    sig = inspect.signature(fn)
    if len(sig.parameters) == 0:
        return fn()
    return fn(args)


def main():
    parser = argparse.ArgumentParser()
    questions = sorted(_funcs.keys())
    parser.add_argument(
        "questions",
        choices=(questions + ["all"]),
        nargs="+",
        help="A question ID to run, or 'all'.",
    )
    parser.add_argument(
        "--plot",
        type=str,
        default=None,
        help="Optional output image path for commands that support it (e.g. vae-train).",
    )
    args = parser.parse_args()
    for q in args.questions:
        if q == "all":
            for q in sorted(_funcs.keys()):
                start = f"== {q} "
                print("\n" + start + "=" * (80 - len(start)))
                run_with_args(q, args)

        else:
            run_with_args(q, args)
