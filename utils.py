import argparse
from pathlib import Path

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


def main():
    parser = argparse.ArgumentParser()
    questions = sorted(_funcs.keys())
    parser.add_argument(
        "questions",
        choices=(questions + ["all"]),
        nargs="+",
        help="A question ID to run, or 'all'.",
    )
    args = parser.parse_args()
    for q in args.questions:
        if q == "all":
            for q in sorted(_funcs.keys()):
                start = f"== {q} "
                print("\n" + start + "=" * (80 - len(start)))
                run(q)

        else:
            run(q)
