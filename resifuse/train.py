import argparse
from types import SimpleNamespace

from resifuse.training.freely import main as freely_main
from resifuse.training.mabe import main as mabe_main
from resifuse.training.calms21 import main as calms21_main
from resifuse.config import load_config_with_defaults


def parse_args():
    parser = argparse.ArgumentParser(description="ResiFuse pretraining and supervised fine-tuning")
    parser.add_argument(
        "-c",
        "--config",
        default="./config.yaml",
        help="Path to config.yaml (or JSON) to use.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed override. When set, seeds python/numpy/torch/cuda "
        "before model construction so the same value gives the same init across "
        "runs/budgets (enables genuine paired comparisons). Unset = current "
        "(unseeded) behavior.",
    )
    return parser.parse_args()


def main():
    cli_args = parse_args()
    cfg = load_config_with_defaults(cli_args.config)
    args = SimpleNamespace(**cfg)
    dataset_name = str(getattr(args, "dataset", "freely_mouse") or "freely_mouse").lower()
    if dataset_name.startswith("calms21") or dataset_name.startswith("calsm21"):
        calms21_main(cli_args.config, seed=cli_args.seed)
    elif dataset_name.startswith("mabe"):
        mabe_main(cli_args.config, seed=cli_args.seed)
    else:
        freely_main(cli_args.config, seed=cli_args.seed)


if __name__ == "__main__":
    main()
