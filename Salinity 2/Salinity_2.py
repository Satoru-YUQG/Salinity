"""CLI entry: repeated train/validate runs with settings from config or arguments."""

import argparse
import json

from config import DEFAULT_DATA_CANDIDATES
from pipeline import RunConfig, config_to_dict, run_experiment


def parse_args():
    parser = argparse.ArgumentParser(description="Salinity classification experiments")
    parser.add_argument("--algorithm", default="random_forest")
    parser.add_argument("--sampling", default="smote", choices=["none", "smote", "adasyn"])
    parser.add_argument("--salinity-threshold", type=float, default=3.0)
    parser.add_argument("--decision-threshold", type=float, default=0.5)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--drop", nargs="*", default=[], help="Extra columns to exclude")
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    normalize = True
    if args.no_normalize:
        normalize = False
    if args.normalize:
        normalize = True

    config = RunConfig(
        data_path=args.data_path,
        class_threshold=args.salinity_threshold,
        decision_threshold=args.decision_threshold,
        drop_columns=args.drop,
        algorithm=args.algorithm,
        sampling=args.sampling,
        normalize=normalize,
        n_runs=args.runs,
        test_size=args.test_size,
        random_seed=args.seed,
    )

    print("Salinity ML — repeated hold-out evaluation")
    print("Data search paths:")
    for p in DEFAULT_DATA_CANDIDATES:
        print(f"  {p}")
    print(json.dumps(config_to_dict(config), indent=2))

    def progress(current, total):
        print(f"Run {current}/{total}")

    result = run_experiment(config, progress_callback=progress)

    print("\n====== SUMMARY (mean ± std over runs) ======")
    for key, value in result.summary.items():
        print(f"{key}: {value:.4f}")

    print("\nClass balance (0=non-saline, 1=saline):")
    print(json.dumps(result.class_balance, indent=2))
    print(f"\nSamples: {result.n_samples}, Features: {result.n_features}")


if __name__ == "__main__":
    main()
