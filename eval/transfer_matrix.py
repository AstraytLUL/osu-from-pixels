"""The transfer matrix: a checkpoint's closed-loop accuracy on a split's maps, per mod and style
(the project proposal's headline figure, mod x style).

    python -m eval.transfer_matrix --checkpoint runs/baseline_seed0/best.pt --split val --mods NM HR
    python -m eval.transfer_matrix --checkpoint ... --split test --mods NM HR DT HD    # final runs only
    python -m eval.transfer_matrix --checkpoint ... --motor cap      # the speed cap, no landing noise (eval/motor.py)

Prints the mean accuracy per (mod, style) and writes every map's result to --out (JSON).
Use the test split only for final results: model selection uses NoMod val (the project proposal).
"""
import argparse
import json
from pathlib import Path

import numpy as np

from eval.motor import MOTORS, Motor
from eval.policy import load_model_policy
from eval.rollout import evaluate, split_maps

STYLES = ("jumps", "streams", "tech")


def main():
    ap = argparse.ArgumentParser(description="Closed-loop accuracy per mod and style.")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--mods", nargs="+", default=["NM"])
    ap.add_argument("--out", type=Path, default=Path("runs/transfer_matrix.json"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--motor", choices=list(MOTORS), default="cap", help="eval/motor.py's HUMAN, CAP or FREE")
    ap.add_argument("--max-step", type=float, help="override the motor's top speed (px per frame)")
    ap.add_argument("--aim-noise", type=float, help="override the motor's noise (fraction of the step)")
    args = ap.parse_args()
    motor = MOTORS[args.motor]
    motor = Motor(args.max_step if args.max_step is not None else motor.max_step,
                  args.aim_noise if args.aim_noise is not None else motor.noise)

    policy = load_model_policy(args.checkpoint, args.device)
    results = {}
    for mods in args.mods:
        report = evaluate(policy, split_maps(args.split, mods), motor=motor)
        results[mods] = report
        print(f"{mods}: mean accuracy {report['accuracy']:.2%}, miss rate {report['miss_rate']:.2%}, "
              f"combo breaks {report['breaks_per_100']:.1f} / 100 objects", flush=True)

    print(f"\n{'':6}" + "".join(f"{s:>10}" for s in STYLES))
    for mods, report in results.items():
        cells = []
        for style in STYLES:
            accs = [r["accuracy"] for r in report["maps"] if r["style"] == style]
            cells.append(f"{np.mean(accs):>10.2%}" if accs else f"{'-':>10}")
        print(f"{mods:6}" + "".join(cells))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"checkpoint": args.checkpoint, "split": args.split,
                                    "motor": {"max_step": motor.max_step, "noise": motor.noise}, "results": results}, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
