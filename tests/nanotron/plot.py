import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    # 현재 디렉토리에서 loss_*.json 찾기
    json_files = sorted(glob.glob("loss_*.json"))
    if not json_files:
        print("No loss_*.json files found in current directory.")
        return

    all_losses = {}
    for path in json_files:
        base = os.path.basename(path)
        if not base.startswith("loss_") or not base.endswith(".json"):
            continue

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        losses = data.get("losses", None)
        if not isinstance(losses, list) or len(losses) == 0:
            print(f"[warn] {path} has no valid 'losses' list. Skipping.")
            continue

        label = data.get("label", base[len("loss_") : -len(".json")])
        all_losses[label] = losses
        print(f"[load] {label}: {len(losses)} steps from {path}")

    if not all_losses:
        print("No valid loss data found.")
        return

    # 그래프 그리기 (모든 loss를 한 그래프에 overlay)
    plt.figure(figsize=(10, 5))
    for label, losses in all_losses.items():
        steps = range(len(losses))
        plt.plot(steps, losses, label=label)

    plt.title("Loss per step (all runs)")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_path = "loss_overlay.png"
    plt.savefig(out_path)
    plt.close()
    print(f"[save] wrote {out_path}")


if __name__ == "__main__":
    main()
