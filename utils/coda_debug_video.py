from __future__ import annotations

import argparse
import glob

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Replay CODA debug NPZ dumps as a side-by-side video view.")
    parser.add_argument(
        "--pattern",
        default="data/coda_debug/*.npz",
        help="Glob pattern for CODA NPZ files.",
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=0.25,
        help="Seconds to pause between frames.",
    )
    args = parser.parse_args()

    files = sorted(glob.glob(args.pattern))
    if not files:
        print(f"No NPZ files found for pattern: {args.pattern}")
        return

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    titles = ["Frame", "Reconstruction", "Predicted"]
    for i, t in enumerate(titles):
        ax[i].set_title(t)
        ax[i].axis("off")

    for path in files:
        d = np.load(path)
        imgs = [d["frame"], d["reconstruction"], d["predicted_reconstruction"]]
        for i, img in enumerate(imgs):
            ax[i].imshow(np.clip(img, 0, 1))
        fig.suptitle(path.split("/")[-1])
        plt.pause(max(0.0, float(args.pause)))

    plt.show()


if __name__ == "__main__":
    main()
