"""Draw the article figures using the original command and output formats."""
import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path

import fitz
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "replay_output" / "main")
    output = parser.parse_args().output
    draw = ROOT / "article" / "figures" / "draw_figures.py"
    subprocess.run([sys.executable, str(draw), "--output", str(output)], check=True)
    shutil.copyfile(output / "plotted_values.tsv", output / "check_values.tsv")

    with (output / "sizes.tsv").open("w", encoding="utf-8", newline="") as stream:
        sizes = csv.writer(stream, delimiter="\t", lineterminator="\n")
        sizes.writerow(["figure", "width_mm", "height_mm", "tiff_px_w", "tiff_px_h"])
        for number in range(1, 7):
            stem = f"Figure_{number}"
            with fitz.open(output / f"{stem}.pdf") as pdf:
                page = pdf[0]
                page.get_pixmap(dpi=600).save(output / f"{stem}.png")
                raster = page.get_pixmap(dpi=1000, colorspace=fitz.csRGB, alpha=False)
                with Image.frombytes("RGB", (raster.width, raster.height), raster.samples) as rgb:
                    rgb.save(output / f"{stem}.tif", compression="tiff_lzw", dpi=(1000, 1000))
                sizes.writerow([stem, round(page.rect.width * 25.4 / 72, 1),
                                round(page.rect.height * 25.4 / 72, 1), raster.width, raster.height])


if __name__ == "__main__":
    main()
