import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-pycrt")

from pyCRT import PCRT

VIDEOS_DIR = Path("/home/eduardo/Data/raquelMastersRoi")
OUTPUT_CSV_PATH = Path("raquel_masters_roi_pcrt.csv")
ROI = "all"
DISPLAY_VIDEO = False
LIVE_PLOT = False
WRITE_HEADER = True


def iterVideoFiles(videosDir):
    for filePath in sorted(
        Path(videosDir).iterdir(), key=lambda path: path.name.lower()
    ):
        if filePath.is_file():
            yield filePath


def measurePcrt(videoPath):
    pcrtObj = PCRT.fromVideoFile(
        str(videoPath),
        roi=ROI,
        displayVideo=DISPLAY_VIDEO,
        livePlot=LIVE_PLOT,
        exclusionCriteria=0.5,
    )
    pcrt, uncertainty = pcrtObj.pCRT
    return float(pcrt), float(uncertainty)


def main():
    videoFiles = list(iterVideoFiles(VIDEOS_DIR))
    with OUTPUT_CSV_PATH.open("w", newline="") as file:
        writer = csv.writer(file)
        if WRITE_HEADER:
            writer.writerow(["video_file_name", "pcrt", "uncertainty"])

        for videoNum, videoPath in enumerate(videoFiles, start=1):
            try:
                pcrt, uncertainty = measurePcrt(videoPath)
            except Exception as err:
                print(
                    f"failed {videoPath.name}: {type(err).__name__}: {err}",
                    flush=True,
                )
                pcrt = float("nan")
                uncertainty = float("nan")

            writer.writerow([videoPath.name, pcrt, uncertainty])
            print(
                f"measured {videoPath.name}: {pcrt}, {uncertainty}, "
                f"{videoNum}/{len(videoFiles)}",
                flush=True,
            )

    print(f"wrote {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    main()
