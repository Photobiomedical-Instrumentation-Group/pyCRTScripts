import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-pycrt")

import cv2 as cv
from pyCRT.frameOperations import drawRoi, rescaleFrame
from pyCRT.videoReading import frameReader, videoCapture
from tomli_w import dump
from tomllib import load

tomlPath = Path("rois_full.toml")
videosDirList = [
    Path("/home/eduardo/Data/miscVideos"),
    Path("/home/eduardo/Data/raquelMasters"),
    Path("/home/eduardo/Data/yutaoNewEquipment"),
    Path("/home/eduardo/Data/yutaoPostNew"),
    Path("/home/eduardo/Data/WorstVideos"),
    # Path("TrainingVideos"),
]
# RUN_ON = Path("/home/eduardo/Data/raquelMasters") / "P06CR2.wmv"
RUN_ON = Path("/home/eduardo/Data/WorstVideos") / "2.mp4"
if tomlPath.exists():
    with tomlPath.open("rb") as file:
        tomlDict = load(file)
else:
    tomlDict = {"roi": {}}

VIDEO_EXTENSIONS = [".MOV", ".wmv", ".mp4"]
SKIP_DUPLICATES = False
RESCALE_FACTOR = 0.5


def iterVideoPathsInDirectory(videoDir):
    # {{{
    for filePath in sorted(
        Path(videoDir).iterdir(), key=lambda path: path.name.lower()
    ):
        if filePath.suffix in VIDEO_EXTENSIONS:
            yield filePath


# }}}


def iterRunOnVideoPaths(runOn):
    # {{{
    if isinstance(runOn, (list, tuple, set)):
        for item in runOn:
            yield from iterRunOnVideoPaths(item)
        return

    runOnPath = Path(runOn)
    if runOnPath.is_dir():
        yield from iterVideoPathsInDirectory(runOnPath)
        return

    if runOnPath.suffix in VIDEO_EXTENSIONS:
        yield runOnPath


# }}}


def selectVideoRoi(filePath):
    # {{{
    roi = tuple(tomlDict["roi"].get(filePath.name, []))
    print(f"\nVideo: {filePath.name}")
    print(f"Directory: {filePath.parent}")
    if roi:
        print(f"Current ROI: {roi}")
    else:
        print("Current ROI: none")

    skipVideo = (filePath.name in tomlDict["roi"]) and SKIP_DUPLICATES
    if skipVideo:
        print(f"Skipping {filePath.stem}...")
        return

    with videoCapture(str(filePath)) as cap:
        for frame in frameReader(cap):
            frame = rescaleFrame(frame, RESCALE_FACTOR)
            if roi:
                frame = drawRoi(frame, roi)

            cv.imshow("frame", frame)
            key = cv.waitKey(2)

            if key == ord("q"):
                break
            if key == ord("s"):
                print(f"Skipping {filePath.stem}...")
                break
            if key == ord(" "):
                roi = cv.selectROI("frame", frame)
                tomlDict["roi"][filePath.name] = roi
                print(f"{filePath.stem}: {roi}")


# }}}


def main():
    # {{{
    for filePath in iterRunOnVideoPaths(RUN_ON):
        if not filePath.exists():
            print(f"Skipping {filePath}: file not found")
            continue

        selectVideoRoi(filePath)

    with tomlPath.open("wb") as file:
        dump(tomlDict, file)
    cv.destroyAllWindows()


# }}}


if __name__ == "__main__":
    main()
