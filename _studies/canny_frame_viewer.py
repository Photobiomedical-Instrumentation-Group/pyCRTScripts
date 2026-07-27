import os
import re
import tkinter as tk
from pathlib import Path

import cv2 as cv
import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-pycrt")

from funcs import cannyEdgeDetector

FRAMES_DIR = Path("Frames")
TRAINING_VIDEOS_DIR = Path("TrainingVideos")
VIDEO_EXTENSIONS = [".MOV", ".wmv", ".mp4"]
RUN_ON = FRAMES_DIR

CANNY_PARAMS = {
    "thresh1": 75,
    "thresh2": 79,
    "blurKernel": 4,
    "l2grad": False,
}

TILE_SIZE = (620, 465)
GAP = 0
MAX_THRESHOLD = 255
MAX_BLUR_KERNEL = 20


def parseFrameIndex(path):
    match = re.fullmatch(r"frame_(\d+)\.png", path.name)
    if match is None:
        return None
    return int(match.group(1))


def iterVideoDirs():
    for videoDir in sorted(FRAMES_DIR.iterdir(), key=lambda path: path.name.lower()):
        if videoDir.is_dir():
            yield videoDir


def pathToFrameDir(path):
    path = Path(path)
    if path.parent == FRAMES_DIR and path.is_dir():
        return path
    return FRAMES_DIR / path.stem


def iterVideoPathsInDirectory(directory):
    for path in sorted(Path(directory).iterdir(), key=lambda item: item.name.lower()):
        if path.suffix in VIDEO_EXTENSIONS:
            yield path


def iterRunOnVideoDirs(runOn):
    if isinstance(runOn, (list, tuple, set)):
        for item in runOn:
            yield from iterRunOnVideoDirs(item)
        return

    runOnPath = Path(runOn)
    if runOnPath.is_dir():
        if runOnPath == FRAMES_DIR:
            yield from iterVideoDirs()
            return

        for videoPath in iterVideoPathsInDirectory(runOnPath):
            yield pathToFrameDir(videoPath)
        return

    yield pathToFrameDir(runOnPath)


def fullLPath(videoDir, frameIndex):
    return videoDir / "LAB" / "L" / "full_raw" / f"frame_{frameIndex:06d}.png"


def roiLPath(videoDir, frameIndex):
    return videoDir / "LAB" / "L" / "roi_raw" / f"frame_{frameIndex:06d}.png"


def collectFrameIndices(videoDir):
    frameIndices = set()
    for framePath in (videoDir / "LAB" / "L" / "full_raw").glob("frame_*.png"):
        frameIndex = parseFrameIndex(framePath)
        if frameIndex is not None and roiLPath(videoDir, frameIndex).exists():
            frameIndices.add(frameIndex)
    return sorted(frameIndices)


def readImage(imagePath):
    image = cv.imread(str(imagePath), cv.IMREAD_UNCHANGED)
    if image is None:
        return np.zeros((TILE_SIZE[1], TILE_SIZE[0]), dtype=np.uint8)
    if image.ndim == 3:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    return image


def fitImageToTile(image):
    if image.ndim == 2:
        image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)

    tileWidth, tileHeight = TILE_SIZE
    imageHeight, imageWidth = image.shape[:2]
    scale = min(tileWidth / imageWidth, tileHeight / imageHeight)
    newWidth = max(1, round(imageWidth * scale))
    newHeight = max(1, round(imageHeight * scale))
    resized = cv.resize(image, (newWidth, newHeight), interpolation=cv.INTER_AREA)

    tile = np.zeros((tileHeight, tileWidth, 3), dtype=np.uint8)
    y0 = (tileHeight - newHeight) // 2
    x0 = (tileWidth - newWidth) // 2
    tile[y0 : y0 + newHeight, x0 : x0 + newWidth] = resized
    return tile


def labelTile(tile, label):
    labeled = tile.copy()
    origin = (10, 30)
    font = cv.FONT_HERSHEY_SIMPLEX
    scale = 0.8
    cv.putText(labeled, label, origin, font, scale, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(labeled, label, origin, font, scale, (255, 255, 255), 1, cv.LINE_AA)
    return labeled


def makeCollage(videoDir, frameIndex, cannyParams):
    fullLFrame = readImage(fullLPath(videoDir, frameIndex))
    roiLFrame = readImage(roiLPath(videoDir, frameIndex))
    cannyFrame = cannyEdgeDetector(
        roiLFrame,
        cannyParams["thresh1"],
        cannyParams["thresh2"],
        cannyParams["blurKernel"],
        cannyParams["l2grad"],
    )

    fullTile = labelTile(
        fitImageToTile(fullLFrame), f"{videoDir.name} frame {frameIndex} L"
    )
    cannyLabel = (
        "Canny ROI "
        f"t1={cannyParams['thresh1']} "
        f"t2={cannyParams['thresh2']} "
        f"blur={cannyParams['blurKernel']} "
        f"l2={int(cannyParams['l2grad'])}"
    )
    cannyTile = labelTile(fitImageToTile(cannyFrame), cannyLabel)
    gap = np.zeros((TILE_SIZE[1], GAP, 3), dtype=np.uint8)
    return np.hstack((fullTile, gap, cannyTile))


def getScreenSize():
    try:
        root = tk.Tk()
        root.withdraw()
        screenSize = root.winfo_screenwidth(), root.winfo_screenheight()
        root.destroy()
        return screenSize
    except tk.TclError:
        return 1600, 900


def showVideo(videoDir, cannyParams):
    frameIndices = collectFrameIndices(videoDir)
    if not frameIndices:
        print(
            f"Skipping {videoDir.name}: no matching LAB/L full_raw and roi_raw frames"
        )
        return

    windowName = f"canny_frame_viewer: {videoDir.name}"
    windowFlags = cv.WINDOW_NORMAL
    if hasattr(cv, "WINDOW_GUI_EXPANDED"):
        windowFlags |= cv.WINDOW_GUI_EXPANDED

    state = {"framePosition": 0, **cannyParams}

    def render():
        frameIndex = frameIndices[state["framePosition"]]
        cv.imshow(windowName, makeCollage(videoDir, frameIndex, state))

    def updateFrame(position):
        state["framePosition"] = int(position)
        render()

    def updateThresh1(value):
        state["thresh1"] = int(value)
        cannyParams["thresh1"] = state["thresh1"]
        render()

    def updateThresh2(value):
        state["thresh2"] = int(value)
        cannyParams["thresh2"] = state["thresh2"]
        render()

    def updateBlurKernel(value):
        state["blurKernel"] = int(value)
        cannyParams["blurKernel"] = state["blurKernel"]
        render()

    def updateL2Grad(value):
        state["l2grad"] = bool(value)
        cannyParams["l2grad"] = state["l2grad"]
        render()

    cv.namedWindow(windowName, windowFlags)
    screenWidth, screenHeight = getScreenSize()
    cv.resizeWindow(windowName, screenWidth, screenHeight)
    cv.createTrackbar("frame", windowName, 0, len(frameIndices) - 1, updateFrame)
    cv.createTrackbar(
        "thresh1", windowName, state["thresh1"], MAX_THRESHOLD, updateThresh1
    )
    cv.createTrackbar(
        "thresh2", windowName, state["thresh2"], MAX_THRESHOLD, updateThresh2
    )
    cv.createTrackbar(
        "blurKernel",
        windowName,
        state["blurKernel"],
        MAX_BLUR_KERNEL,
        updateBlurKernel,
    )
    cv.createTrackbar("l2grad", windowName, int(state["l2grad"]), 1, updateL2Grad)
    render()

    while True:
        key = cv.waitKey(30) & 0xFF
        if key == ord("q"):
            break

    cv.destroyWindow(windowName)


def main():
    cannyParams = {
        "thresh1": int(CANNY_PARAMS["thresh1"]),
        "thresh2": int(CANNY_PARAMS["thresh2"]),
        "blurKernel": int(CANNY_PARAMS["blurKernel"]),
        "l2grad": bool(CANNY_PARAMS["l2grad"]),
    }

    for videoDir in iterRunOnVideoDirs(RUN_ON):
        if not videoDir.exists():
            print(f"Skipping {videoDir.name}: frame directory not found")
            continue

        showVideo(videoDir, cannyParams)

    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
