import os
import re
import tkinter as tk
from pathlib import Path

import cv2 as cv
import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-pycrt")

from funcs import laplacianEdgeDetector

FRAMES_DIR = Path("Frames")
TRAINING_VIDEOS_DIR = Path("TrainingVideos")
VIDEO_EXTENSIONS = [".MOV", ".wmv", ".mp4"]
RUN_ON = FRAMES_DIR

LAPLACIAN_PARAMS = {
    "ksize": 1,
    "blurKernel": 1,
    "scale": 3,
    "delta": 0,
}

TILE_SIZE = (620, 465)
GAP = 0
MAX_KSIZE = 3
MAX_BLUR_KERNEL = 20
MAX_SCALE = 20
MIN_DELTA = -100
MAX_DELTA = 100


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


def deltaToTrackbar(delta):
    return int(delta) - MIN_DELTA


def trackbarToDelta(value):
    return int(value) + MIN_DELTA


def makeCollage(videoDir, frameIndex, laplacianParams):
    fullLFrame = readImage(fullLPath(videoDir, frameIndex))
    roiLFrame = readImage(roiLPath(videoDir, frameIndex))
    laplacianFrame = laplacianEdgeDetector(
        roiLFrame,
        laplacianParams["ksize"],
        laplacianParams["blurKernel"],
        laplacianParams["scale"],
        laplacianParams["delta"],
    )
    laplacianFrame = cv.convertScaleAbs(laplacianFrame)

    fullTile = labelTile(fitImageToTile(fullLFrame), f"{videoDir.name} frame {frameIndex} L")
    laplacianLabel = (
        "Laplacian ROI "
        f"ksize={(2 * laplacianParams['ksize']) + 1} "
        f"blur={laplacianParams['blurKernel']} "
        f"scale={laplacianParams['scale']} "
        f"delta={laplacianParams['delta']}"
    )
    laplacianTile = labelTile(fitImageToTile(laplacianFrame), laplacianLabel)
    gap = np.zeros((TILE_SIZE[1], GAP, 3), dtype=np.uint8)
    return np.hstack((fullTile, gap, laplacianTile))


def getScreenSize():
    try:
        root = tk.Tk()
        root.withdraw()
        screenSize = root.winfo_screenwidth(), root.winfo_screenheight()
        root.destroy()
        return screenSize
    except tk.TclError:
        return 1600, 900


def showVideo(videoDir, laplacianParams):
    frameIndices = collectFrameIndices(videoDir)
    if not frameIndices:
        print(f"Skipping {videoDir.name}: no matching LAB/L full_raw and roi_raw frames")
        return

    windowName = f"laplacian_frame_viewer: {videoDir.name}"
    windowFlags = cv.WINDOW_NORMAL
    if hasattr(cv, "WINDOW_GUI_EXPANDED"):
        windowFlags |= cv.WINDOW_GUI_EXPANDED

    state = {"framePosition": 0, **laplacianParams}

    def render():
        frameIndex = frameIndices[state["framePosition"]]
        cv.imshow(windowName, makeCollage(videoDir, frameIndex, state))

    def updateFrame(position):
        state["framePosition"] = int(position)
        render()

    def updateKsize(value):
        state["ksize"] = int(value)
        laplacianParams["ksize"] = state["ksize"]
        render()

    def updateBlurKernel(value):
        state["blurKernel"] = int(value)
        laplacianParams["blurKernel"] = state["blurKernel"]
        render()

    def updateScale(value):
        state["scale"] = int(value)
        laplacianParams["scale"] = state["scale"]
        render()

    def updateDelta(value):
        state["delta"] = trackbarToDelta(value)
        laplacianParams["delta"] = state["delta"]
        render()

    cv.namedWindow(windowName, windowFlags)
    screenWidth, screenHeight = getScreenSize()
    cv.resizeWindow(windowName, screenWidth, screenHeight)
    cv.createTrackbar("frame", windowName, 0, len(frameIndices) - 1, updateFrame)
    cv.createTrackbar("ksize", windowName, state["ksize"], MAX_KSIZE, updateKsize)
    cv.createTrackbar(
        "blurKernel",
        windowName,
        state["blurKernel"],
        MAX_BLUR_KERNEL,
        updateBlurKernel,
    )
    cv.createTrackbar("scale", windowName, state["scale"], MAX_SCALE, updateScale)
    cv.createTrackbar(
        "delta",
        windowName,
        deltaToTrackbar(state["delta"]),
        MAX_DELTA - MIN_DELTA,
        updateDelta,
    )
    render()

    while True:
        key = cv.waitKey(30) & 0xFF
        if key == ord("q"):
            break

    cv.destroyWindow(windowName)


def main():
    laplacianParams = {
        "ksize": int(LAPLACIAN_PARAMS["ksize"]),
        "blurKernel": int(LAPLACIAN_PARAMS["blurKernel"]),
        "scale": int(LAPLACIAN_PARAMS["scale"]),
        "delta": int(LAPLACIAN_PARAMS["delta"]),
    }

    for videoDir in iterRunOnVideoDirs(RUN_ON):
        if not videoDir.exists():
            print(f"Skipping {videoDir.name}: frame directory not found")
            continue

        showVideo(videoDir, laplacianParams)

    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
