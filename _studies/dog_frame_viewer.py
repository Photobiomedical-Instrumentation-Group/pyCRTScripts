import re
import tkinter as tk
from pathlib import Path

import cv2 as cv
import numpy as np

FRAMES_DIR = Path("Frames")
TRAINING_VIDEOS_DIR = Path("TrainingVideos")
VIDEO_EXTENSIONS = [".MOV", ".wmv", ".mp4"]
RUN_ON = FRAMES_DIR
COLORSPACE = "LAB"
CHANNEL = "L"

DOG_PARAMS = {
    "blurA": 1,
    "blurB": 4,
    "weightA": 1.0,
    "weightB": 1.0,
    "useAbs": False,
}

TILE_SIZE = (620, 465)
GAP = 0
MAX_BLUR = 40
WEIGHT_SCALE = 100
MAX_WEIGHT = 5.0


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


def fullChannelPath(videoDir, frameIndex):
    return (
        videoDir
        / COLORSPACE
        / CHANNEL
        / "full_raw"
        / f"frame_{frameIndex:06d}.png"
    )


def roiChannelPath(videoDir, frameIndex):
    return (
        videoDir
        / COLORSPACE
        / CHANNEL
        / "roi_raw"
        / f"frame_{frameIndex:06d}.png"
    )


def collectFrameIndices(videoDir):
    frameIndices = set()
    for framePath in (videoDir / COLORSPACE / CHANNEL / "full_raw").glob("frame_*.png"):
        frameIndex = parseFrameIndex(framePath)
        if frameIndex is not None and roiChannelPath(videoDir, frameIndex).exists():
            frameIndices.add(frameIndex)
    return sorted(frameIndices)


def readImage(imagePath):
    image = cv.imread(str(imagePath), cv.IMREAD_UNCHANGED)
    if image is None:
        return np.zeros((TILE_SIZE[1], TILE_SIZE[0]), dtype=np.uint8)
    if image.ndim == 3:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    return image


def minMaxNormalize(frame):
    frame = frame.astype(np.float32)
    frameMin = np.min(frame)
    frameMax = np.max(frame)
    frameRange = frameMax - frameMin
    if frameRange <= 0:
        return np.zeros_like(frame, dtype=np.float32)
    return (frame - frameMin) / frameRange


def applyGaussianBlur(frame, blurKernel):
    if blurKernel <= 0:
        return frame
    kernel = (2 * int(blurKernel)) + 1
    return cv.GaussianBlur(frame, (kernel, kernel), 0)


def dogEdgeDetector(frame, blurA, blurB, weightA, weightB, useAbs):
    frame = minMaxNormalize(frame)
    frameA = frame.copy()
    frameB = frame.copy()
    frameA = applyGaussianBlur(frameA, blurA)
    frameB = applyGaussianBlur(frameB, blurB)
    dogFrame = (float(weightA) * frameA) - (float(weightB) * frameB)
    if useAbs:
        dogFrame = np.abs(dogFrame)
    else:
        dogFrame = minMaxNormalize(dogFrame)
    return np.uint8(np.round(255 * dogFrame))


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


def weightToTrackbar(weight):
    return int(round(float(weight) * WEIGHT_SCALE))


def trackbarToWeight(value):
    return int(value) / WEIGHT_SCALE


def makeCollage(videoDir, frameIndex, dogParams):
    fullFrame = readImage(fullChannelPath(videoDir, frameIndex))
    roiFrame = readImage(roiChannelPath(videoDir, frameIndex))
    dogFrame = dogEdgeDetector(
        roiFrame,
        dogParams["blurA"],
        dogParams["blurB"],
        dogParams["weightA"],
        dogParams["weightB"],
        dogParams["useAbs"],
    )

    fullTile = labelTile(
        fitImageToTile(fullFrame),
        f"{videoDir.name} frame {frameIndex} {COLORSPACE}/{CHANNEL}",
    )
    dogLabel = (
        "DoG ROI "
        f"blurA={dogParams['blurA']} "
        f"blurB={dogParams['blurB']} "
        f"weightA={dogParams['weightA']:.2f} "
        f"weightB={dogParams['weightB']:.2f} "
        f"mode={'abs' if dogParams['useAbs'] else 'minmax'}"
    )
    dogTile = labelTile(fitImageToTile(dogFrame), dogLabel)
    gap = np.zeros((TILE_SIZE[1], GAP, 3), dtype=np.uint8)
    return np.hstack((fullTile, gap, dogTile))


def getScreenSize():
    try:
        root = tk.Tk()
        root.withdraw()
        screenSize = root.winfo_screenwidth(), root.winfo_screenheight()
        root.destroy()
        return screenSize
    except tk.TclError:
        return 1600, 900


def showVideo(videoDir, dogParams):
    frameIndices = collectFrameIndices(videoDir)
    if not frameIndices:
        print(
            f"Skipping {videoDir.name}: "
            f"no matching {COLORSPACE}/{CHANNEL} full_raw and roi_raw frames"
        )
        return

    windowName = f"dog_frame_viewer: {videoDir.name}"
    windowFlags = cv.WINDOW_NORMAL
    if hasattr(cv, "WINDOW_GUI_EXPANDED"):
        windowFlags |= cv.WINDOW_GUI_EXPANDED

    state = {"framePosition": 0, **dogParams}

    def render():
        frameIndex = frameIndices[state["framePosition"]]
        cv.imshow(windowName, makeCollage(videoDir, frameIndex, state))

    def updateFrame(position):
        state["framePosition"] = int(position)
        render()

    def updateBlurA(value):
        state["blurA"] = int(value)
        dogParams["blurA"] = state["blurA"]
        render()

    def updateBlurB(value):
        state["blurB"] = int(value)
        dogParams["blurB"] = state["blurB"]
        render()

    def updateWeightA(value):
        state["weightA"] = trackbarToWeight(value)
        dogParams["weightA"] = state["weightA"]
        render()

    def updateWeightB(value):
        state["weightB"] = trackbarToWeight(value)
        dogParams["weightB"] = state["weightB"]
        render()

    def updateUseAbs(value):
        state["useAbs"] = bool(value)
        dogParams["useAbs"] = state["useAbs"]
        render()

    cv.namedWindow(windowName, windowFlags)
    screenWidth, screenHeight = getScreenSize()
    cv.resizeWindow(windowName, screenWidth, screenHeight)
    cv.createTrackbar("frame", windowName, 0, len(frameIndices) - 1, updateFrame)
    cv.createTrackbar("blurA", windowName, state["blurA"], MAX_BLUR, updateBlurA)
    cv.createTrackbar("blurB", windowName, state["blurB"], MAX_BLUR, updateBlurB)
    cv.createTrackbar(
        "weightA",
        windowName,
        weightToTrackbar(state["weightA"]),
        weightToTrackbar(MAX_WEIGHT),
        updateWeightA,
    )
    cv.createTrackbar(
        "weightB",
        windowName,
        weightToTrackbar(state["weightB"]),
        weightToTrackbar(MAX_WEIGHT),
        updateWeightB,
    )
    cv.createTrackbar("useAbs", windowName, int(state["useAbs"]), 1, updateUseAbs)
    render()

    while True:
        key = cv.waitKey(30) & 0xFF
        if key == ord("q"):
            break

    cv.destroyWindow(windowName)


def main():
    dogParams = {
        "blurA": int(DOG_PARAMS["blurA"]),
        "blurB": int(DOG_PARAMS["blurB"]),
        "weightA": float(DOG_PARAMS["weightA"]),
        "weightB": float(DOG_PARAMS["weightB"]),
        "useAbs": bool(DOG_PARAMS["useAbs"]),
    }

    for videoDir in iterRunOnVideoDirs(RUN_ON):
        if not videoDir.exists():
            print(f"Skipping {videoDir.name}: frame directory not found")
            continue

        showVideo(videoDir, dogParams)

    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
