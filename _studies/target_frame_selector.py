import json
import re
import tkinter as tk
from pathlib import Path

import cv2 as cv
import numpy as np
import tomllib

FRAMES_DIR = Path("Frames")
TRAINING_VIDEOS_DIR = Path("TrainingVideos")
VIDEOS_DIR_LIST = [
    Path("/home/eduardo/Data/miscVideos"),
    Path("/home/eduardo/Data/raquelMasters1"),
    Path("/home/eduardo/Data/raquelMasters2"),
    Path("/home/eduardo/Data/raquelMasters3"),
    Path("/home/eduardo/Data/yutaoNewEquipment"),
    Path("/home/eduardo/Data/yutaoPostNew"),
    # Path("TrainingVideos"),
]
VIDEO_EXTENSIONS = [".MOV", ".wmv", ".mp4"]
VIDEO_EXTENSIONS_LOWER = {suffix.lower() for suffix in VIDEO_EXTENSIONS}
FRAME_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")
TARGETS_PATH = Path("target_frames.toml")
SAVE_TARGETS = True
SKIP_EXISTING_TARGETS = False
# RUN_ON = FRAMES_DIR
RUN_ON = VIDEOS_DIR_LIST

COLORSPACE_ORDER = ("LAB", "BGR")
CHANNEL_ORDER = {
    "LAB": ("A",),
    "BGR": ("G",),
}
MODE_ORDER = ("full_minmax",)
ABORTED = object()

TILE_SIZE = (420, 315)
GAP = 0


def parseFrameIndex(path):
    match = re.fullmatch(r"frame_(\d+)\.(?:png|jpe?g)", path.name, re.IGNORECASE)
    if match is None:
        return None
    return int(match.group(1))


def findVideoName(videoStem):
    for videoDir in (TRAINING_VIDEOS_DIR, *VIDEOS_DIR_LIST):
        for suffix in VIDEO_EXTENSIONS:
            videoPath = videoDir / f"{videoStem}{suffix}"
            if videoPath.exists():
                return videoPath.name
    return videoStem


def loadTargetFrames(targetsPath=None):
    if targetsPath is None:
        targetsPath = TARGETS_PATH
    targetsPath = Path(targetsPath)
    if not targetsPath.exists():
        return {}

    with targetsPath.open("rb") as file:
        targetConfig = tomllib.load(file)
    return {
        videoName: int(frameIndex)
        for videoName, frameIndex in targetConfig.get("targets", {}).items()
    }


def saveTargetFrames(targets, targetsPath=None):
    if targetsPath is None:
        targetsPath = TARGETS_PATH
    targetsPath = Path(targetsPath)
    targetsPath.parent.mkdir(parents=True, exist_ok=True)
    lines = ["[targets]"]
    lines.extend(
        f"{json.dumps(videoName)} = {int(frameIndex)}"
        for videoName, frameIndex in targets.items()
    )

    temporaryPath = targetsPath.with_name(f"{targetsPath.name}.tmp")
    temporaryPath.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporaryPath.replace(targetsPath)


def updateTargetFrame(targets, videoName, frameIndex, targetsPath=None):
    targets[videoName] = int(frameIndex)
    saveTargetFrames(targets, targetsPath)


def iterVideoDirs():
    if not FRAMES_DIR.is_dir():
        return
    for videoDir in sorted(FRAMES_DIR.iterdir(), key=lambda path: path.name.lower()):
        if videoDir.is_dir():
            yield videoDir


def isFramesRoot(path):
    return Path(path).resolve() == FRAMES_DIR.resolve()


def isFrameVideoDir(path):
    path = Path(path)
    return path.is_dir() and path.parent.resolve() == FRAMES_DIR.resolve()


def pathToFrameDir(path):
    path = Path(path)
    if isFrameVideoDir(path):
        return path
    return FRAMES_DIR / path.stem


def iterVideoPathsInDirectory(directory):
    for path in sorted(Path(directory).iterdir(), key=lambda item: item.name.lower()):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS_LOWER:
            yield path


def iterRunOnVideoDirs(runOn):
    if isinstance(runOn, (list, tuple, set)):
        for item in runOn:
            yield from iterRunOnVideoDirs(item)
        return

    runOnPath = Path(runOn)
    if runOnPath.is_dir():
        if isFramesRoot(runOnPath):
            yield from iterVideoDirs()
            return

        if isFrameVideoDir(runOnPath):
            yield runOnPath
            return

        for videoPath in iterVideoPathsInDirectory(runOnPath):
            yield pathToFrameDir(videoPath)
        return

    yield pathToFrameDir(runOnPath)


def collectFrameIndices(videoDir):
    frameIndices = set()
    for framePath in videoDir.rglob("frame_*"):
        if framePath.suffix.lower() not in FRAME_IMAGE_EXTENSIONS:
            continue
        frameIndex = parseFrameIndex(framePath)
        if frameIndex is not None:
            frameIndices.add(frameIndex)
    return sorted(frameIndices)


def imagePathFor(videoDir, colorspace, channel, mode, frameIndex):
    imageDir = videoDir / colorspace / channel / mode
    for extension in FRAME_IMAGE_EXTENSIONS:
        imagePath = imageDir / f"frame_{frameIndex:06d}{extension}"
        if imagePath.exists():
            return imagePath
    return imageDir / f"frame_{frameIndex:06d}.png"


def readTile(imagePath):
    if not imagePath.exists():
        return np.zeros((TILE_SIZE[1], TILE_SIZE[0], 3), dtype=np.uint8)

    image = cv.imread(str(imagePath), cv.IMREAD_UNCHANGED)
    if image is None:
        return np.zeros((TILE_SIZE[1], TILE_SIZE[0], 3), dtype=np.uint8)

    if image.ndim == 2:
        image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = cv.cvtColor(image, cv.COLOR_BGRA2BGR)

    return fitImageToTile(image)


def fitImageToTile(image):
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
    origin = (8, 24)
    font = cv.FONT_HERSHEY_SIMPLEX
    scale = 0.65
    cv.putText(
        labeled,
        label,
        origin,
        font,
        scale,
        (0, 0, 0),
        3,
        cv.LINE_AA,
    )
    cv.putText(
        labeled,
        label,
        origin,
        font,
        scale,
        (255, 255, 255),
        1,
        cv.LINE_AA,
    )
    return labeled


def hstackWithGap(images):
    if not images:
        return np.zeros((TILE_SIZE[1], TILE_SIZE[0], 3), dtype=np.uint8)

    gap = np.zeros((images[0].shape[0], GAP, 3), dtype=np.uint8)
    row = images[0]
    for image in images[1:]:
        row = np.hstack((row, gap, image))
    return row


def vstackWithGap(images):
    if not images:
        return np.zeros((TILE_SIZE[1], TILE_SIZE[0], 3), dtype=np.uint8)

    gap = np.zeros((GAP, images[0].shape[1], 3), dtype=np.uint8)
    collage = images[0]
    for image in images[1:]:
        collage = np.vstack((collage, gap, image))
    return collage


def makeCollage(videoDir, frameIndex):
    tiles = []
    for colorspace in COLORSPACE_ORDER:
        if not (videoDir / colorspace).is_dir():
            continue

        channelOrder = CHANNEL_ORDER.get(colorspace, ())
        for channel in channelOrder:
            channelDir = videoDir / colorspace / channel
            if not channelDir.is_dir():
                continue

            mode = MODE_ORDER[0]
            imagePath = imagePathFor(videoDir, colorspace, channel, mode, frameIndex)
            tile = readTile(imagePath)
            tiles.append(
                labelTile(tile, f"{videoDir.name} {frameIndex} {colorspace}/{channel}")
            )

    if not tiles:
        return np.zeros((TILE_SIZE[1], TILE_SIZE[0], 3), dtype=np.uint8)

    rows = [hstackWithGap(tiles[i : i + 3]) for i in range(0, len(tiles), 3)]
    return vstackWithGap(rows)


def getScreenSize():
    try:
        root = tk.Tk()
        root.withdraw()
        screenSize = root.winfo_screenwidth(), root.winfo_screenheight()
        root.destroy()
        return screenSize
    except tk.TclError:
        return 1600, 900


def nearestFramePosition(frameIndices, targetFrameIndex):
    if targetFrameIndex is None:
        return 0

    frameIndicesArr = np.asarray(frameIndices)
    return int(np.argmin(np.abs(frameIndicesArr - int(targetFrameIndex))))


def selectTargetFrame(videoDir, initialFrameIndex=None):
    frameIndices = collectFrameIndices(videoDir)
    if not frameIndices:
        print(f"Skipping {videoDir.name}: no frame images found")
        return None

    windowName = f"target_frame_selector: {videoDir.name}"
    windowFlags = cv.WINDOW_NORMAL
    if hasattr(cv, "WINDOW_GUI_EXPANDED"):
        windowFlags |= cv.WINDOW_GUI_EXPANDED

    initialPosition = nearestFramePosition(frameIndices, initialFrameIndex)
    selectedPosition = {"value": initialPosition}

    def render(position):
        selectedPosition["value"] = int(position)
        frameIndex = frameIndices[selectedPosition["value"]]
        collage = makeCollage(videoDir, frameIndex)
        cv.imshow(windowName, collage)

    cv.namedWindow(windowName, windowFlags)
    screenWidth, screenHeight = getScreenSize()
    cv.resizeWindow(windowName, screenWidth, screenHeight)
    cv.createTrackbar(
        "frame", windowName, initialPosition, len(frameIndices) - 1, render
    )
    render(initialPosition)

    while True:
        key = cv.waitKey(30) & 0xFF
        if key == ord("q"):
            break
        if key == ord("a"):
            cv.destroyWindow(windowName)
            return ABORTED

    cv.destroyWindow(windowName)
    return frameIndices[selectedPosition["value"]]


def main():
    targets = loadTargetFrames()

    for videoDir in iterRunOnVideoDirs(RUN_ON):
        if not videoDir.exists():
            print(f"Skipping {videoDir.name}: frame directory not found")
            continue

        videoName = findVideoName(videoDir.name)
        if SKIP_EXISTING_TARGETS and videoName in targets:
            print(f"Skipping {videoName}: target frame already exists")
            continue

        initialFrameIndex = None
        if not SKIP_EXISTING_TARGETS and videoName in targets:
            initialFrameIndex = targets[videoName]
            print(
                f"Verifying {videoName}: existing target frame {initialFrameIndex}",
                flush=True,
            )

        selectedFrameIndex = selectTargetFrame(videoDir, initialFrameIndex)
        if selectedFrameIndex is ABORTED:
            saveTargetFrames(targets)
            print(f"aborted on {videoName}", flush=True)
            break

        if selectedFrameIndex is None:
            continue

        print(f'"{videoName}": {selectedFrameIndex},', flush=True)
        if SAVE_TARGETS:
            updateTargetFrame(targets, videoName, selectedFrameIndex)
            print(f"Saved {videoName} to {TARGETS_PATH}", flush=True)

    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
