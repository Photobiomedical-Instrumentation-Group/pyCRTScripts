import json
from pathlib import Path

import cv2 as cv
import numpy as np
import tomllib
from funcs import loadVideoRoi
from matplotlib import pyplot as plt
from pyCRT.frameOperations import cropFrame, drawRoi, rescaleFrame
from pyCRT.videoReading import frameReader, videoCapture

plt.style.use("bmh")

ROIS_PATH = "rois_full.toml"
TENTATIVE_TIMESTAMPS_PATH = Path("tentative_timestamps.toml")
VIDEOS_DIR_LIST = [
    # Path("/home/eduardo/Data/miscVideos"),
    # Path("/home/eduardo/Data/raquelMasters1"),
    # Path("/home/eduardo/Data/raquelMasters2"),
    # Path("/home/eduardo/Data/raquelMasters3"),
    # Path("/home/eduardo/Data/yutaoNewEquipment"),
    Path("/home/eduardo/Data/yutaoPostNew"),
    # Path("TrainingVideos"),
]
DEST_DIR = Path("/home/eduardo/Code/Python/pyCRTScripts/_studies/Frames")
VIDEO_EXTENSIONS = [".MOV", ".wmv", ".mp4"]
VIDEO_EXTENSIONS_LOWER = {suffix.lower() for suffix in VIDEO_EXTENSIONS}

# Accepts a video file, a directory of videos, or a nested list of either.
RUN_ON = VIDEOS_DIR_LIST

TENTATIVE_TIMESTAMPS = {}

FRAME_WINDOW = 60
RESCALE_FRAMES = True
RESCALE_FACTOR = 0.5
TENTATIVE_SELECTION_RESCALE_FACTOR = 0.5
TENTATIVE_SELECTION_WAITKEY_MS = 2
SKIP_EXISTING = True
PNG_COMPRESSION = 9
SAVE_FRAMES_AS_JPEG = True
JPEG_QUALITY = 50
VERBOSE = True
# When enabled, only exports LAB A and BGR G full-frame min-max images and plots.
ONLY_SAVE_FULL_MINMAX_BGR_G_AND_LAB_A = True

COLORSPACES = {
    "LAB": (lambda frame: cv.cvtColor(frame, cv.COLOR_BGR2LAB), ("L", "A", "B")),
    "YCrCb": (
        lambda frame: cv.cvtColor(frame, cv.COLOR_BGR2YCrCb),
        ("Y", "Cr", "Cb"),
    ),
    "BGR": (lambda frame: frame, ("B", "G", "R")),
}

RAW_CHANNEL_RANGES = {
    ("LAB", "L"): (0, 100),
    ("LAB", "A"): (-127, 127),
    ("LAB", "B"): (-127, 127),
    ("YCrCb", "Y"): (0, 1),
    ("YCrCb", "Cr"): (0, 1),
    ("YCrCb", "Cb"): (0, 1),
    ("BGR", "B"): (0, 1),
    ("BGR", "G"): (0, 1),
    ("BGR", "R"): (0, 1),
}

CHANNEL_COLORS = {
    ("LAB", "L"): "black",
    ("LAB", "A"): "green",
    ("LAB", "B"): "blue",
    ("YCrCb", "Y"): "black",
    ("YCrCb", "Cr"): "red",
    ("YCrCb", "Cb"): "blue",
    ("BGR", "B"): "blue",
    ("BGR", "G"): "green",
    ("BGR", "R"): "red",
}

EXPORT_VARIANTS = {
    "roi_minmax": {"crop": True, "normalize": True},
    "roi_raw": {"crop": True, "normalize": False},
    "full_minmax": {"crop": False, "normalize": True},
    "full_raw": {"crop": False, "normalize": False},
}


def loadTentativeTimestamps():
    if not TENTATIVE_TIMESTAMPS_PATH.exists():
        return {}

    with TENTATIVE_TIMESTAMPS_PATH.open("rb") as file:
        config = tomllib.load(file)

    timestamps = config.get("timestamps", {})
    return {
        str(videoName): float(timestamp) for videoName, timestamp in timestamps.items()
    }


def saveTentativeTimestamps(timestamps):
    lines = ["[timestamps]"]
    for videoName in sorted(timestamps, key=str.lower):
        lines.append(f"{json.dumps(videoName)} = {timestamps[videoName]:.6f}")

    temporaryPath = TENTATIVE_TIMESTAMPS_PATH.with_suffix(".tmp")
    temporaryPath.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporaryPath.replace(TENTATIVE_TIMESTAMPS_PATH)


def iterRunOnVideoPaths(runOn):
    if isinstance(runOn, (list, tuple, set)):
        for item in runOn:
            yield from iterRunOnVideoPaths(item)
        return

    runOnPath = Path(runOn)
    if runOnPath.is_dir():
        for videoPath in sorted(
            runOnPath.iterdir(), key=lambda path: path.name.lower()
        ):
            if (
                videoPath.is_file()
                and videoPath.suffix.lower() in VIDEO_EXTENSIONS_LOWER
            ):
                yield videoPath
        return

    yield runOnPath


def iterSelectedOutputs():
    if ONLY_SAVE_FULL_MINMAX_BGR_G_AND_LAB_A:
        for colorspaceName, channelName in (("LAB", "A"), ("BGR", "G")):
            yield (
                colorspaceName,
                channelName,
                "full_minmax",
                EXPORT_VARIANTS["full_minmax"],
            )
        return

    for colorspaceName, (_, channelNames) in COLORSPACES.items():
        for channelName in channelNames:
            for variantName, variantConfig in EXPORT_VARIANTS.items():
                yield colorspaceName, channelName, variantName, variantConfig


def minMaxNormalizeFrame(frame):
    frame = frame.astype(np.float32)
    frameMin = frame.min()
    frameMax = frame.max()
    if frameMax == frameMin:
        return np.zeros_like(frame, dtype=np.uint8)
    return np.uint8(np.round(255 * (frame - frameMin) / (frameMax - frameMin)))


def frameImageExtension():
    return ".jpg" if SAVE_FRAMES_AS_JPEG else ".png"


def frameImageWriteParams():
    if SAVE_FRAMES_AS_JPEG:
        return [cv.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
    return [cv.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION]


def rawChannelToUint8(channelFrame, colorspaceName, channelName):
    lowerBound, upperBound = RAW_CHANNEL_RANGES[colorspaceName, channelName]
    channelFrame = channelFrame.astype(np.float32)
    channelFrame = (channelFrame - lowerBound) / (upperBound - lowerBound)
    return np.uint8(np.round(255 * np.clip(channelFrame, 0, 1)))


def convertChannelForVariant(
    channelFrame, roi, colorspaceName, channelName, variantConfig
):
    outputFrame = channelFrame
    if variantConfig["crop"]:
        outputFrame = cropFrame(outputFrame, roi)
    if variantConfig["normalize"]:
        return minMaxNormalizeFrame(outputFrame)
    return rawChannelToUint8(outputFrame, colorspaceName, channelName)


def normalizeTimestamps(timestamps):
    if isinstance(timestamps, (list, tuple, set)):
        return tuple(float(timestamp) for timestamp in timestamps)
    return (float(timestamps),)


def frameIndicesAroundTimestamps(timestamps, timeScdsArr):
    frameIndices = set()
    for timestamp in normalizeTimestamps(timestamps):
        centerIndex = int(np.argmin(np.abs(timeScdsArr - timestamp)))
        lowerIndex = max(0, centerIndex - FRAME_WINDOW)
        upperIndex = min(len(timeScdsArr) - 1, centerIndex + FRAME_WINDOW)
        frameIndices.update(range(lowerIndex, upperIndex + 1))
    return frameIndices


def readVideoTimes(videoPath):
    timeList = []
    with videoCapture(str(videoPath)) as cap:
        for frame in frameReader(cap):
            timeList.append(cap.get(cv.CAP_PROP_POS_MSEC) / 1000.0)

    return np.array(timeList)


def iterOutputPaths(videoStem, frameIndex):
    for colorspaceName, channelName, variantName, _ in iterSelectedOutputs():
        outputDir = DEST_DIR / videoStem / colorspaceName / channelName / variantName
        yield outputDir / f"frame_{frameIndex:06d}{frameImageExtension()}"


def iterPlotPaths(videoStem):
    for colorspaceName, channelName, variantName, _ in iterSelectedOutputs():
        outputDir = DEST_DIR / videoStem / colorspaceName / channelName / variantName
        yield outputDir / "avg_intensity.png"


def videoOutputsExistFast(videoStem):
    videoOutputDir = DEST_DIR / videoStem
    if not videoOutputDir.exists():
        return False

    for colorspaceName, channelName, variantName, _ in iterSelectedOutputs():
        outputDir = DEST_DIR / videoStem / colorspaceName / channelName / variantName
        if not outputDir.exists():
            return False
        if not (outputDir / "avg_intensity.png").exists():
            return False
        if not any(outputDir.glob(f"frame_*{frameImageExtension()}")):
            return False

    return True


def videoOutputsExist(videoPath, timestamps):
    timeScdsArr = readVideoTimes(videoPath)
    frameIndices = frameIndicesAroundTimestamps(timestamps, timeScdsArr)

    return all(
        outputPath.exists()
        for frameIndex in frameIndices
        for outputPath in iterOutputPaths(videoPath.stem, frameIndex)
    ) and all(plotPath.exists() for plotPath in iterPlotPaths(videoPath.stem))


def calcVariantRoiAverage(outputFrame, roi, variantConfig):
    if variantConfig["crop"]:
        return np.mean(outputFrame)
    return np.mean(cropFrame(outputFrame, roi))


def drawRoiOnFullFrame(outputFrame, roi, variantConfig):
    if variantConfig["crop"]:
        return outputFrame

    if outputFrame.ndim == 2:
        outputFrame = cv.cvtColor(outputFrame, cv.COLOR_GRAY2BGR)
    else:
        outputFrame = outputFrame.copy()

    return drawRoi(outputFrame, roi)


def processFrameVariants(frame, roi, videoStem, frameIndex, saveImages=False):
    intensityByOutput = {}
    for colorspaceName, (convertFrame, channelNames) in COLORSPACES.items():
        selectedOutputs = [
            (channelName, variantName, variantConfig)
            for (
                selectedColorspaceName,
                channelName,
                variantName,
                variantConfig,
            ) in iterSelectedOutputs()
            if selectedColorspaceName == colorspaceName
        ]
        if not selectedOutputs:
            continue

        convertedFrame = convertFrame(frame)
        if RESCALE_FRAMES:
            convertedFrame = rescaleFrame(convertedFrame, RESCALE_FACTOR)

        channels = cv.split(convertedFrame)

        for channelName, channelFrame in zip(channelNames, channels):
            baseOutputDir = DEST_DIR / videoStem / colorspaceName / channelName
            for (
                selectedChannelName,
                variantName,
                variantConfig,
            ) in selectedOutputs:
                if selectedChannelName != channelName:
                    continue
                outputFrame = convertChannelForVariant(
                    channelFrame,
                    roi,
                    colorspaceName,
                    channelName,
                    variantConfig,
                )
                intensityByOutput[colorspaceName, channelName, variantName] = (
                    calcVariantRoiAverage(outputFrame, roi, variantConfig)
                )

                if saveImages:
                    outputDir = baseOutputDir / variantName
                    outputDir.mkdir(parents=True, exist_ok=True)
                    framePath = (
                        outputDir / f"frame_{frameIndex:06d}{frameImageExtension()}"
                    )
                    outputFrame = drawRoiOnFullFrame(
                        outputFrame,
                        roi,
                        variantConfig,
                    )
                    cv.imwrite(
                        str(framePath),
                        outputFrame,
                        frameImageWriteParams(),
                    )

    return intensityByOutput


def saveIntensityPlots(videoStem, timeList, intensityByOutput, savedFrameIndices):
    if not savedFrameIndices:
        return

    savedFrameIndices = sorted(savedFrameIndices)
    markedIndices = (
        savedFrameIndices[0],
        savedFrameIndices[len(savedFrameIndices) // 2],
        savedFrameIndices[-1],
    )

    for (
        colorspaceName,
        channelName,
        variantName,
    ), intensityList in intensityByOutput.items():
        outputDir = DEST_DIR / videoStem / colorspaceName / channelName / variantName
        outputDir.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(6, 3), layout="tight")
        ax.plot(
            timeList,
            intensityList,
            color=CHANNEL_COLORS[colorspaceName, channelName],
            lw=1,
        )
        for label, frameIndex in zip(("first", "middle", "last"), markedIndices):
            if frameIndex < len(timeList):
                timestamp = timeList[frameIndex]
                ax.axvline(
                    timestamp,
                    color="black",
                    ls="--",
                    label=f"{label}: {timestamp:.3f}s, frame {frameIndex}",
                    lw=1,
                )

        ax.set_title(f"{videoStem} {colorspaceName} {channelName} {variantName}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Average ROI intensity")
        ax.legend(loc="best")
        try:
            fig.savefig(
                outputDir / "avg_intensity.png",
                dpi=150,
            )
        finally:
            plt.close(fig)


def saveFramesAroundTimestamps(videoPath, timestamps):
    roi = loadVideoRoi(videoPath, ROIS_PATH)
    timeScdsArr = readVideoTimes(videoPath)
    frameIndices = frameIndicesAroundTimestamps(timestamps, timeScdsArr)

    savedFrameIndices = []
    intensityByOutput = {
        (colorspaceName, channelName, variantName): []
        for colorspaceName, channelName, variantName, _ in iterSelectedOutputs()
    }

    savedCount = 0
    with videoCapture(str(videoPath)) as cap:
        for frameIndex, frame in enumerate(frameReader(cap)):
            if frameIndex not in frameIndices:
                saveImages = False
            else:
                saveImages = True
                savedFrameIndices.append(frameIndex)
                savedCount += 1

            frame = np.clip(frame.astype(np.float32) / 255, 0, 1)
            frameIntensities = processFrameVariants(
                frame,
                roi,
                videoPath.stem,
                frameIndex,
                saveImages=saveImages,
            )
            if saveImages and VERBOSE:
                print(f"saved {videoPath.stem} frame {frameIndex}")
            for outputKey, intensity in frameIntensities.items():
                intensityByOutput[outputKey].append(intensity)

    saveIntensityPlots(
        videoPath.stem,
        timeScdsArr,
        intensityByOutput,
        savedFrameIndices,
    )

    print(f"{videoPath.name}: saved {savedCount} frames to {DEST_DIR}")


def selectTentativeTimestamp(videoPath):
    windowName = f"Select tentative timestamp: {videoPath.name}"
    print(f"Selecting timestamp for {videoPath.name}: press Space to select, q to skip")

    try:
        with videoCapture(str(videoPath)) as cap:
            for frame in frameReader(cap):
                displayFrame = rescaleFrame(
                    frame,
                    TENTATIVE_SELECTION_RESCALE_FACTOR,
                )
                cv.imshow(windowName, displayFrame)
                key = cv.waitKey(TENTATIVE_SELECTION_WAITKEY_MS) & 0xFF
                if key == ord(" "):
                    timestamp = cap.get(cv.CAP_PROP_POS_MSEC) / 1000.0
                    TENTATIVE_TIMESTAMPS[videoPath.name] = timestamp
                    print(f"Selected {videoPath.name}: {timestamp:.3f}s")
                    return timestamp
                if key == ord("q"):
                    print(f"Skipped {videoPath.name}: no timestamp selected")
                    return None
    finally:
        cv.destroyWindow(windowName)

    print(f"Skipped {videoPath.name}: reached end without selecting a timestamp")
    return None


def main():
    global TENTATIVE_TIMESTAMPS

    TENTATIVE_TIMESTAMPS = loadTentativeTimestamps()
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    videoPaths = list(iterRunOnVideoPaths(RUN_ON))

    try:
        for videoPath in videoPaths:
            videoName = videoPath.name
            if not videoPath.exists():
                print(f"Skipping {videoName}: file not found at {videoPath}")
                continue

            if SKIP_EXISTING and videoOutputsExistFast(videoPath.stem):
                print(f"Skipping timestamp selection for {videoName}: output files already exist")
                continue

            if videoName not in TENTATIVE_TIMESTAMPS:
                selectTentativeTimestamp(videoPath)

        for videoPath in videoPaths:
            videoName = videoPath.name
            timestamps = TENTATIVE_TIMESTAMPS.get(videoName)
            print(f"Working on {videoName}...")
            if not videoPath.exists():
                print(f"Skipping {videoName}: file not found at {videoPath}")
                continue

            if timestamps is None:
                print(f"Skipping {videoName}: no tentative timestamp selected")
                continue

            if SKIP_EXISTING:
                if videoOutputsExistFast(videoPath.stem):
                    print(f"Skipping {videoName}: output files already exist")
                    continue

                if videoOutputsExist(videoPath, timestamps):
                    print(f"Skipping {videoName}: output files already exist")
                    continue

            saveFramesAroundTimestamps(videoPath, timestamps)
    finally:
        saveTentativeTimestamps(TENTATIVE_TIMESTAMPS)


if __name__ == "__main__":
    main()
