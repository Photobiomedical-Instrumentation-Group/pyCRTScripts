import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-pycrt")

import cv2 as cv
import numpy as np
from crt_signal_processing import (
    DEFAULT_CRT90_10_BOOTSTRAP_COUNT,
    DEFAULT_CRT90_10_BOOTSTRAP_SEED,
    DEFAULT_CRT90_10_GAUSSIAN_SIGMA_SECONDS,
    DEFAULT_MAX_CRT90_10_FIT_SECONDS,
    calcCRT90_10BootstrapUncertainty,
    calcCRT90_10Gaussian,
)
from funcs import loadVideoRoi
from matplotlib import pyplot as plt
from pyCRT.frameOperations import cropFrame, rescaleFrame
from pyCRT.videoReading import frameReader, videoCapture
from release_frame_processing import (
    cannySumFromLFrame,
    findReleaseIndex,
    frameToLChannelRoi,
    normalizeCannyArr,
    optimizableFindCrtIntervalIndices,
)

plt.style.use("bmh")

ROIS_PATH = "rois_full.toml"
VIDEOS_DIR = Path("/home/eduardo/Code/Python/pyCRTScripts/_studies/TrainingVideos")
PLOTS_PATH = Path("Plots/Colorspaces")

RESCALE_FACTOR = 0.5
MEDIAN_KERNEL_RADIUS = 1
VIDEO_EXTENSIONS = [".MOV", ".wmv", ".mp4"]

RELAX_MAX_GRAD = 1.0
RELAX_MIN_GRAD = -3.0
STRICT_MAX_GRAD = 1.0
STRICT_MIN_GRAD = -0.5
CANNY_PARAMS = {
    "thresh1": 56,
    "thresh2": 85,
    "blurKernel": 6,
    "l2grad": False,
    "cannyPlateau": 0.1,
    "cannySmoothingKernel": 9,
    "cannyGradientSmoothingKernel": 1,
    "strictMinGrad": STRICT_MIN_GRAD,
    "strictMaxGrad": STRICT_MAX_GRAD,
    "relaxMinGrad": RELAX_MIN_GRAD,
    "relaxMaxGrad": RELAX_MAX_GRAD,
    "indexOffset": 0,
}

MAX_CRT90_10_FIT_SECONDS = DEFAULT_MAX_CRT90_10_FIT_SECONDS
CRT90_10_GAUSSIAN_SIGMA_SECONDS = DEFAULT_CRT90_10_GAUSSIAN_SIGMA_SECONDS
CRT90_10_BOOTSTRAP_COUNT = DEFAULT_CRT90_10_BOOTSTRAP_COUNT
CRT90_10_BOOTSTRAP_SEED = DEFAULT_CRT90_10_BOOTSTRAP_SEED

RELEASE_TIMES = {
    "0528_test2.1.mp4": 4.93,
    "0528_test2.2.mp4": 4.83,
    "0528_test3.2.mp4": 4.77,
    "0528_test3.6.mp4": 4.93,
    "0528_test4.2.mp4": 4.8,
    "CRT1.mp4": 12.10,
    "CRT2.mp4": 12.17,
    "CRT3.mp4": 12.17,
    "DSC_0069.MOV": 11.34,
    "P02CR1.wmv": 17.20,
    "P02CR4.wmv": 17.20,
    "P06CR1.wmv": 17.4,
    "P09CR1.wmv": 17.4,
    "P15CR2.wmv": 19.4,
    "P19CR5.wmv": 17.9,
    "P06CR3.wmv": 17.0,
    "P06CR4.wmv": 16.6,
    "P20CR1.wmv": 16.0,
    "P20CR2.wmv": 16.5,
    "P20CR3.wmv": 17.0,
    "P26CR1.wmv": 19.30,
    "P26CR2.wmv": 17.54,
    "P26CR3.wmv": 18.59,
    "P30CR3.wmv": 17.41,
    "P30CR4.wmv": 17.41,
    "P31CR2.wmv": 17.51,
    "P31CR5.wmv": 17.00,
    "Video 142.mp4": 9.36,
    "Video 148.mp4": 8.69,
    "Video 154.mp4": 8.51,
    "experiment 1.wmv": 8.22,
    "experiment 2.wmv": 6.35,
    "experiment 3.mp4": 9.71,
    "experiment 4.wmv": 12.05,
    "v1.mp4": 9.62,
    "video.mp4": 15.60,
}

COLORSPACES = {
    "LAB": (lambda frame: cv.cvtColor(frame, cv.COLOR_BGR2LAB), ("L", "A", "B")),
    "YCrCb": (
        lambda frame: cv.cvtColor(frame, cv.COLOR_BGR2YCrCb),
        ("Y", "Cr", "Cb"),
    ),
    "LUV": (lambda frame: cv.cvtColor(frame, cv.COLOR_BGR2LUV), ("L", "U", "V")),
    "BGR": (lambda frame: frame, ("B", "G", "R")),
}

CHANNEL_COLORS = {
    ("LAB", "L"): "black",
    ("LAB", "A"): "green",
    ("LAB", "B"): "blue",
    ("YCrCb", "Y"): "black",
    ("YCrCb", "Cr"): "red",
    ("YCrCb", "Cb"): "blue",
    ("LUV", "L"): "black",
    ("LUV", "U"): "blue",
    ("LUV", "V"): "red",
    ("BGR", "B"): "blue",
    ("BGR", "G"): "green",
    ("BGR", "R"): "red",
}


def iterTrainingVideos():
    for videoPath in sorted(VIDEOS_DIR.iterdir(), key=lambda path: path.name.lower()):
        if videoPath.suffix in VIDEO_EXTENSIONS:
            yield videoPath


def measureVideo(videoPath):
    roi = loadVideoRoi(videoPath, ROIS_PATH)
    avgIntensLists = {colorspaceName: [] for colorspaceName in COLORSPACES}
    metricList = []
    timeScdsList = []

    with videoCapture(str(videoPath)) as cap:
        for frame in frameReader(cap):
            timeScdsList.append(cap.get(cv.CAP_PROP_POS_MSEC) / 1000.0)

            lFrame = frameToLChannelRoi(
                frame,
                roi,
                medianKernelRadius=MEDIAN_KERNEL_RADIUS,
                rescaleFactor=RESCALE_FACTOR,
            )
            metricList.append(cannySumFromLFrame(lFrame, CANNY_PARAMS))

            floatFrame = np.clip(frame.astype(np.float32) / 255, 0, 1)
            for colorspaceName, (convertFrame, _) in COLORSPACES.items():
                convertedFrame = convertFrame(floatFrame)
                convertedFrame = rescaleFrame(convertedFrame, RESCALE_FACTOR)
                frameRoi = cropFrame(convertedFrame, roi)
                avgIntensLists[colorspaceName].append(np.mean(frameRoi, axis=(0, 1)))

    if not metricList:
        raise ValueError(f"No frames read from {videoPath}.")

    return (
        np.asarray(timeScdsList, dtype=float),
        {
            colorspaceName: np.asarray(avgIntensList)
            for colorspaceName, avgIntensList in avgIntensLists.items()
        },
        normalizeCannyArr(metricList),
    )


def calculateOrientedChannelCRT90_10(
    timeScdsArr,
    channelIntensArr,
    metricArr,
    inverted,
):
    releaseIndex = findReleaseIndex(metricArr, timeScdsArr, CANNY_PARAMS)
    indices = optimizableFindCrtIntervalIndices(
        channelIntensArr,
        timeScdsArr,
        releaseIndex,
        CANNY_PARAMS["strictMinGrad"],
        CANNY_PARAMS["strictMaxGrad"],
        CANNY_PARAMS["relaxMinGrad"],
        CANNY_PARAMS["relaxMaxGrad"],
    )
    crt90_10 = calcCRT90_10Gaussian(
        timeScdsArr,
        channelIntensArr,
        indices,
        maxFitSeconds=MAX_CRT90_10_FIT_SECONDS,
        gaussianSigmaSeconds=CRT90_10_GAUSSIAN_SIGMA_SECONDS,
    )

    try:
        uncertainty = calcCRT90_10BootstrapUncertainty(
            timeScdsArr,
            channelIntensArr,
            indices,
            maxFitSeconds=MAX_CRT90_10_FIT_SECONDS,
            gaussianSigmaSeconds=CRT90_10_GAUSSIAN_SIGMA_SECONDS,
            nBootstraps=CRT90_10_BOOTSTRAP_COUNT,
            randomSeed=CRT90_10_BOOTSTRAP_SEED,
        )
    except Exception as err:
        uncertainty = None
        uncertaintyError = err
    else:
        uncertaintyError = None

    return {
        "indices": indices,
        "crt90_10": crt90_10,
        "uncertainty": uncertainty,
        "uncertaintyError": uncertaintyError,
        "inverted": inverted,
    }


def calculateChannelCRT90_10(timeScdsArr, channelIntensArr, metricArr):
    try:
        return calculateOrientedChannelCRT90_10(
            timeScdsArr,
            channelIntensArr,
            metricArr,
            inverted=False,
        )
    except Exception as directError:
        invertedChannel = 1.0 - channelIntensArr
        try:
            return calculateOrientedChannelCRT90_10(
                timeScdsArr,
                invertedChannel,
                metricArr,
                inverted=True,
            )
        except Exception as invertedError:
            raise ValueError(
                f"raw signal failed ({directError}); "
                f"inverted signal failed ({invertedError})"
            ) from invertedError


def calculateColorspaceCRT90_10(
    videoPath,
    timeScdsArr,
    avgIntensArr,
    metricArr,
    channelNames,
):
    results = []
    for channelIndex, channelName in enumerate(channelNames):
        try:
            result = calculateChannelCRT90_10(
                timeScdsArr,
                avgIntensArr[:, channelIndex],
                metricArr,
            )
        except Exception as err:
            print(
                f"{videoPath.name} {channelName}: CRT90_10 failed: "
                f"{type(err).__name__}: {err}"
            )
            result = {"error": err}
        else:
            uncertaintyError = result["uncertaintyError"]
            if uncertaintyError is not None:
                print(
                    f"{videoPath.name} {channelName}: CRT90_10 uncertainty "
                    f"failed: {type(uncertaintyError).__name__}: "
                    f"{uncertaintyError}"
                )
        results.append(result)
    return results


def crt90_10DecayLabel(result):
    uncertainty = result["uncertainty"]
    if uncertainty is None:
        decayLabel = f"{result['crt90_10']['crt90_10']:.3f}s"
    else:
        ci95Low, ci95High = uncertainty["crt90_10_ci95"]
        ci95HalfWidth = (ci95High - ci95Low) / 2
        decayLabel = f"{uncertainty['crt90_10_mean']:.3f}+-{ci95HalfWidth:.3f}s"

    return decayLabel


def plotIntensitiesColorspace(
    videoPath,
    timeScdsArr,
    avgIntensArr,
    releaseIndex,
    channelResults,
    colorspaceName,
    channelNames,
    filenameSuffix="",
    signalStartToEnd=False,
):
    outputDir = PLOTS_PATH / videoPath.stem
    outputDir.mkdir(parents=True, exist_ok=True)

    fullTimeScdsArr = timeScdsArr
    releaseTime = fullTimeScdsArr[releaseIndex] if releaseIndex is not None else None
    if signalStartToEnd:
        filenameSuffix = f"{filenameSuffix}-signalStartToEnd"

    if len(fullTimeScdsArr) == 0:
        print(
            f"Skipping empty plot for {videoPath.name} {colorspaceName}{filenameSuffix}"
        )
        return

    fig, axes = plt.subplots(
        layout="tight",
        nrows=3,
        figsize=(8, 6),
        sharex=not signalStartToEnd,
    )

    for channelIndex, ax in enumerate(axes):
        channelName = channelNames[channelIndex]
        channelResult = channelResults[channelIndex]
        channelTimes = fullTimeScdsArr
        channelValues = avgIntensArr[:, channelIndex]
        channelLabel = channelName
        if "error" not in channelResult and channelResult["inverted"]:
            channelValues = 1.0 - channelValues
            channelLabel = f"1 - {channelName}"

        if signalStartToEnd and "error" not in channelResult:
            _, startIndex, _ = channelResult["indices"]
            channelTimes = channelTimes[startIndex:]
            channelValues = channelValues[startIndex:]

        if "error" in channelResult:
            measurementLabel = "CRT90_10 unavailable"
        else:
            measurementLabel = f"CRT90_10: {crt90_10DecayLabel(channelResult)}"

        ax.plot(
            channelTimes,
            channelValues,
            color=CHANNEL_COLORS[colorspaceName, channelName],
            lw=1,
        )
        ax.set_title(channelLabel, loc="left", fontsize=9)
        configuredReleaseTime = RELEASE_TIMES.get(videoPath.name)
        if configuredReleaseTime is not None:
            ax.axvline(
                configuredReleaseTime,
                color="tab:purple",
                ls="-.",
                lw=0.9,
            )
        if releaseTime is not None:
            ax.axvline(
                releaseTime,
                color="black",
                ls="--",
                lw=1,
            )

        if "error" not in channelResult:
            crtIntervalStartIndex, startIndex, crtIntervalEndIndex = channelResult[
                "indices"
            ]
            crt90_10 = channelResult["crt90_10"]
            ax.axvline(
                fullTimeScdsArr[crtIntervalStartIndex],
                color="tab:gray",
                ls=":",
                lw=1,
            )
            ax.axvline(
                fullTimeScdsArr[startIndex],
                color="tab:orange",
                ls="-.",
                lw=1,
            )
            ax.axvline(
                fullTimeScdsArr[crtIntervalEndIndex],
                color="tab:gray",
                ls="--",
                lw=1,
            )
            ax.axvline(
                crt90_10["startTime"],
                color="tab:green",
                ls=":",
                lw=1,
            )
            ax.axvline(
                crt90_10["time90"],
                color="tab:blue",
                ls="--",
                label=f"90%: {crt90_10['time90']:.3f}s",
                lw=1,
            )
            ax.axvline(
                crt90_10["time10"],
                color="tab:red",
                ls="--",
                label=f"10%: {crt90_10['time10']:.3f}s",
                lw=1,
            )

        ax.plot(
            [],
            [],
            color=CHANNEL_COLORS[colorspaceName, channelName],
            lw=1,
            label=measurementLabel,
        )
        if signalStartToEnd and len(channelTimes):
            ax.set_xlim(channelTimes[0], fullTimeScdsArr[-1])
        ax.legend(loc="upper right", fontsize=7)

    fig.suptitle(f"{videoPath.stem} {colorspaceName}{filenameSuffix}")
    fig.supxlabel("Time (s)")
    fig.supylabel("Average ROI intensity")
    fig.savefig(outputDir / f"{colorspaceName}{filenameSuffix}.jpg", dpi=150)
    plt.close(fig)


def main():
    PLOTS_PATH.mkdir(parents=True, exist_ok=True)

    for videoPath in iterTrainingVideos():
        print(f"Working on {videoPath.name}...")
        try:
            loadVideoRoi(videoPath, ROIS_PATH)
        except (KeyError, ValueError) as err:
            print(f"Skipping {videoPath.name}: {err}")
            continue

        try:
            timeScdsArr, colorspaceIntensities, metricArr = measureVideo(videoPath)
        except Exception as err:
            print(
                f"Skipping {videoPath.name}: processing failed: "
                f"{type(err).__name__}: {err}"
            )
            continue

        try:
            releaseIndex = findReleaseIndex(
                metricArr,
                timeScdsArr,
                CANNY_PARAMS,
            )
        except Exception as err:
            print(
                f"{videoPath.name}: release detection failed: "
                f"{type(err).__name__}: {err}"
            )
            releaseIndex = None

        for colorspaceName, (_, channelNames) in COLORSPACES.items():
            avgIntensArr = colorspaceIntensities[colorspaceName]
            channelResults = calculateColorspaceCRT90_10(
                videoPath,
                timeScdsArr,
                avgIntensArr,
                metricArr,
                channelNames,
            )
            plotIntensitiesColorspace(
                videoPath,
                timeScdsArr,
                avgIntensArr,
                releaseIndex,
                channelResults,
                colorspaceName,
                channelNames,
            )
            plotIntensitiesColorspace(
                videoPath,
                timeScdsArr,
                avgIntensArr,
                releaseIndex,
                channelResults,
                colorspaceName,
                channelNames,
                signalStartToEnd=True,
            )


if __name__ == "__main__":
    main()
