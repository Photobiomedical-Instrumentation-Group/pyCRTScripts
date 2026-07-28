from __future__ import annotations

import json
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

import cv2 as cv
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pyCRT.curveFitting import (
    calcPCRTFirstPositivePeak,
    exponential,
    pCRTFromParameters,
)
from pyCRT.frameOperations import cropFrame, drawRoi, rescaleFrame
from pyCRT.videoReading import frameReader, videoCapture
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.signal import savgol_filter

import gui

try:
    import tomllib as tomli  # Python 3.11+
except ModuleNotFoundError:
    import tomli  # fallback for older Python

from pyCRT.simpleUI import DATETIME_FORMAT, DISPLAY_FORMAT, PCRT, RoiTuple

plt.style.use("bmh")

PLAYBACK_FPS_SPEED = {
    "fast": np.inf,
    "normal": 0,
    "slow": 25,
}

NORM_LEVEL = 25
logging.addLevelName(NORM_LEVEL, "NORM")
VIDEO_FORMATS = (".mp4", ".wmv", ".avi", ".mov", ".mkv")


def getLogger(name: str) -> logging.Logger:
    # {{{
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger  # prevent duplicate handlers
    fmt = logging.Formatter("[%(levelname)s] %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)
    return logger
    # }}}


LOGGER = getLogger("mauricio")


def enableFileLogging(
    logger: logging.Logger,
    filename: str = "log.txt",
    max_mb: int = 4,
) -> None:
    # {{{
    # Do nothing if file logging is already enabled
    for h in logger.handlers:
        if isinstance(h, RotatingFileHandler):
            return

    handler = RotatingFileHandler(
        filename,
        maxBytes=max_mb * 1024 * 1024,
        backupCount=1,
        encoding="utf-8",
    )

    formatter = logging.Formatter("[%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Mark start of this execution
    separator = (
        "\n"
        + "=" * 80
        + f"\nFile logging enabled at {datetime.now().strftime(DISPLAY_FORMAT)}\n"
        + "=" * 80
    )
    handler.stream.write(separator + "\n")
    handler.flush()


# }}}


def loadConfigFile(tomlPath, defaultsPath: Path | str = "defaults.toml") -> dict:
    # {{{
    with open(tomlPath, "rb") as arq:
        configDict = tomli.load(arq)
    with open(defaultsPath, "rb") as arq:
        defaultsDict = tomli.load(arq)

    setDefaults(defaultsDict, configDict)
    return defaultsDict  # counter-intuitive, but it is correct


# }}}


def setDefaults(
    target: dict[str, Any],
    source: dict[str, Any],
) -> None:
    # {{{
    # {{{
    """
    Recursively update `target` in-place using `dict.update()` semantics.
    - Nested dicts are merged recursively.
    - Non-dict values overwrite defaults.
    """
    # }}}
    for key, value in source.items():
        if key in target and isinstance(target[key], dict) and isinstance(value, dict):
            setDefaults(target[key], value)
        else:
            # dict.update semantics at this level
            target.update({key: value})


# }}}


def normalizeIntensities(array: np.ndarray) -> np.ndarray:
    # {{{
    array = np.asarray(array, dtype=float)
    if len(array) == 0 or array.max() == array.min():
        return np.zeros_like(array)
    return (array - array.min()) / (array.max() - array.min())


# }}}


def calculateCrt90_10Details(
    signalArr: np.ndarray,
    timeArr: np.ndarray,
    crtIntervalStartIndex: int,
    crtIntervalEndIndex: int,
    maxFitSeconds: float = 10.0,
    gaussianSigmaSeconds: float = 0.1,
) -> dict[str, Any]:
    # {{{
    intervalTimes = timeArr[crtIntervalStartIndex:crtIntervalEndIndex]
    intervalValues = signalArr[crtIntervalStartIndex:crtIntervalEndIndex]
    fitEndTime = min(intervalTimes[0] + maxFitSeconds, intervalTimes[-1])
    fitEndOffset = int(np.searchsorted(intervalTimes, fitEndTime, side="right"))
    fitEndOffset = max(fitEndOffset, 2)
    crt90Times = intervalTimes[:fitEndOffset]
    crt90Values = np.asarray(intervalValues[:fitEndOffset], dtype=float)
    positiveTimeDiffs = np.diff(crt90Times)
    positiveTimeDiffs = positiveTimeDiffs[positiveTimeDiffs > 0]
    if len(positiveTimeDiffs) == 0:
        raise ValueError("Need increasing timestamps to calculate CRT90_10.")

    medianFrameTime = float(np.median(positiveTimeDiffs))
    sigmaSamples = gaussianSigmaSeconds / medianFrameTime
    if sigmaSamples > 0:
        smoothedRawValues = gaussian_filter1d(crt90Values, sigma=sigmaSamples)
    else:
        smoothedRawValues = crt90Values
    smoothedValues = normalizeIntensities(smoothedRawValues)
    startOffset90 = int(np.argmax(smoothedValues))
    value90 = 0.9 * smoothedValues[startOffset90]
    value10 = 0.1 * smoothedValues[startOffset90]
    ninetyCandidates = np.flatnonzero(smoothedValues[startOffset90:] < value90)
    if len(ninetyCandidates) == 0:
        raise ValueError("No signal point fell below the 90% value.")
    time90Offset = startOffset90 + int(ninetyCandidates[0])
    tenCandidates = np.flatnonzero(smoothedValues[time90Offset:] < value10)
    if len(tenCandidates) == 0:
        raise ValueError("No signal point fell below the 10% value.")
    time10Offset = time90Offset + int(tenCandidates[0])

    return {
        "crt90_10": float(crt90Times[time10Offset] - crt90Times[time90Offset]),
        "startTime": float(crt90Times[startOffset90]),
        "time90": float(crt90Times[time90Offset]),
        "time10": float(crt90Times[time10Offset]),
        "times": crt90Times,
        "values": smoothedValues,
        "rawValues": crt90Values,
        "smoothedRawValues": smoothedRawValues,
        "sigmaSamples": float(sigmaSamples),
        "maxFitSeconds": float(maxFitSeconds),
        "gaussianSigmaSeconds": float(gaussianSigmaSeconds),
    }


# }}}


def validateUncertaintyRatio(
    metricName: str,
    metricValue: float,
    uncertainty: float,
    maxUncertaintyRatio: float,
) -> float:
    # {{{
    metricValue = float(metricValue)
    uncertainty = float(uncertainty)
    maxUncertaintyRatio = float(maxUncertaintyRatio)

    if np.isnan(uncertainty):
        raise ValueError(f"{metricName} uncertainty is NaN.")
    if not np.isfinite(uncertainty) or uncertainty < 0:
        raise ValueError(
            f"{metricName} uncertainty must be finite and non-negative: "
            f"{uncertainty}."
        )
    if not np.isfinite(metricValue) or metricValue == 0:
        raise ValueError(
            f"{metricName} value must be finite and non-zero to calculate "
            f"relative uncertainty: {metricValue}."
        )

    uncertaintyRatio = abs(uncertainty / metricValue)
    if uncertaintyRatio > maxUncertaintyRatio:
        raise ValueError(
            f"{metricName} relative uncertainty {uncertaintyRatio:.6g} exceeds "
            f"maxUncertaintyRatio {maxUncertaintyRatio:.6g}."
        )

    return float(uncertaintyRatio)


# }}}


def calculatePcrtDetails(
    signalArr: np.ndarray,
    timeArr: np.ndarray,
    crtIntervalStartIndex: int,
    startIndex: int,
    crtIntervalEndIndex: int,
    maxUncertaintyRatio: float = np.inf,
) -> dict[str, Any]:
    # {{{
    intervalValues = signalArr[crtIntervalStartIndex:crtIntervalEndIndex]
    fitTimes = timeArr[startIndex:crtIntervalEndIndex] - timeArr[startIndex]
    fitValues = signalArr[startIndex:crtIntervalEndIndex]
    if len(fitTimes) < 4 or intervalValues.max() == intervalValues.min():
        raise ValueError("Not enough usable samples to calculate pCRT.")

    referenceMin = intervalValues.min()
    referenceRange = intervalValues.max() - referenceMin
    fitValues = (fitValues - referenceMin) / referenceRange
    pcrtTuple, criticalTime = calcPCRTFirstPositivePeak(fitTimes, fitValues)
    pcrtValue, pcrtUncertainty = pCRTFromParameters(pcrtTuple)
    uncertaintyRatio = validateUncertaintyRatio(
        "pCRT",
        pcrtValue,
        pcrtUncertainty,
        maxUncertaintyRatio,
    )
    pcrtParams, _ = pcrtTuple
    fitValues = exponential(fitTimes, *pcrtParams) * referenceRange + referenceMin

    return {
        "pcrt": float(pcrtValue),
        "pcrtUncertainty": float(pcrtUncertainty),
        "pcrtUncertaintyRatio": float(uncertaintyRatio),
        "criticalTime": float(criticalTime + timeArr[startIndex]),
        "fitTimes": fitTimes + timeArr[startIndex],
        "fitValues": fitValues,
    }


# }}}


def createCrtMeasurementPlot(
    videoPath: Path | str,
    labAIntensArr: np.ndarray,
    bgrGIntensArr: np.ndarray,
    timeArr: np.ndarray,
    releaseIndex: int,
    crtIntervalStartIndex: int,
    startIndex: int,
    crtIntervalEndIndex: int,
    metrics: dict[str, tuple[float, float]],
    metricDetails: dict[str, dict[str, Any]] | None = None,
    metricErrors: dict[str, str] | None = None,
    releaseMetricData: dict[str, Any] | None = None,
    plotSections: set[str] | None = None,
):
    # {{{
    videoPath = Path(videoPath)
    metricDetails = metricDetails or {}
    metricErrors = metricErrors or {}
    plotSections = set(plotSections or {"bgr", "lab", "edge"})
    selectedChannels = []
    if "bgr" in plotSections:
        selectedChannels.append(("BGR G", bgrGIntensArr, "tab:green", "bgr_g", "bgr"))
    if "lab" in plotSections:
        selectedChannels.append(("-LAB A", -1 * labAIntensArr, "tab:blue", "lab_a", "lab"))
    if not selectedChannels and "edge" not in plotSections:
        selectedChannels.append(("BGR G", bgrGIntensArr, "tab:green", "bgr_g", "bgr"))

    mosaic = []
    if selectedChannels:
        mosaic.append([f"{channelKey}_full" for *_, channelKey in selectedChannels])
        mosaic.append([f"{channelKey}_slice" for *_, channelKey in selectedChannels])
    if "edge" in plotSections:
        edgeRowWidth = max(len(selectedChannels), 1)
        mosaic.append(["release_metric"] * edgeRowWidth)

    figsize = (6 * max(len(selectedChannels), 1), 2.8 * len(mosaic))
    fig, axes = plt.subplot_mosaic(
        mosaic,
        figsize=figsize,
        num=f"CRT channels - {videoPath.name}",
    )
    intervalTimes = timeArr[crtIntervalStartIndex:crtIntervalEndIndex]
    intervalEndMarkerIndex = min(crtIntervalEndIndex, len(timeArr) - 1)
    intervalEndTime = timeArr[intervalEndMarkerIndex]
    for channelLabel, signalArr, color, metricPrefix, channelKey in selectedChannels:
        pcrtKey = f"{metricPrefix}_pcrt"
        crt90Key = f"{metricPrefix}_crt90_10"
        pcrt = metricDetails.get(pcrtKey)
        crt90 = metricDetails.get(crt90Key)

        fullAx = axes[f"{channelKey}_full"]
        fullAx.plot(timeArr, normalizeIntensities(signalArr), color=color, lw=1)
        fullAx.axvspan(
            timeArr[crtIntervalStartIndex],
            intervalEndTime,
            color="tab:gray",
            alpha=0.12,
        )
        fullAx.axvline(timeArr[releaseIndex], color="black", ls="--", label="release")
        fullAx.axvline(
            timeArr[crtIntervalStartIndex],
            color="black",
            ls=":",
            label="CRT start",
        )
        fullAx.axvline(timeArr[startIndex], color="tab:green", ls="-", label="start")
        fullAx.axvline(intervalEndTime, color="black", ls="-.", label="CRT end")
        if crt90 is not None:
            fullAx.axvline(
                crt90["time90"],
                color="tab:green",
                ls="--",
                label=f"90%={crt90['time90']:.3f}s",
            )
            fullAx.axvline(
                crt90["time10"],
                color="tab:red",
                ls="--",
                label=f"10%={crt90['time10']:.3f}s",
            )
        if pcrt is not None:
            fullAx.axvline(
                pcrt["criticalTime"],
                color="tab:purple",
                ls=":",
                label=f"critical={pcrt['criticalTime']:.3f}s",
            )
        fullAx.set_title(f"{channelLabel} full average intensity")
        fullAx.grid(True)
        fullAx.legend(loc="lower left", fontsize="small")

        detailAx = axes[f"{channelKey}_slice"]
        intervalNorm = normalizeIntensities(
            signalArr[crtIntervalStartIndex:crtIntervalEndIndex]
        )
        detailAx.plot(intervalTimes, intervalNorm, color=color, lw=1, label="slice")
        if crt90 is not None and crt90Key in metrics:
            crt90Value, crt90Uncertainty = metrics[crt90Key]
            detailAx.plot(
                crt90["times"],
                crt90["values"],
                color="tab:orange",
                lw=1.4,
                label=f"CRT90_10={crt90Value:.3f} +/- {crt90Uncertainty:.3f}s",
            )
            detailAx.axvline(crt90["startTime"], color="tab:green", ls=":")
            detailAx.axvline(
                crt90["time90"],
                color="tab:green",
                ls="--",
                label=f"90%={crt90['time90']:.3f}s",
            )
            detailAx.axvline(
                crt90["time10"],
                color="tab:red",
                ls="--",
                label=f"10%={crt90['time10']:.3f}s",
            )
        if pcrt is not None and pcrtKey in metrics:
            pcrtValue, pcrtUncertainty = metrics[pcrtKey]
            referenceValues = signalArr[crtIntervalStartIndex:crtIntervalEndIndex]
            referenceMin = referenceValues.min()
            referenceRange = referenceValues.max() - referenceMin
            if referenceRange != 0:
                normalizedFitValues = (pcrt["fitValues"] - referenceMin) / referenceRange
                detailAx.plot(
                    pcrt["fitTimes"],
                    normalizedFitValues,
                    color="tab:purple",
                    lw=1.4,
                    label=f"pCRT={pcrtValue:.3f} +/- {pcrtUncertainty:.3f}s",
                )
            detailAx.axvline(
                pcrt["criticalTime"],
                color="tab:purple",
                ls=":",
                label=f"critical={pcrt['criticalTime']:.3f}s",
            )
        failedMetricLabels = []
        if crt90Key in metricErrors:
            failedMetricLabels.append("CRT90_10 failed")
        if pcrtKey in metricErrors:
            failedMetricLabels.append("pCRT failed")
        if failedMetricLabels:
            detailAx.text(
                0.98,
                0.02,
                "\n".join(failedMetricLabels),
                ha="right",
                va="bottom",
                transform=detailAx.transAxes,
                fontsize="small",
            )
        detailAx.set_title(f"{channelLabel} sliced CRT")
        detailAx.grid(True)
        handles, _ = detailAx.get_legend_handles_labels()
        if handles:
            detailAx.legend(loc="lower left", fontsize="small")

    if "edge" in plotSections:
        metricAx = axes["release_metric"]
        if releaseMetricData is None or len(releaseMetricData.get("times", [])) == 0:
            metricAx.text(
                0.5,
                0.5,
                "No release metric samples were available.",
                ha="center",
                va="center",
                transform=metricAx.transAxes,
            )
        else:
            metricTimes = np.asarray(releaseMetricData["times"], dtype=float)
            metricValues = np.asarray(releaseMetricData["values"], dtype=float)
            metricLabel = releaseMetricData.get("label", "release metric")
            metricAx.plot(
                metricTimes,
                metricValues,
                color="tab:blue",
                lw=1,
                label=metricLabel,
            )
            metricAx.axvline(
                timeArr[releaseIndex],
                color="black",
                ls="--",
                label=f"release={releaseIndex}",
            )
            metricAx.legend(loc="lower left", fontsize="small")
        metricAx.set_title("Release detection metric")
        metricAx.grid(True)

    fig.suptitle(videoPath.name)
    fig.supxlabel("Time (s)")
    fig.text(0.005, 0.5, "Normalized intensity", va="center", rotation="vertical")
    fig.tight_layout(rect=(0.025, 0.035, 0.985, 0.955))
    return fig

# }}}


def createAverageIntensityFailurePlot(
    videoPath: Path | str,
    labAIntensArr: np.ndarray,
    bgrGIntensArr: np.ndarray,
    timeArr: np.ndarray,
    releaseIndex: int | None = None,
    crtIntervalStartIndex: int | None = None,
    startIndex: int | None = None,
    crtIntervalEndIndex: int | None = None,
    error: Exception | None = None,
    releaseMetricData: dict[str, Any] | None = None,
    plotSections: set[str] | None = None,
):
    # {{{
    videoPath = Path(videoPath)
    plotSections = set(plotSections or {"bgr", "lab", "edge"})
    selectedChannels = []
    if "bgr" in plotSections:
        selectedChannels.append(("BGR G", bgrGIntensArr, "tab:green", "bgr"))
    if "lab" in plotSections:
        selectedChannels.append(("LAB A", labAIntensArr, "tab:blue", "lab"))
    if not selectedChannels and "edge" not in plotSections:
        selectedChannels.append(("BGR G", bgrGIntensArr, "tab:green", "bgr"))

    mosaic = []
    if selectedChannels:
        mosaic.append([f"{channelKey}_full" for *_, channelKey in selectedChannels])
    if "edge" in plotSections:
        edgeRowWidth = max(len(selectedChannels), 1)
        mosaic.append(["release_metric"] * edgeRowWidth)

    fig, axes = plt.subplot_mosaic(
        mosaic,
        figsize=(6 * max(len(selectedChannels), 1), 3 * len(mosaic)),
        num=f"CRT calculation failed - {videoPath.name}",
    )
    markerSpecs = (
        (releaseIndex, "release", "black", "--"),
        (crtIntervalStartIndex, "CRT start", "black", ":"),
        (startIndex, "start", "tab:green", "-"),
        (crtIntervalEndIndex, "CRT end", "black", "-."),
    )
    for channelLabel, signalArr, color, channelKey in selectedChannels:
        ax = axes[f"{channelKey}_full"]
        if len(timeArr) == 0 or len(signalArr) == 0:
            ax.text(
                0.5,
                0.5,
                "No average intensity samples were available.",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
        else:
            ax.plot(timeArr, normalizeIntensities(signalArr), color=color, lw=1)
        for index, label, lineColor, lineStyle in markerSpecs:
            if len(timeArr) == 0:
                continue
            if index is None:
                continue
            markerIndex = min(max(int(index), 0), len(timeArr) - 1)
            ax.axvline(
                timeArr[markerIndex],
                color=lineColor,
                ls=lineStyle,
                label=f"{label}={markerIndex}",
            )
        ax.set_title(f"{channelLabel} full average intensity")
        ax.grid(True)
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="lower left", fontsize="small")

    if "edge" in plotSections:
        metricAx = axes["release_metric"]
        if releaseMetricData is None or len(releaseMetricData.get("times", [])) == 0:
            metricAx.text(
                0.5,
                0.5,
                "No release metric samples were available.",
                ha="center",
                va="center",
                transform=metricAx.transAxes,
            )
        else:
            metricTimes = np.asarray(releaseMetricData["times"], dtype=float)
            metricValues = np.asarray(releaseMetricData["values"], dtype=float)
            metricLabel = releaseMetricData.get("label", "release metric")
            metricAx.plot(
                metricTimes,
                metricValues,
                color="tab:blue",
                lw=1,
                label=metricLabel,
            )
            if releaseIndex is not None and len(timeArr) > 0:
                markerIndex = min(max(int(releaseIndex), 0), len(timeArr) - 1)
                metricAx.axvline(
                    timeArr[markerIndex],
                    color="black",
                    ls="--",
                    label=f"release={markerIndex}",
                )
            metricAx.legend(loc="lower left", fontsize="small")
        metricAx.set_title("Release detection metric")
        metricAx.grid(True)

    title = f"CRT calculation failed - {videoPath.name}"
    if error is not None:
        title += f"\n{type(error).__name__}: {error}"
    fig.suptitle(title)
    fig.supxlabel("Time (s)")
    fig.text(0.005, 0.5, "Normalized intensity", va="center", rotation="vertical")
    fig.tight_layout(rect=(0.025, 0.08, 0.985, 0.88))
    return fig

# }}}


def measureCRTVideoFromConfig(
    videoPath: Path | str,
    configDict: dict[str, Any],
    savePlot: bool = False,
) -> dict[str, Any]:
    # {{{
    videoPath = Path(videoPath)
    generalConfig = configDict.get("General", {})
    showAllPlots = bool(
        generalConfig.get("showAllPlots", generalConfig.get("showPlots", False))
    )
    plotSections = set()
    if showAllPlots or generalConfig.get("showBGRPlot", False):
        plotSections.add("bgr")
    if showAllPlots or generalConfig.get("showLABPlot", False):
        plotSections.add("lab")
    if showAllPlots or generalConfig.get("showEdgeDetectionPlot", False):
        plotSections.add("edge")
    showPlots = bool(plotSections)

    roi = configDict["Measurement"]["roi"]
    if roi == -1:
        pass
    elif not isValidRoi(roi):
        raise ValueError(
            f"{roi} is not a valid value for the ROI. "
            "Valid values are 4-element iterables."
        )
    else:
        roi = tuple(roi)

    releaseConfig = configDict["ReleaseFrameDetection"]
    releaseAlgorithm = str(releaseConfig.get("algorithm", "canny"))
    releaseAlgorithm = releaseAlgorithm.strip().lower().replace("-", "_")
    if releaseAlgorithm == "laplace":
        releaseAlgorithm = "laplacian"
    if releaseAlgorithm not in {"canny", "laplacian"}:
        raise ValueError(
            f"Unsupported release detection algorithm: {releaseAlgorithm}. "
            "Expected 'canny' or 'laplacian'."
        )
    releaseParams = {
        key: value
        for key, value in releaseConfig.items()
        if not isinstance(value, dict)
    }
    releaseParams.update(releaseConfig[releaseAlgorithm])
    releaseParams["algorithm"] = releaseAlgorithm

    slicingParams = dict(configDict["AverageIntensitySlicing"])
    videoConfig = configDict["Video"]
    (
        labAIntensArr,
        bgrGIntensArr,
        timeArr,
        releaseIndex,
        releaseMetricData,
    ) = detectReleaseFrameFromVideo(
        videoPath,
        roi,
        releaseParams,
        showEdgeDetection=videoConfig.get("showEdgeDetection", False),
        showVideoFrames=videoConfig.get("showVideoFrames", False),
        playbackSpeed=videoConfig.get("playbackSpeed", "fast"),
        returnReleaseMetric=True,
    )
    crtIntervalStartIndex = None
    startIndex = None
    crtIntervalEndIndex = None
    try:
        crtIndices = findCrtIntervalIndices(
            labAIntensArr,
            timeArr,
            releaseIndex,
            slicingParams,
        )
        crtIntervalStartIndex, startIndex, crtIntervalEndIndex = crtIndices
        metrics, metricDetails, metricErrors = calculateCrtMetrics(
            labAIntensArr,
            bgrGIntensArr,
            timeArr,
            crtIntervalStartIndex,
            startIndex,
            crtIntervalEndIndex,
            maxUncertaintyRatio=configDict["Measurement"].get(
                "maxUncertaintyRatio",
                np.inf,
            ),
        )
    except Exception as err:
        if showPlots:
            failureFig = createAverageIntensityFailurePlot(
                videoPath,
                labAIntensArr,
                bgrGIntensArr,
                timeArr,
                releaseIndex=releaseIndex,
                crtIntervalStartIndex=crtIntervalStartIndex,
                startIndex=startIndex,
                crtIntervalEndIndex=crtIntervalEndIndex,
                error=err,
                releaseMetricData=releaseMetricData,
                plotSections=plotSections,
            )
            failureFig.show()
            plt.show()
        raise

    result = {
        "videoPath": str(videoPath),
        "roi": roi,
        "releaseIndex": int(releaseIndex),
        "releaseTime": float(timeArr[releaseIndex]),
        "crtIntervalStartIndex": int(crtIntervalStartIndex),
        "startIndex": int(startIndex),
        "crtIntervalEndIndex": int(crtIntervalEndIndex),
        "measurementTime": datetime.now().strftime(DATETIME_FORMAT),
        "releaseParams": releaseParams,
        "metrics": metrics,
        "metricDetails": metricDetails,
        "metricErrors": metricErrors,
        "labAIntensArr": labAIntensArr,
        "bgrGIntensArr": bgrGIntensArr,
        "timeArr": timeArr,
        "releaseMetricData": releaseMetricData,
    }

    if savePlot or showPlots:
        fig = createCrtMeasurementPlot(
            videoPath,
            labAIntensArr,
            bgrGIntensArr,
            timeArr,
            releaseIndex,
            crtIntervalStartIndex,
            startIndex,
            crtIntervalEndIndex,
            metrics,
            metricDetails=metricDetails,
            metricErrors=metricErrors,
            releaseMetricData=releaseMetricData,
            plotSections=plotSections or {"bgr", "lab", "edge"},
        )
        if savePlot:
            plotPath = Path(configDict["Files"]["plotPath"])
            plotPath.mkdir(parents=True, exist_ok=True)
            outputPath = plotPath / f"{videoPath.stem}_crt_channels.png"
            if not configDict["Files"].get("overwrite", False):
                outputPath = findUniquePath(outputPath)
            fig.savefig(outputPath, dpi=150)
            LOGGER.info(f"CRT channels plot saved in {outputPath}")
            result["plotPath"] = str(outputPath)
        if showPlots:
            fig.show()
            plt.show()
        else:
            plt.close(fig)

    return result

# }}}


def detectReleaseFrameFromVideo(
    videoPath: Path | str,
    roi: RoiTuple | None,
    params: dict[str, Any],
    showEdgeDetection: bool = False,
    showVideoFrames: bool = False,
    playbackSpeed: str = "fast",
    returnReleaseMetric: bool = False,
):
    # {{{
    videoPath = Path(videoPath)
    algorithm = params.get(
        "algorithm",
        params.get(
            "filterType", params.get("filter_type", params.get("releaseAlgorithm"))
        ),
    )
    if algorithm is None:
        laplacianKeys = {"ksize", "laplacianPlateau", "laplacianSmoothingKernel"}
        algorithm = "laplacian" if laplacianKeys & set(params) else "canny"
    algorithm = str(algorithm).strip().lower().replace("-", "_")
    if algorithm == "laplace":
        algorithm = "laplacian"
    if algorithm not in {"canny", "laplacian"}:
        raise ValueError(
            f"Unsupported release detection algorithm: {algorithm}. "
            "Expected 'canny' or 'laplacian'."
        )

    playbackSpeed = str(playbackSpeed).strip().lower()
    playbackWaitMsBySpeed = {
        "fast": 1,
        "medium": 33,
        "slow": 100,
    }
    if playbackSpeed not in playbackWaitMsBySpeed:
        raise ValueError(
            f"'{playbackSpeed}' is not a valid playbackSpeed. "
            "Valid values are 'fast', 'medium' or 'slow'."
        )

    medianKernelRadius = int(params.get("medianKernelRadius", 1))
    rescaleFactor = float(params.get("rescaleFactor", 0.5))
    roiSelected = isValidRoi(roi) or roi == "all"
    requireRoiSelection = not roiSelected
    bgrWindowName = "BGR frame with ROI"

    if requireRoiSelection:
        with videoCapture(str(videoPath)) as cap:
            for frame in frameReader(cap):
                if medianKernelRadius > 0:
                    medianFrame = cv.medianBlur(frame, (2 * medianKernelRadius) + 1)
                else:
                    medianFrame = frame

                rescaledBgrFrame = rescaleFrame(medianFrame, rescaleFactor)
                cv.namedWindow(bgrWindowName, cv.WINDOW_NORMAL | cv.WINDOW_GUI_NORMAL)
                cv.resizeWindow(
                    bgrWindowName,
                    rescaledBgrFrame.shape[1],
                    rescaledBgrFrame.shape[0],
                )
                cv.imshow(bgrWindowName, rescaledBgrFrame)
                key = cv.waitKey(playbackWaitMsBySpeed[playbackSpeed]) & 0xFF
                if key == ord(" "):
                    selectedRoi = cv.selectROI(bgrWindowName, rescaledBgrFrame)
                    if selectedRoi[2] > 0 and selectedRoi[3] > 0:
                        roi = tuple(int(value) for value in selectedRoi)
                        roiSelected = True
                        print(f"Selected ROI: {list(roi)}. Restarting video.")
                        break
                    print("No ROI selected. Press spacebar and drag a non-empty ROI.")
                elif key == ord("q"):
                    break

        if showVideoFrames is False:
            try:
                cv.destroyWindow(bgrWindowName)
            except cv.error:
                pass

    if requireRoiSelection and not roiSelected:
        raise RuntimeError(
            "No ROI was selected before the video ended. "
            "Press the spacebar while the BGR video is displayed, "
            "drag a non-empty ROI, and confirm the selection before running "
            "CRT calculation."
        )

    labAIntensities = []
    bgrGIntensities = []
    releaseMetrics = []
    times = []

    with videoCapture(str(videoPath)) as cap:
        for frame in frameReader(cap):
            timeScds = cap.get(cv.CAP_PROP_POS_MSEC) / 1000.0
            if medianKernelRadius > 0:
                medianFrame = cv.medianBlur(frame, (2 * medianKernelRadius) + 1)
            else:
                medianFrame = frame

            rescaledBgrFrame = rescaleFrame(medianFrame, rescaleFactor)
            if showVideoFrames:
                cv.namedWindow(bgrWindowName, cv.WINDOW_NORMAL | cv.WINDOW_GUI_NORMAL)
                displayFrame = drawRoi(rescaledBgrFrame.copy(), roi)
                cv.resizeWindow(
                    bgrWindowName,
                    displayFrame.shape[1],
                    displayFrame.shape[0],
                )
                cv.imshow(bgrWindowName, displayFrame)

            croppedBgrFrame = cropFrame(rescaledBgrFrame, roi)
            float32Frame = medianFrame.astype(np.float32) / 255
            labFrame = cv.cvtColor(float32Frame, cv.COLOR_BGR2LAB)
            rescaledLabFrame = rescaleFrame(labFrame, rescaleFactor)
            croppedLabFrame = cropFrame(rescaledLabFrame, roi)
            lFrame = np.uint8(np.round(croppedLabFrame[..., 0] * 255 / 100))

            times.append(timeScds)
            labAIntensities.append(float(np.mean(croppedLabFrame[..., 1])))
            bgrGIntensities.append(float(np.mean(croppedBgrFrame[..., 1])))

            if algorithm == "canny":
                edgeFrame = lFrame
                blurKernel = int(params.get("blurKernel", 0))
                if blurKernel > 0:
                    kernel = (2 * blurKernel) + 1
                    edgeFrame = cv.GaussianBlur(edgeFrame, (kernel, kernel), 0)
                cannyFrame = cv.Canny(
                    edgeFrame,
                    threshold1=int(params["thresh1"]),
                    threshold2=int(params["thresh2"]),
                    L2gradient=bool(params.get("l2grad")),
                )
                releaseMetrics.append(float(np.sum(cannyFrame.astype(np.float32))))
                if showEdgeDetection:
                    cv.imshow("Canny release frame", cannyFrame)
            else:
                edgeFrame = lFrame
                blurKernel = int(params.get("blurKernel", 0))
                if blurKernel > 0:
                    kernel = (2 * blurKernel) + 1
                    edgeFrame = cv.GaussianBlur(edgeFrame, (kernel, kernel), 0)
                laplacianFrame = cv.Laplacian(
                    edgeFrame,
                    ddepth=cv.CV_64F,
                    ksize=(2 * int(params["ksize"])) + 1,
                    scale=float(params.get("scale", 1)),
                    delta=float(params.get("delta", 0)),
                )
                releaseMetrics.append(float(np.sum(laplacianFrame**2)))
                if showEdgeDetection:
                    normalizedLaplacianFrame = cv.convertScaleAbs(
                        255 * normalizeIntensities(laplacianFrame)
                    )
                    cv.imshow(
                        "Laplacian | L",
                        cv.hconcat([normalizedLaplacianFrame, lFrame]),
                    )

            if showEdgeDetection or showVideoFrames:
                key = cv.waitKey(playbackWaitMsBySpeed[playbackSpeed]) & 0xFF
                if key == ord(" ") and showVideoFrames and not requireRoiSelection:
                    selectedRoi = cv.selectROI(bgrWindowName, rescaledBgrFrame)
                    if selectedRoi[2] > 0 and selectedRoi[3] > 0:
                        roi = tuple(int(value) for value in selectedRoi)
                        labAIntensities = []
                        bgrGIntensities = []
                        releaseMetrics = []
                        times = []
                        print(f"Selected ROI: {list(roi)}")
                    else:
                        print("No ROI selected. Press spacebar and drag a non-empty ROI.")
                elif key == ord("q"):
                    break

    if showEdgeDetection or showVideoFrames:
        for windowName in (
            bgrWindowName,
            "Canny release frame",
            "Laplacian | L",
        ):
            try:
                cv.destroyWindow(windowName)
            except cv.error:
                pass

    timeArr = np.asarray(times, dtype=float)
    labAIntensArr = np.asarray(labAIntensities, dtype=float)
    bgrGIntensArr = np.asarray(bgrGIntensities, dtype=float)
    releaseMetricArr = np.asarray(releaseMetrics, dtype=float)
    if len(timeArr) == 0:
        raise RuntimeError(f"No frames were read from {videoPath}.")

    if releaseMetricArr.max() == releaseMetricArr.min():
        releaseMetricArr = np.zeros_like(releaseMetricArr)
    else:
        releaseMetricArr = 100 * (
            (releaseMetricArr - releaseMetricArr.min())
            / (releaseMetricArr.max() - releaseMetricArr.min())
        )

    keepIndices = [0]
    lastTime = timeArr[0]
    for index in range(1, len(timeArr)):
        if timeArr[index] > lastTime:
            keepIndices.append(index)
            lastTime = timeArr[index]
    keepIndices = np.asarray(keepIndices, dtype=int)
    filteredMetricArr = releaseMetricArr[keepIndices]
    filteredTimeArr = timeArr[keepIndices]
    if len(filteredMetricArr) < 2:
        raise ValueError("Need at least two valid timestamped release samples.")

    if algorithm == "laplacian":
        plateau = float(params["laplacianPlateau"])
        smoothingKernel = int(params["laplacianSmoothingKernel"])
        gradientSmoothingKernel = int(params.get("laplacianGradientSmoothingKernel", 1))
    else:
        plateau = float(params["cannyPlateau"])
        smoothingKernel = int(params["cannySmoothingKernel"])
        gradientSmoothingKernel = int(params.get("cannyGradientSmoothingKernel", 1))

    if smoothingKernel > 1:
        smoothedMetricArr = median_filter(filteredMetricArr, size=smoothingKernel)
        if smoothedMetricArr.max() != smoothedMetricArr.min():
            smoothedMetricArr = 100 * (
                (smoothedMetricArr - smoothedMetricArr.min())
                / (smoothedMetricArr.max() - smoothedMetricArr.min())
            )
    else:
        smoothedMetricArr = filteredMetricArr
    metricGradientArr = np.gradient(smoothedMetricArr, filteredTimeArr)
    if gradientSmoothingKernel > 1:
        metricGradientArr = median_filter(
            metricGradientArr, size=gradientSmoothingKernel
        )

    minGradOffset = int(np.argmin(metricGradientArr))
    maxGradOffset = int(np.argmax(metricGradientArr))
    if minGradOffset < maxGradOffset:
        releaseMetricArr = releaseMetricArr.max() - releaseMetricArr
        filteredMetricArr = releaseMetricArr[keepIndices]
        if smoothingKernel > 1:
            smoothedMetricArr = median_filter(filteredMetricArr, size=smoothingKernel)
            if smoothedMetricArr.max() != smoothedMetricArr.min():
                smoothedMetricArr = 100 * (
                    (smoothedMetricArr - smoothedMetricArr.min())
                    / (smoothedMetricArr.max() - smoothedMetricArr.min())
                )
        else:
            smoothedMetricArr = filteredMetricArr
        metricGradientArr = np.gradient(smoothedMetricArr, filteredTimeArr)
        if gradientSmoothingKernel > 1:
            metricGradientArr = median_filter(
                metricGradientArr, size=gradientSmoothingKernel
            )
        minGradOffset = int(np.argmin(metricGradientArr))

    releaseCandidates = np.argwhere(
        (np.arange(len(smoothedMetricArr)) >= minGradOffset)
        & (np.abs(metricGradientArr) <= plateau)
    )
    if len(releaseCandidates) == 0:
        raise ValueError("No release candidate satisfied the plateau condition.")

    releaseIndex = keepIndices[int(releaseCandidates[0, 0])] + int(
        params.get("indexOffset", 0)
    )
    releaseIndex = int(np.clip(releaseIndex, 0, len(timeArr) - 1))
    if returnReleaseMetric:
        releaseMetricData = {
            "times": filteredTimeArr,
            "values": smoothedMetricArr,
            "label": f"{algorithm} release metric",
        }
        return labAIntensArr, bgrGIntensArr, timeArr, releaseIndex, releaseMetricData
    return labAIntensArr, bgrGIntensArr, timeArr, releaseIndex


# }}}


def findCrtIntervalIndices(
    labAIntensArr: np.ndarray,
    timeArr: np.ndarray,
    releaseIndex: int,
    params: dict[str, Any],
) -> tuple[int, int, int]:
    # {{{
    strictMinGrad = float(params["strictMinGrad"])
    strictMaxGrad = float(params["strictMaxGrad"])
    relaxMinGrad = float(params["relaxMinGrad"])
    relaxMaxGrad = float(params["relaxMaxGrad"])
    offsetTime = float(params.get("offsetTime", 0.0))
    if not strictMinGrad < strictMaxGrad:
        raise ValueError("strictMinGrad must be less than strictMaxGrad.")
    if not relaxMinGrad < relaxMaxGrad:
        raise ValueError("relaxMinGrad must be less than relaxMaxGrad.")
    if relaxMinGrad > strictMinGrad or relaxMaxGrad < strictMaxGrad:
        raise ValueError("The relaxed gradient range must contain the strict range.")

    labAIntensArr = np.asarray(labAIntensArr, dtype=float)
    timeArr = np.asarray(timeArr, dtype=float)
    releaseIndex = int(releaseIndex)
    if not 0 <= releaseIndex < len(timeArr):
        raise ValueError(f"releaseIndex is outside the time array: {releaseIndex}.")

    releaseTime = timeArr[releaseIndex]
    searchEndTime = min(releaseTime + 30, timeArr[-1])
    searchEndIndex = int(np.argmin(np.abs(timeArr - searchEndTime)))
    if searchEndIndex <= releaseIndex + 1:
        raise ValueError("Need at least two samples after releaseIndex.")

    if labAIntensArr.max() == labAIntensArr.min():
        normalizedLabA = np.zeros_like(labAIntensArr)
    else:
        normalizedLabA = (labAIntensArr - labAIntensArr.min()) / (
            labAIntensArr.max() - labAIntensArr.min()
        )
    searchSignal = normalizedLabA[releaseIndex:searchEndIndex]
    searchTimes = timeArr[releaseIndex:searchEndIndex]

    keepOffsets = [0]
    lastTime = searchTimes[0]
    for offset in range(1, len(searchTimes)):
        if searchTimes[offset] > lastTime:
            keepOffsets.append(offset)
            lastTime = searchTimes[offset]
    sourceOffsets = np.asarray(keepOffsets, dtype=int)
    filteredSignal = searchSignal[sourceOffsets]
    filteredTimes = searchTimes[sourceOffsets]
    if len(filteredSignal) < 2:
        raise ValueError("Need at least two increasing timestamps for CRT interval.")

    smoothedSignal = filteredSignal
    medianFrameTime = float(np.median(np.diff(filteredTimes)))
    if medianFrameTime > 0:
        windowLength = round(0.1 / medianFrameTime)
        if windowLength % 2 == 0:
            windowLength += 1
        windowLength = max(windowLength, 5)
        if windowLength <= len(filteredSignal):
            smoothedSignal = savgol_filter(
                filteredSignal,
                window_length=windowLength,
                polyorder=3,
                mode="interp",
            )
    labAGradient = np.gradient(smoothedSignal, filteredTimes)
    strictStartMask = (strictMinGrad <= labAGradient) & (labAGradient <= strictMaxGrad)
    relaxMask = (relaxMinGrad <= labAGradient) & (labAGradient <= relaxMaxGrad)

    strictOffsets = np.flatnonzero(strictStartMask)
    if len(strictOffsets) == 0:
        raise ValueError("No LAB A gradient point satisfied the strict start range.")
    crtIntervalStartOffset = int(strictOffsets[0])
    relaxedFailures = np.flatnonzero(~relaxMask[crtIntervalStartOffset:])
    if len(relaxedFailures) == 0:
        crtIntervalEndOffset = len(relaxMask)
    else:
        crtIntervalEndOffset = crtIntervalStartOffset + int(relaxedFailures[0])
    if crtIntervalEndOffset <= crtIntervalStartOffset:
        raise ValueError("Detected CRT interval end is not after its start.")

    crtIntervalStartOffset = int(sourceOffsets[crtIntervalStartOffset])
    if crtIntervalEndOffset == len(sourceOffsets):
        crtIntervalEndOffset = int(sourceOffsets[-1]) + 1
    else:
        crtIntervalEndOffset = int(sourceOffsets[crtIntervalEndOffset])

    intervalTimes = timeArr[
        releaseIndex + crtIntervalStartOffset : releaseIndex + crtIntervalEndOffset
    ]
    if len(intervalTimes) < 2:
        raise ValueError("Detected CRT interval has fewer than two samples.")
    endTimeGaussian = min(intervalTimes[0] + 1, intervalTimes[-1])
    endIndexGaussian = (
        crtIntervalStartOffset
        + int(np.argmin(np.abs(intervalTimes - endTimeGaussian)))
        + 1
    )
    endIndexGaussian = min(
        max(endIndexGaussian, crtIntervalStartOffset + 1),
        crtIntervalEndOffset,
    )
    labAStartIntervalSmooth = gaussian_filter1d(
        searchSignal[crtIntervalStartOffset:endIndexGaussian],
        sigma=3,
    )
    startIndex = (
        releaseIndex + crtIntervalStartOffset + int(np.argmax(labAStartIntervalSmooth))
    )
    crtIntervalStartIndex = releaseIndex + crtIntervalStartOffset
    crtIntervalEndIndex = releaseIndex + crtIntervalEndOffset
    if offsetTime != 0:
        offsetStartTime = timeArr[crtIntervalStartIndex] + offsetTime
        offsetStartIndex = int(np.argmin(np.abs(timeArr - offsetStartTime)))
        crtIntervalStartIndex = int(
            np.clip(offsetStartIndex, 0, crtIntervalEndIndex - 1)
        )

    return int(crtIntervalStartIndex), int(startIndex), int(crtIntervalEndIndex)


# }}}


def calculateCrtMetrics(
    labAIntensArr: np.ndarray,
    bgrGIntensArr: np.ndarray,
    timeArr: np.ndarray,
    crtIntervalStartIndex: int,
    startIndex: int,
    crtIntervalEndIndex: int,
    maxUncertaintyRatio: float = np.inf,
) -> tuple[
    dict[str, tuple[float, float]],
    dict[str, dict[str, Any]],
    dict[str, str],
]:
    # {{{
    labAIntensArr = np.asarray(labAIntensArr, dtype=float)
    bgrGIntensArr = np.asarray(bgrGIntensArr, dtype=float)
    timeArr = np.asarray(timeArr, dtype=float)
    crtIntervalStartIndex = int(crtIntervalStartIndex)
    startIndex = int(startIndex)
    crtIntervalEndIndex = int(crtIntervalEndIndex)
    if not (
        0 <= crtIntervalStartIndex < crtIntervalEndIndex <= len(timeArr)
        and 0 <= startIndex < crtIntervalEndIndex
    ):
        raise ValueError(
            "Expected indices satisfying "
            "0 <= crtIntervalStartIndex < crtIntervalEndIndex <= len(timeArr) "
            "and 0 <= startIndex < crtIntervalEndIndex."
        )

    result: dict[str, tuple[float, float]] = {}
    details: dict[str, dict[str, Any]] = {}
    errors: dict[str, str] = {}
    for channelName, signalArr in (
        ("bgr_g", bgrGIntensArr),
        ("lab_a", -1 * labAIntensArr),
    ):
        pcrtKey = f"{channelName}_pcrt"
        try:
            pcrt = calculatePcrtDetails(
                signalArr,
                timeArr,
                crtIntervalStartIndex,
                startIndex,
                crtIntervalEndIndex,
                maxUncertaintyRatio=maxUncertaintyRatio,
            )
        except Exception as err:
            errors[pcrtKey] = str(err)
        else:
            result[pcrtKey] = (
                pcrt["pcrt"],
                pcrt["pcrtUncertainty"],
            )
            details[pcrtKey] = pcrt

        crt90Key = f"{channelName}_crt90_10"
        try:
            crt90 = calculateCrt90_10Details(
                signalArr,
                timeArr,
                crtIntervalStartIndex,
                crtIntervalEndIndex,
            )
            crt90Value = crt90["crt90_10"]
            crt90Times = crt90["times"]
            crt90Values = crt90["rawValues"]
            smoothedValues = crt90["smoothedRawValues"]
            sigmaSamples = crt90["sigmaSamples"]
            residuals = crt90Values - smoothedValues
            noiseSigma = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 0.0
            rng = np.random.default_rng(0)
            bootstrapSamples = []
            for _ in range(500):
                syntheticValues = crt90Values + rng.normal(
                    0.0,
                    noiseSigma,
                    size=len(crt90Values),
                )
                if sigmaSamples > 0:
                    syntheticSmooth = gaussian_filter1d(
                        syntheticValues,
                        sigma=sigmaSamples,
                    )
                else:
                    syntheticSmooth = syntheticValues
                syntheticNorm = normalizeIntensities(syntheticSmooth)
                syntheticStartOffset = int(np.argmax(syntheticNorm))
                syntheticValue90 = 0.9 * syntheticNorm[syntheticStartOffset]
                syntheticValue10 = 0.1 * syntheticNorm[syntheticStartOffset]
                synthetic90Candidates = np.flatnonzero(
                    syntheticNorm[syntheticStartOffset:] < syntheticValue90
                )
                if len(synthetic90Candidates) == 0:
                    continue
                syntheticTime90Offset = syntheticStartOffset + int(
                    synthetic90Candidates[0]
                )
                synthetic10Candidates = np.flatnonzero(
                    syntheticNorm[syntheticTime90Offset:] < syntheticValue10
                )
                if len(synthetic10Candidates) == 0:
                    continue
                syntheticTime10Offset = syntheticTime90Offset + int(
                    synthetic10Candidates[0]
                )
                bootstrapSamples.append(
                    crt90Times[syntheticTime10Offset]
                    - crt90Times[syntheticTime90Offset]
                )
            if len(bootstrapSamples) == 0:
                raise ValueError("All CRT90_10 bootstrap iterations failed.")
            ci95 = np.percentile(np.asarray(bootstrapSamples), [2.5, 97.5])
        except Exception as err:
            errors[crt90Key] = str(err)
        else:
            crt90Uncertainty = float((ci95[1] - ci95[0]) / 2)
            uncertaintyRatio = validateUncertaintyRatio(
                "CRT90_10",
                crt90Value,
                crt90Uncertainty,
                maxUncertaintyRatio,
            )
            result[crt90Key] = (
                float(crt90Value),
                crt90Uncertainty,
            )
            details[crt90Key] = {
                **crt90,
                "crt90_10UncertaintyRatio": float(uncertaintyRatio),
            }

    if len(result) == 0:
        errorSummary = "; ".join(
            f"{key}: {value}" for key, value in sorted(errors.items())
        )
        raise ValueError(f"All CRT measurements failed. {errorSummary}")

    return result, details, errors

# }}}


def isValidRoi(obj: Any) -> bool:
    # {{{
    try:
        t = tuple(obj)
    except TypeError:
        return False
    return len(t) == 4


# }}}


def findUniquePath(path: str | Path) -> Path:
    # {{{
    path = Path(path)
    if not path.exists():
        return path

    stem = path.stem
    suffix = path.suffix
    parent = path.parent

    i = 1
    while True:
        candidate = parent / f"{stem} ({i}){suffix}"
        if not candidate.exists():
            return candidate
        i += 1


# }}}


def savePlots(
    pcrtObj: PCRT,
    videoName: str,
    plotPath: Path | str,
    overwrite: bool = False,
) -> None:
    # {{{
    plotPath = Path(plotPath)
    if not plotPath.exists() or not plotPath.is_dir():
        raise ValueError(f"{plotPath} does not exist or is not a directory.")
    avgIntensPath = plotPath / f"{videoName}_avgIntens.png"
    pcrtPath = plotPath / f"{videoName}_pCRTPlot.png"
    if not overwrite:
        avgIntensPath = findUniquePath(avgIntensPath)
        pcrtPath = findUniquePath(pcrtPath)
    pcrtObj.saveAvgIntensPlot(avgIntensPath)
    LOGGER.info(f"AvgIntens plot saved in {avgIntensPath}")
    pcrtObj.savePCRTPlot(pcrtPath)
    LOGGER.info(f"CRTPlot plot saved in {pcrtPath}")


# }}}


CRT_METRIC_KEYS = (
    "bgr_g_pcrt",
    "bgr_g_crt90_10",
    "lab_a_pcrt",
    "lab_a_crt90_10",
)


def metricColumnPrefix(metricKey: str) -> str:
    # {{{
    return metricKey


# }}}


def measurementVideoName(
    measurement: dict[str, Any],
    videoName: str | None = None,
) -> str:
    # {{{
    if videoName is not None:
        return str(videoName)
    videoPath = measurement.get("videoPath")
    if videoPath is not None:
        return Path(videoPath).name
    return str(measurement.get("videoName", "unknown_video"))


# }}}


def measurementTimeString(measurement: dict[str, Any]) -> str:
    # {{{
    measurementTime = measurement.get("measurementTime")
    if measurementTime is None:
        return datetime.now().strftime(DATETIME_FORMAT)
    if isinstance(measurementTime, datetime):
        return measurementTime.strftime(DATETIME_FORMAT)
    return str(measurementTime)


# }}}


def metricSummaryValues(measurement: dict[str, Any], metricKey: str) -> dict[str, Any]:
    # {{{
    metrics = measurement.get("metrics", {})
    details = measurement.get("metricDetails", {})
    metricValue = np.nan
    uncertainty = np.nan
    if metricKey in metrics:
        metricValue, uncertainty = metrics[metricKey]

    metricDetails = details.get(metricKey, {})
    return {
        "value": float(metricValue) if np.isfinite(metricValue) else np.nan,
        "uncertainty": float(uncertainty) if np.isfinite(uncertainty) else np.nan,
        "criticalTime": float(metricDetails.get("criticalTime", np.nan)),
    }


# }}}


def measurementCSVColumns() -> list[str]:
    # {{{
    columns = ["Video", "releaseTime", "measurementTime"]
    for metricKey in CRT_METRIC_KEYS:
        prefix = metricColumnPrefix(metricKey)
        columns.extend(
            [
                prefix,
                f"{prefix}_uncertainty",
                f"{prefix}_criticalTime",
            ]
        )
    return columns


# }}}


def measurementCSVRow(
    measurement: dict[str, Any],
    videoName: str | None = None,
) -> dict[str, Any]:
    # {{{
    row = {
        "Video": measurementVideoName(measurement, videoName),
        "releaseTime": float(measurement.get("releaseTime", np.nan)),
        "measurementTime": measurementTimeString(measurement),
    }
    for metricKey in CRT_METRIC_KEYS:
        prefix = metricColumnPrefix(metricKey)
        summary = metricSummaryValues(measurement, metricKey)
        row[prefix] = summary["value"]
        row[f"{prefix}_uncertainty"] = summary["uncertainty"]
        row[f"{prefix}_criticalTime"] = summary["criticalTime"]
    return row


# }}}


def saveCSV(
    measurement: dict[str, Any],
    csvPath: Path | str,
    videoName: str | None = None,
    overwrite: bool = False,
    provideXLSX: bool = True,
) -> None:
    # {{{
    csvPath = Path(csvPath)
    csvPath.parent.mkdir(parents=True, exist_ok=True)
    row = measurementCSVRow(measurement, videoName)
    colLabels = measurementCSVColumns()

    if csvPath.exists():
        csvSheet = pd.read_csv(csvPath, sep=",", encoding="utf-8-sig")
        if "Video" not in csvSheet.columns and len(csvSheet.columns) > 0:
            firstColumn = csvSheet.columns[0]
            if str(firstColumn).startswith("Unnamed"):
                csvSheet = csvSheet.rename(columns={firstColumn: "Video"})

        for column in colLabels:
            if column not in csvSheet.columns:
                csvSheet[column] = np.nan

        extraColumns = [column for column in csvSheet.columns if column not in colLabels]
        csvSheet = csvSheet[colLabels + extraColumns]
        if "Video" not in csvSheet:
            raise RuntimeError(f"Could not load {csvPath}. CSV is missing Video column.")

        videoNameToSave = row["Video"]
        matchingRows = csvSheet.index[csvSheet["Video"] == videoNameToSave].tolist()
        if matchingRows and overwrite:
            csvSheet.loc[matchingRows[0], colLabels] = [
                row[column] for column in colLabels
            ]
        else:
            if matchingRows:
                oldVideoName = videoNameToSave
                videoNameToSave = findUniqueName(oldVideoName, list(csvSheet["Video"]))
                row["Video"] = videoNameToSave
                LOGGER.info(f"{oldVideoName} found in CSV, saving as {videoNameToSave}")
            csvSheet.loc[len(csvSheet), colLabels] = [row[column] for column in colLabels]
    else:
        csvSheet = pd.DataFrame([row], columns=colLabels)

    csvSheet.to_csv(csvPath, index=False, encoding="utf-8-sig")
    if provideXLSX:
        xlsxPath = csvPath.with_suffix(".xlsx")
        csvSheet.to_excel(xlsxPath, index=False, engine="openpyxl")
    LOGGER.info(
        f"Saved CRT in {csvPath}" + (f" and in {xlsxPath}" if provideXLSX else "")
    )


# }}}


def safeJsonDumps(value: Any) -> str:
    # {{{
    def default(obj):
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, datetime):
            return obj.strftime(DATETIME_FORMAT)
        return str(obj)

    return json.dumps(value, default=default, sort_keys=True)


# }}}


def measurementNpzData(
    measurement: dict[str, Any],
    videoName: str | None = None,
) -> dict[str, Any]:
    # {{{
    timeArr = np.asarray(measurement["timeArr"], dtype=float)
    labAIntensArr = np.asarray(measurement["labAIntensArr"], dtype=float)
    bgrGIntensArr = np.asarray(measurement["bgrGIntensArr"], dtype=float)
    crtIntervalStartIndex = int(measurement["crtIntervalStartIndex"])
    crtIntervalEndIndex = int(measurement["crtIntervalEndIndex"])
    intervalSlice = slice(crtIntervalStartIndex, crtIntervalEndIndex)

    data: dict[str, Any] = {
        "videoName": measurementVideoName(measurement, videoName),
        "videoPath": str(measurement.get("videoPath", "")),
        "roi": np.asarray(measurement.get("roi", [])),
        "measurementTime": measurementTimeString(measurement),
        "releaseIndex": int(measurement.get("releaseIndex", -1)),
        "releaseTime": float(measurement.get("releaseTime", np.nan)),
        "crtIntervalStartIndex": crtIntervalStartIndex,
        "startIndex": int(measurement.get("startIndex", -1)),
        "crtIntervalEndIndex": crtIntervalEndIndex,
        "timeArr": timeArr,
        "labAIntensArr": labAIntensArr,
        "bgrGIntensArr": bgrGIntensArr,
        "slicedTimeArr": timeArr[intervalSlice],
        "slicedLabAIntensArr": labAIntensArr[intervalSlice],
        "slicedBgrGIntensArr": bgrGIntensArr[intervalSlice],
        "releaseParamsJson": safeJsonDumps(measurement.get("releaseParams", {})),
        "metricErrorsJson": safeJsonDumps(measurement.get("metricErrors", {})),
    }

    releaseMetricData = measurement.get("releaseMetricData") or {}
    data["releaseMetricTimes"] = np.asarray(
        releaseMetricData.get("times", []),
        dtype=float,
    )
    data["releaseMetricValues"] = np.asarray(
        releaseMetricData.get("values", []),
        dtype=float,
    )
    data["releaseMetricLabel"] = str(releaseMetricData.get("label", ""))

    for metricKey in CRT_METRIC_KEYS:
        summary = metricSummaryValues(measurement, metricKey)
        data[metricKey] = summary["value"]
        data[f"{metricKey}_uncertainty"] = summary["uncertainty"]
        data[f"{metricKey}_criticalTime"] = summary["criticalTime"]

    return data


# }}}


def saveNpz(
    measurement: dict[str, Any],
    npzPath: Path | str,
    videoName: str | None = None,
    overwrite: bool = False,
) -> None:
    # {{{
    npzPath = Path(npzPath)
    npzPath.mkdir(parents=True, exist_ok=True)
    if not npzPath.is_dir():
        raise ValueError(f"{npzPath} is not a directory.")

    npzFilePath = (
        npzPath / f"{Path(measurementVideoName(measurement, videoName)).stem}.npz"
    )
    if not overwrite:
        npzFilePath = findUniquePath(npzFilePath)

    np.savez_compressed(npzFilePath, **measurementNpzData(measurement, videoName))
    LOGGER.info(f"CRT measurement data saved in {npzFilePath}.")


# }}}


def saveMeasurementOutputsFromConfig(
    measurement: dict[str, Any],
    configDict: dict[str, Any],
) -> None:
    # {{{
    filesConfig = configDict["Files"]
    overwrite = bool(filesConfig.get("overwrite", False))

    csvPath = filesConfig.get("csvPath")
    if csvPath:
        saveCSV(
            measurement,
            csvPath,
            overwrite=overwrite,
            provideXLSX=bool(filesConfig.get("provideXLSX", True)),
        )

    npzPath = filesConfig.get("npzPath")
    if npzPath:
        saveNpz(
            measurement,
            npzPath,
            overwrite=overwrite,
        )


def findUniqueName(name: str, nameList: list[str]) -> str:
    # {{{
    i = 1
    while True:
        candidate = f"{name} ({i})"
        if candidate not in nameList:
            return candidate
        i += 1


# }}}


def roiToString(roi: RoiTuple) -> str:
    # {{{
    if not isValidRoi:
        raise ValueError(
            f"{roi} is not a valid value for the ROI. "
            "Valid values are 4-element iterables."
        )
    return str(roi).replace(",", "c")


# }}}


def stringToRoi(roiString: str) -> RoiTuple:
    # {{{
    if not isinstance(roiString, str):
        raise TypeError("The argument to stringToRoi must be a string")
    numbers = roiString.strip()[1:-2]
    roi = tuple(int(x) for x in numbers.split("c "))
    return roi


# }}}


def singleVideoPipeline(
    configPath: Path | str,
    defaultsPath: Path | str,
) -> bool:
    # {{{
    configDict = loadConfigFile(configPath, defaultsPath)

    LOGGER.setLevel(configDict["General"]["logLevel"])
    LOGGER.info(f"Config loaded from {configPath}.")
    if configDict["General"]["writeLogFile"]:
        enableFileLogging(LOGGER)

    crtVideoPath = gui.selectFile()
    videoName = crtVideoPath.stem

    try:
        measurement = measureCRTVideoFromConfig(
            crtVideoPath,
            configDict,
            savePlot=bool(configDict["General"].get("showAllPlots", configDict["General"].get("showPlots", False)) or configDict["General"].get("showBGRPlot", False) or configDict["General"].get("showLABPlot", False) or configDict["General"].get("showEdgeDetectionPlot", False)),
        )
        saveMeasurementOutputsFromConfig(measurement, configDict)
    except Exception as e:
        LOGGER.error(f"CRT calculation failed on {crtVideoPath}:\n{e}")
        return False

    LOGGER.log(NORM_LEVEL, f"{videoName}: {measurement['metrics']}")
    return True


# }}}


def multiVideoPipeline(
    configPath: Path | str,
    defaultsPath: Path | str,
) -> int:
    # {{{
    configDict = loadConfigFile(configPath, defaultsPath)

    LOGGER.setLevel(configDict["General"]["logLevel"])
    LOGGER.info(f"Config loaded from {configPath}.")
    if configDict["General"]["writeLogFile"]:
        enableFileLogging(LOGGER)

    askConfirmation = configDict["General"]["askConfirmation"]
    processedPaths = []
    failedMeasurements = 0
    showPlots = bool(configDict["General"].get("showAllPlots", configDict["General"].get("showPlots", False)) or configDict["General"].get("showBGRPlot", False) or configDict["General"].get("showLABPlot", False) or configDict["General"].get("showEdgeDetectionPlot", False))

    dirPath = gui.selectDirectory()
    for candidatePath in dirPath.iterdir():
        if not candidatePath.is_file():
            LOGGER.debug(
                f"Skipping {candidatePath.stem}{candidatePath.suffix} (not a file)..."
            )
            continue
        if candidatePath.suffix not in VIDEO_FORMATS:
            LOGGER.debug(
                f"Skipping {candidatePath.stem}{candidatePath.suffix} (not a video)..."
            )
            continue
        if candidatePath in processedPaths:
            LOGGER.debug(
                f"Skipping {candidatePath.stem}{candidatePath.suffix} "
                "(already processed)..."
            )
            continue
        if askConfirmation:
            answer = gui.askNextVideo(candidatePath)
            if answer == "skip":
                processedPaths.append(candidatePath)
                continue
            if answer == "abort":
                break
            if answer == "select":
                actualPath = gui.selectFile()
            else:
                actualPath = candidatePath
        else:
            actualPath = candidatePath

        LOGGER.info(f"Processing video {actualPath}.")
        videoName = actualPath.stem
        try:
            measurement = measureCRTVideoFromConfig(
                actualPath,
                configDict,
                savePlot=showPlots,
            )
            saveMeasurementOutputsFromConfig(measurement, configDict)
            processedPaths.append(actualPath)
        except Exception as e:
            LOGGER.error(f"CRT calculation failed on {actualPath}:\n{e}")
            failedMeasurements += 1
            processedPaths.append(actualPath)
            continue

        LOGGER.log(NORM_LEVEL, f"{videoName}: {measurement['metrics']}")

    LOGGER.info(f"Finished processing videos in {dirPath}")

    return failedMeasurements


# }}}
