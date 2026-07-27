import csv
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-pycrt")

from crt_signal_processing import (
    DEFAULT_CRT90_10_BOOTSTRAP_COUNT,
    DEFAULT_CRT90_10_BOOTSTRAP_SEED,
    DEFAULT_CRT90_10_GAUSSIAN_SIGMA_SECONDS,
    DEFAULT_MAX_CRT90_10_FIT_SECONDS,
    calcCRT90_10BootstrapUncertainty,
    calcCRT90_10Gaussian,
    calcPCRTFit,
)
from funcs import loadVideoRoi
from release_frame_processing import (
    cannySumsFromLFrames,
    findReleaseIndex,
    laplacianSumsFromLFrames,
    optimizableFindCrtIntervalIndices,
    releaseParamsFromLaplacianParams,
)

ROIS_PATH = "rois_full.toml"
CACHE_DIR = Path("Npz/Cache")
OUTPUT_CSV_PATH = Path("crt_pipeline_comparison.csv")

VIDEOS_DIR_LIST = [
    # Path("/home/eduardo/Data/miscVideos"),
    Path("/home/eduardo/Data/raquelMasters"),
    # Path("/home/eduardo/Data/yutaoNewEquipment"),
    # Path("/home/eduardo/Data/yutaoPostNew"),
    # Path("TrainingVideos"),
]
RUN_ON = VIDEOS_DIR_LIST
VIDEO_EXTENSIONS = [".MOV", ".wmv", ".mp4"]

FILTER_TYPE = "canny"
# FILTER_TYPE = "laplacian"

RELAX_MAX_GRAD = 1.0
RELAX_MIN_GRAD = -3.0
STRICT_MAX_GRAD = 1
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
LAPLACIAN_PARAMS = {
    "ksize": 3,
    "blurKernel": 7,
    "scale": 1,
    "delta": 0,
    "laplacianPlateau": 2.0,
    "laplacianSmoothingKernel": 19,
    "laplacianGradientSmoothingKernel": 3,
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

ERROR_ROI = -1
ERROR_CACHE = -2
ERROR_METRIC = -3
ERROR_CRT_INTERVAL = -4
ERROR_CRT90_10 = -5
ERROR_CRT90_10_UNCERTAINTY = -6
ERROR_PCRT = -7


def iterVideoPathsInDirectory(videoDir):
    for filePath in sorted(
        Path(videoDir).iterdir(), key=lambda path: path.name.lower()
    ):
        if filePath.suffix.lower() in {suffix.lower() for suffix in VIDEO_EXTENSIONS}:
            yield filePath


def iterRunOnVideoPaths(runOn):
    if isinstance(runOn, (list, tuple, set)):
        for item in runOn:
            yield from iterRunOnVideoPaths(item)
        return

    runOnPath = Path(runOn)
    if runOnPath.is_dir():
        yield from iterVideoPathsInDirectory(runOnPath)
        return

    yield runOnPath


def cachePathForVideo(videoPath):
    return CACHE_DIR / f"{videoPath.stem}.npz"


def roiToString(roi):
    if roi is None:
        return str(ERROR_ROI)
    return ",".join(str(value) for value in roi)


def loadCachedVideo(videoPath):
    cachePath = cachePathForVideo(videoPath)
    if not cachePath.exists():
        raise FileNotFoundError(f"No cache found for {videoPath.name}: {cachePath}")

    with np.load(cachePath) as data:
        missingKeys = {"lFrames", "timesScdsArr", "avgAArr", "avgGArr"} - set(
            data.files
        )
        if missingKeys:
            raise KeyError(
                f"Cache {cachePath} is missing key(s): {', '.join(sorted(missingKeys))}"
            )
        return {
            "lFrames": data["lFrames"],
            "timesScdsArr": data["timesScdsArr"].astype(float),
            "avgAArr": data["avgAArr"].astype(float),
            "avgGArr": data["avgGArr"].astype(float),
        }


def paramsForFilter():
    if FILTER_TYPE == "canny":
        return CANNY_PARAMS, CANNY_PARAMS
    if FILTER_TYPE == "laplacian":
        return LAPLACIAN_PARAMS, releaseParamsFromLaplacianParams(LAPLACIAN_PARAMS)
    raise ValueError(f"Unsupported FILTER_TYPE: {FILTER_TYPE}")


def metricArrFromCachedFrames(lFrames, params):
    if FILTER_TYPE == "canny":
        return cannySumsFromLFrames(lFrames, params)
    if FILTER_TYPE == "laplacian":
        return laplacianSumsFromLFrames(lFrames, params)
    raise ValueError(f"Unsupported FILTER_TYPE: {FILTER_TYPE}")


def emptyMeasurement(errorCode):
    return {
        "crt90_10": errorCode,
        "crt90_10_uncertainty": errorCode,
        "pcrt": errorCode,
        "pcrt_uncertainty": errorCode,
    }


def measureChannel(signalArr, timeArr, metricArr, releaseParams):
    try:
        releaseIndex = findReleaseIndex(metricArr, timeArr, releaseParams)
        indices = optimizableFindCrtIntervalIndices(
            signalArr,
            timeArr,
            releaseIndex,
            releaseParams["strictMinGrad"],
            releaseParams["strictMaxGrad"],
            releaseParams["relaxMinGrad"],
            releaseParams["relaxMaxGrad"],
        )
    except Exception as err:
        print(f"  CRT interval failed: {type(err).__name__}: {err}")
        return emptyMeasurement(ERROR_CRT_INTERVAL)

    try:
        crt90_10 = calcCRT90_10Gaussian(
            timeArr,
            signalArr,
            indices,
            maxFitSeconds=MAX_CRT90_10_FIT_SECONDS,
            gaussianSigmaSeconds=CRT90_10_GAUSSIAN_SIGMA_SECONDS,
        )
    except Exception as err:
        print(f"  CRT90_10 failed: {type(err).__name__}: {err}")
        crt90_10Value = ERROR_CRT90_10
        crt90_10UncertaintyValue = ERROR_CRT90_10
    else:
        crt90_10Value = crt90_10["crt90_10"]
        try:
            crt90_10Uncertainty = calcCRT90_10BootstrapUncertainty(
                timeArr,
                signalArr,
                indices,
                maxFitSeconds=MAX_CRT90_10_FIT_SECONDS,
                gaussianSigmaSeconds=CRT90_10_GAUSSIAN_SIGMA_SECONDS,
                nBootstraps=CRT90_10_BOOTSTRAP_COUNT,
                randomSeed=CRT90_10_BOOTSTRAP_SEED,
            )
        except Exception as err:
            print(f"  CRT90_10 uncertainty failed: {type(err).__name__}: {err}")
            crt90_10Value = ERROR_CRT90_10_UNCERTAINTY
            crt90_10UncertaintyValue = ERROR_CRT90_10_UNCERTAINTY
        else:
            ci95Low, ci95High = crt90_10Uncertainty["crt90_10_ci95"]
            crt90_10Value = crt90_10Uncertainty["crt90_10_mean"]
            crt90_10UncertaintyValue = (ci95High - ci95Low) / 2

    pcrtFit = calcPCRTFit(timeArr, signalArr, indices)
    if pcrtFit is None:
        pcrtValue = ERROR_PCRT
        pcrtUncertainty = ERROR_PCRT
    else:
        pcrtValue = pcrtFit["pcrt"]
        pcrtUncertainty = pcrtFit["pcrtUncertainty"]

    return {
        "crt90_10": float(crt90_10Value),
        "crt90_10_uncertainty": float(crt90_10UncertaintyValue),
        "pcrt": float(pcrtValue),
        "pcrt_uncertainty": float(pcrtUncertainty),
    }


def formatMeasurement(value, uncertainty):
    if value < 0 or uncertainty < 0:
        return str(int(value if value < 0 else uncertainty))
    return f"{value:.3f}+-{uncertainty:.3f}"


def measureVideo(videoPath):
    try:
        roi = loadVideoRoi(videoPath, ROIS_PATH)
    except (KeyError, ValueError) as err:
        print(f"  ROI failed: {type(err).__name__}: {err}")
        return (
            roiToString(None),
            emptyMeasurement(ERROR_ROI),
            emptyMeasurement(ERROR_ROI),
        )

    try:
        cachedVideo = loadCachedVideo(videoPath)
    except Exception as err:
        print(f"  Cache failed: {type(err).__name__}: {err}")
        return (
            roiToString(roi),
            emptyMeasurement(ERROR_CACHE),
            emptyMeasurement(ERROR_CACHE),
        )

    filterParams, releaseParams = paramsForFilter()
    try:
        metricArr = metricArrFromCachedFrames(cachedVideo["lFrames"], filterParams)
    except Exception as err:
        print(f"  Metric failed: {type(err).__name__}: {err}")
        return (
            roiToString(roi),
            emptyMeasurement(ERROR_METRIC),
            emptyMeasurement(ERROR_METRIC),
        )

    gMeasurement = measureChannel(
        cachedVideo["avgGArr"],
        cachedVideo["timesScdsArr"],
        metricArr,
        releaseParams,
    )
    aMeasurement = measureChannel(
        cachedVideo["avgAArr"],
        cachedVideo["timesScdsArr"],
        metricArr,
        releaseParams,
    )
    return roiToString(roi), gMeasurement, aMeasurement


def rowForVideo(videoPath, roi, gMeasurement, aMeasurement):
    return {
        "video": videoPath.name,
        "roi": roi,
        "g_crt90_10_s": gMeasurement["crt90_10"],
        "g_crt90_10_uncertainty_s": gMeasurement["crt90_10_uncertainty"],
        "g_pcrt_s": gMeasurement["pcrt"],
        "g_pcrt_uncertainty_s": gMeasurement["pcrt_uncertainty"],
        "a_crt90_10_s": aMeasurement["crt90_10"],
        "a_crt90_10_uncertainty_s": aMeasurement["crt90_10_uncertainty"],
        "a_pcrt_s": aMeasurement["pcrt"],
        "a_pcrt_uncertainty_s": aMeasurement["pcrt_uncertainty"],
    }


def progressString(videoPath, gMeasurement, aMeasurement, videoNum, totalVideos):
    gPcrt = formatMeasurement(
        gMeasurement["pcrt"],
        gMeasurement["pcrt_uncertainty"],
    )
    gCrt90_10 = formatMeasurement(
        gMeasurement["crt90_10"],
        gMeasurement["crt90_10_uncertainty"],
    )
    aPcrt = formatMeasurement(
        aMeasurement["pcrt"],
        aMeasurement["pcrt_uncertainty"],
    )
    aCrt90_10 = formatMeasurement(
        aMeasurement["crt90_10"],
        aMeasurement["crt90_10_uncertainty"],
    )
    return (
        f"measured {videoPath.name}: {gPcrt}, {gCrt90_10}, "
        f"{aPcrt}, {aCrt90_10}, {videoNum}/{totalVideos}"
    )


def main():
    videoPaths = list(iterRunOnVideoPaths(RUN_ON))
    fieldNames = [
        "video",
        "roi",
        "g_crt90_10_s",
        "g_crt90_10_uncertainty_s",
        "g_pcrt_s",
        "g_pcrt_uncertainty_s",
        "a_crt90_10_s",
        "a_crt90_10_uncertainty_s",
        "a_pcrt_s",
        "a_pcrt_uncertainty_s",
    ]

    with OUTPUT_CSV_PATH.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldNames)
        writer.writeheader()

        totalVideos = len(videoPaths)
        for videoNum, videoPath in enumerate(videoPaths, start=1):
            roi, gMeasurement, aMeasurement = measureVideo(videoPath)
            writer.writerow(rowForVideo(videoPath, roi, gMeasurement, aMeasurement))
            print(
                progressString(
                    videoPath,
                    gMeasurement,
                    aMeasurement,
                    videoNum,
                    totalVideos,
                ),
                flush=True,
            )

    print(f"wrote {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    main()
