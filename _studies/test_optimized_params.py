import os
from pathlib import Path

import cv2 as cv
import numpy as np
import tomllib

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-pycrt")

from crt_signal_processing import (
    DEFAULT_CRT90_10_BOOTSTRAP_COUNT,
    DEFAULT_CRT90_10_BOOTSTRAP_SEED,
    DEFAULT_CRT90_10_GAUSSIAN_SIGMA_SECONDS,
    DEFAULT_MAX_CRT90_10_FIT_SECONDS,
    calcCRT90_10BootstrapUncertainty,
    calcCRT90_10Gaussian,
    calcPCRTFit,
    calcSignalGradientDiagnostics,
)
from funcs import laplacianEdgeDetector, loadVideoRoi
from matplotlib import pyplot as plt
from release_frame_processing import (
    calcGradientWithFilteredSamples,
    cannySumFromLFrame,
    findReleaseIndex,
    laplacianSumFromLFrame,
    minMaxNormalizeOrZeros,
    normalizeCannyArr,
    optimizableFindCrtIntervalIndices,
    releaseParamsFromLaplacianParams,
    smoothCannyArr,
    smoothCannyGradientArr,
)

plt.style.use("bmh")

ROIS_PATH = "rois_full.toml"
CACHE_DIR = Path("Npz/Cache")
TARGETS_PATH = Path("target_frames.toml")
OPTIMIZED_CANNY_PARAMS_PATH = Path("optimized_params_canny.toml")
OPTIMIZED_LAPLACIAN_PARAMS_PATH = Path("optimized_params_laplacian.toml")

VIDEOS_DIR_LIST = [
    # Path("/home/eduardo/Data/miscVideos"),
    # Path("/home/eduardo/Data/raquelMasters"),
    Path("/home/eduardo/Data/yutaoNewEquipment"),
    # Path("/home/eduardo/Data/yutaoPostNew"),
    # Path("TrainingVideos"),
]
VIDEO_EXTENSIONS = [".MOV", ".wmv", ".mp4"]
problem_paths = [
    # Path("TrainingVideos") / "experiment 2.wmv",
    Path("/home/eduardo/Data/raquelMasters") / "P01CR3.wmv",
    Path("/home/eduardo/Data/raquelMasters") / "P01CR4.wmv",
    Path("/home/eduardo/Data/raquelMasters") / "P02CR1.wmv",
    Path("/home/eduardo/Data/raquelMasters") / "P02CR4.wmv",
    Path("/home/eduardo/Data/raquelMasters") / "P02CR5.wmv",
    # Path("/home/eduardo/Data/raquelMasters") / "P02CR5.wmv",
]
# RUN_ON = problem_paths
RUN_ON = VIDEOS_DIR_LIST
FILTER_TYPE = "canny"
# FILTER_TYPE = "laplacian"


def loadTargets():
    with TARGETS_PATH.open("rb") as file:
        targetConfig = tomllib.load(file)

    return {
        videoName: int(targetIndex)
        for videoName, targetIndex in targetConfig["targets"].items()
    }


TARGETS = loadTargets()
VIDEO_CACHE = {}
# RELAX_MAX_GRAD = 1.0
# RELAX_MIN_GRAD = -3.0
# STRICT_MAX_GRAD = 1
# STRICT_MIN_GRAD = -0.5
RELAX_MAX_GRAD = 5.0
RELAX_MIN_GRAD = -10.0
STRICT_MAX_GRAD = 1
STRICT_MIN_GRAD = -0.5

# RELAX_MAX_GRAD = np.inf
# RELAX_MIN_GRAD = -RELAX_MAX_GRAD
# STRICT_MAX_GRAD = np.inf
# STRICT_MIN_GRAD = -STRICT_MAX_GRAD

CANNY_GRADIENT_SMOOTHING_KERNEL = 3
DEFAULT_CANNY_PARAMS = {
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
DEFAULT_LAPLACIAN_PARAMS = {
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


def loadOptimizedParamsToml(paramsPath, defaultParams, required=False):
    if not paramsPath.exists():
        if required:
            raise FileNotFoundError(f"Optimized params TOML not found: {paramsPath}")
        print(f"Using default params: {paramsPath} not found")
        return dict(defaultParams)

    with paramsPath.open("rb") as file:
        config = tomllib.load(file)

    optimizedParams = config.get("params")
    if not isinstance(optimizedParams, dict):
        raise ValueError(f"No [params] table found in {paramsPath}")

    return {**defaultParams, **optimizedParams}


CANNY_PARAMS = loadOptimizedParamsToml(
    OPTIMIZED_CANNY_PARAMS_PATH,
    DEFAULT_CANNY_PARAMS,
    required=FILTER_TYPE == "canny",
)
LAPLACIAN_PARAMS = loadOptimizedParamsToml(
    OPTIMIZED_LAPLACIAN_PARAMS_PATH,
    DEFAULT_LAPLACIAN_PARAMS,
    required=FILTER_TYPE == "laplacian",
)
params = CANNY_PARAMS if FILTER_TYPE == "canny" else LAPLACIAN_PARAMS

# TENTATIVE_TIMESTAMPS = {{{{
#     "0528_test2.1.mp4": 4.93,
#     "0528_test2.2.mp4": 4.83,
#     "0528_test3.2.mp4": 4.77,
#     "0528_test3.6.mp4": 4.93,
#     "0528_test4.2.mp4": 4.8,
#     "CRT1.mp4": 12.10,
#     "CRT2.mp4": 12.17,
#     "CRT3.mp4": 12.17,
#     "DSC_0069.MOV": 11.34,
#     "P02CR1.wmv": 17.20,
#     "P02CR4.wmv": 17.20,
#     "P06CR1.wmv": 17.4,
#     "P09CR1.wmv": 17.4,
#     "P15CR2.wmv": 19.4,
#     "P19CR5.wmv": 17.9,
#     "P06CR3.wmv": 17.0,
#     "P06CR4.wmv": 16.6,
#     "P20CR1.wmv": 16.0,
#     "P20CR2.wmv": 16.5,
#     "P20CR3.wmv": 17.0,
#     "P26CR1.wmv": 19.30,
#     "P26CR2.wmv": 17.54,
#     "P26CR3.wmv": 18.59,
#     "P30CR3.wmv": 17.41,
#     "P30CR4.wmv": 17.41,
#     "P31CR2.wmv": 17.51,
#     "P31CR5.wmv": 17.00,
#     "Video 142.mp4": 9.36,
#     "Video 148.mp4": 8.69,
#     "Video 154.mp4": 8.51,
#     "experiment 1.wmv": 8.22,
#     "experiment 2.wmv": 6.35,
#     "experiment 3.mp4": 9.71,
#     "experiment 4.wmv": 12.05,
#     "v1.mp4": 9.62,
#     "video.mp4": 15.60,
# }}}}

NORMALIZE_VISUALIZATION = True
MAX_CRT90_10_FIT_SECONDS = DEFAULT_MAX_CRT90_10_FIT_SECONDS
CRT90_10_GAUSSIAN_SIGMA_SECONDS = DEFAULT_CRT90_10_GAUSSIAN_SIGMA_SECONDS
CRT90_10_BOOTSTRAP_COUNT = DEFAULT_CRT90_10_BOOTSTRAP_COUNT
CRT90_10_BOOTSTRAP_SEED = DEFAULT_CRT90_10_BOOTSTRAP_SEED


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

    yield runOnPath


# }}}


def cachePathForVideo(videoPath):
    # {{{
    return CACHE_DIR / f"{videoPath.stem}.npz"


# }}}


def loadCachedVideo(videoPath):
    # {{{
    cachePath = cachePathForVideo(videoPath)
    if not cachePath.exists():
        raise FileNotFoundError(f"No cache found for {videoPath.name}: {cachePath}")

    with np.load(cachePath) as data:
        cachedVideo = data["lFrames"]
        timesScdsArr = data["timesScdsArr"]
        avgAArr = data["avgAArr"]

    VIDEO_CACHE[videoPath] = (cachedVideo, timesScdsArr, avgAArr)
    return VIDEO_CACHE[videoPath]


# }}}


def releaseParamsForFilter(params, filterType):
    # {{{
    if filterType == "laplacian":
        return releaseParamsFromLaplacianParams(params)
    return params


# }}}


def metricLabel(filterType):
    # {{{
    return "Laplacian" if filterType == "laplacian" else "Canny"


# }}}


def measureCannyFrame(LFrame, params, show=False):
    # {{{
    if show:
        # Keep the debug view local to this script; the numeric sum uses the shared
        # path.
        from funcs import cannyEdgeDetector

        cannyFrame = cannyEdgeDetector(
            LFrame,
            params["thresh1"],
            params["thresh2"],
            params["blurKernel"],
            params["l2grad"],
        )
        combined = cv.hconcat([cannyFrame, LFrame])
        cv.imshow("Canny | L", combined)

    return cannySumFromLFrame(LFrame, params)


# }}}


def measureLaplacianFrame(LFrame, params, show=False):
    # {{{
    laplacianFrame = laplacianEdgeDetector(
        LFrame,
        params["ksize"],
        params["blurKernel"],
        params["scale"],
        params["delta"],
    )
    if show:
        combined = cv.hconcat(
            [cv.convertScaleAbs(255 * minMaxNormalizeOrZeros(laplacianFrame)), LFrame]
        )
        # combined = cv.hconcat([minMaxNormalize(laplacianFrame), LFrame])
        cv.imshow("Laplacian | L", combined)

    return laplacianSumFromLFrame(LFrame, params)


# }}}


def measureMetricsFrame(LFrame, params, filterType=FILTER_TYPE, show=False):
    # {{{
    if filterType == "laplacian":
        return measureLaplacianFrame(LFrame, params, show=show)
    if filterType == "canny":
        return measureCannyFrame(LFrame, params, show=show)
    raise ValueError(f"Unsupported FILTER_TYPE: {filterType}")


# }}}


def selectFrames(videoPath, params, filterType=FILTER_TYPE, show=False):
    # {{{
    assert isinstance(videoPath, Path)
    metricList = []

    cachedVideo, timesScdsArr, avgAArr = loadCachedVideo(videoPath)
    for frame in cachedVideo:
        metricSum = measureMetricsFrame(frame, params, filterType=filterType, show=show)
        metricList.append(metricSum)

        if show:
            key = cv.waitKey(10)

            if key == ord("q"):
                break

    metricArr = np.array(metricList)
    metricArr = normalizeCannyArr(metricArr)
    releaseParams = releaseParamsForFilter(params, filterType)

    try:
        releaseIndex = findReleaseIndex(metricArr, timesScdsArr, releaseParams)
        crtIntervalStartIndex, startIndex, crtIntervalEndIndex = (
            optimizableFindCrtIntervalIndices(
                avgAArr,
                timesScdsArr,
                releaseIndex,
                releaseParams["strictMinGrad"],
                releaseParams["strictMaxGrad"],
                releaseParams["relaxMinGrad"],
                releaseParams["relaxMaxGrad"],
            )
        )
    except Exception as err:
        print(f"{videoPath.stem}: release detection failed: {err}")
        plotVisualization(
            videoPath,
            metricArr,
            params,
            filterType=filterType,
            indices=None,
            error=err,
        )
        return None

    indices = (
        crtIntervalStartIndex,
        startIndex,
        crtIntervalEndIndex,
    )

    # if show:
    plotVisualization(
        videoPath, metricArr, params, filterType=filterType, indices=indices
    )

    return indices


# }}}


def plotVisualization(
    videoPath,
    metricArr,
    params,
    filterType=FILTER_TYPE,
    indices=None,
    error=None,
):
    # {{{
    cachedVideo, timesScdsArr, avgAArr = VIDEO_CACHE[videoPath]

    fig = plt.figure(
        figsize=(15, 8), num=f"{videoPath.name} - {metricLabel(filterType)}"
    )
    gs = fig.add_gridspec(3, 3)
    ax1 = fig.add_subplot(gs[0, :2])
    axCRT = fig.add_subplot(gs[1, 0])
    axCRT90_10 = fig.add_subplot(gs[1, 1])
    ax2 = fig.add_subplot(gs[2, :2])

    releaseParams = releaseParamsForFilter(params, filterType)
    try:
        releaseIndex = findReleaseIndex(metricArr, timesScdsArr, releaseParams)
    except Exception as err:
        print(f"{videoPath.stem}: releaseIndex calculation failed: {err}")
        releaseIndex = None

    if indices is not None:
        crtIntervalStartIndex, startIndex, crtIntervalEndIndex = indices
        crtIntervalStartLocalIndex = crtIntervalStartIndex
        startLocalIndex = startIndex
        crtIntervalEndLocalIndex = crtIntervalEndIndex

    if releaseIndex is not None:
        frameIndices = [
            np.clip(releaseIndex - 1 + i, 0, len(cachedVideo) - 1) for i in range(3)
        ]
    elif indices is None:
        frameIndices = np.linspace(0, len(cachedVideo) - 1, 3, dtype=int)
    else:
        frameIndices = [
            np.clip(crtIntervalStartLocalIndex - 1 + i, 0, len(cachedVideo) - 1)
            for i in range(3)
        ]

    for i in range(3):
        ax = fig.add_subplot(gs[i, 2])
        frameLocalIndex = frameIndices[i]
        releaseRoi = cachedVideo[frameLocalIndex]
        ax.set_title(f"Frame {frameLocalIndex}, {timesScdsArr[frameLocalIndex]:.3f} s")
        if NORMALIZE_VISUALIZATION:
            ax.imshow(releaseRoi, cmap="gray", aspect="equal")
        else:
            ax.imshow(releaseRoi, cmap="gray", vmin=0, vmax=255, aspect="equal")
        ax.set_axis_off()

    smoothedMetric = smoothCannyArr(metricArr, releaseParams)
    smoothedMetricTimes = timesScdsArr[: len(smoothedMetric)]

    ax2.plot(
        smoothedMetricTimes,
        smoothedMetric,
        color="blue",
        label=metricLabel(filterType),
    )
    try:
        smoothedMetricGrad, _, smoothedMetricGradTimes, _ = (
            calcGradientWithFilteredSamples(
                smoothedMetric,
                smoothedMetricTimes,
                smoothingWindowSeconds=0,
            )
        )
        smoothedMetricGrad = smoothCannyGradientArr(
            smoothedMetricGrad,
            releaseParams,
        )
    except Exception as err:
        print(f"{videoPath.stem}: {metricLabel(filterType)} gradient failed: {err}")
        ax2Grad = None
    else:
        ax2Grad = ax2.twinx()
        ax2Grad.plot(
            smoothedMetricGradTimes,
            smoothedMetricGrad,
            color="tab:gray",
            lw=0.8,
            alpha=0.75,
            label=f"grad {metricLabel(filterType)}",
        )
        ax2Grad.legend(loc="upper right")
    if releaseIndex is not None:
        estimatedTime = timesScdsArr[releaseIndex]
        ax2.axvline(
            estimatedTime,
            color="black",
            ls="--",
            label=f"release: {estimatedTime:.2f}s, {releaseIndex}",
        )
    targetIndex = TARGETS.get(videoPath.name)
    if targetIndex is not None and targetIndex < len(timesScdsArr):
        targetTime = timesScdsArr[targetIndex]
        ax2.axvline(
            targetTime,
            color="green",
            ls="-",
            label=f"target: {targetTime:.2f}s, {targetIndex}",
        )
    ax2.legend(loc="lower left")

    ax1.plot(timesScdsArr, avgAArr, lw=1, label="avg A")
    if error is not None:
        ax1.set_title(f"Release detection failed: {error}")
        axCRT.set_title("CRT detail unavailable")
        axCRT90_10.set_title("CRT90_10 unavailable")

    if indices is None:
        plt.tight_layout()
        plt.show()
        return

    try:
        avgADiagnostics = calcSignalGradientDiagnostics(
            avgAArr,
            timesScdsArr,
            crtIntervalStartLocalIndex,
            crtIntervalEndLocalIndex,
            releaseParams,
        )
    except Exception as err:
        print(f"{videoPath.stem}: A-gradient diagnostics failed: {err}")
        avgADiagnostics = None
    else:
        ax1.axvspan(
            timesScdsArr[crtIntervalStartLocalIndex],
            timesScdsArr[crtIntervalEndLocalIndex],
            color="tab:gray",
            alpha=0.12,
            label="A low-gradient interval",
        )
        avgARange = avgAArr.max() - avgAArr.min()
        avgAIntervalSmoothRawScale = (
            avgADiagnostics["signalInterval"] * avgARange + avgAArr.min()
        )
        ax1.plot(
            avgADiagnostics["times"],
            avgAIntervalSmoothRawScale,
            color="tab:orange",
            lw=1.4,
            label="smoothed A slice",
        )
        ax1Grad = ax1.twinx()
        ax1Grad.plot(
            avgADiagnostics["fullTimes"],
            avgADiagnostics["fullSignalGrad"],
            color="tab:gray",
            lw=0.8,
            alpha=0.75,
            label="full grad norm A",
        )
        ax1Grad.axhline(
            avgADiagnostics["relaxMinGrad"],
            color="tab:red",
            lw=0.8,
            ls=":",
            label=f"relaxMinGrad={avgADiagnostics['relaxMinGrad']:.3f}",
        )
        ax1Grad.axhline(
            avgADiagnostics["relaxMaxGrad"],
            color="tab:red",
            lw=0.8,
            ls=":",
            label=f"relaxMaxGrad={avgADiagnostics['relaxMaxGrad']:.3f}",
        )
        ax1Grad.axhline(
            avgADiagnostics["strictMinGrad"],
            color="tab:orange",
            lw=0.8,
            ls=":",
            label=f"strictMinGrad={avgADiagnostics['strictMinGrad']:.3f}",
        )
        ax1Grad.axhline(
            avgADiagnostics["strictMaxGrad"],
            color="tab:orange",
            lw=0.8,
            ls=":",
            label=f"strictMaxGrad={avgADiagnostics['strictMaxGrad']:.3f}",
        )
        # ax1Grad.legend(loc="upper right")

    ax1.axvline(
        timesScdsArr[crtIntervalStartLocalIndex],
        color="black",
        ls="--",
        label=f"CRT interval start {crtIntervalStartLocalIndex}",
    )
    ax1.axvline(
        timesScdsArr[startLocalIndex],
        color="green",
        ls="-",
        label=f"start {startLocalIndex}",
    )
    ax1.axvline(
        timesScdsArr[crtIntervalEndLocalIndex],
        color="black",
        ls="--",
        label=f"CRT interval end {crtIntervalEndLocalIndex}",
    )
    ax1.legend(loc="lower left")

    crtArrRaw = avgAArr[crtIntervalStartLocalIndex:crtIntervalEndLocalIndex]
    crtTimes = timesScdsArr[crtIntervalStartLocalIndex:crtIntervalEndLocalIndex]

    axCRT.scatter(crtTimes, crtArrRaw, s=5, marker=".", label="raw A slice")
    axCRT.set_title("Raw A + pCRT")
    axCRT.axvline(timesScdsArr[startLocalIndex], color="green", ls="-")
    axCRT90_10.set_title("CRT90_10")

    localIndices = (
        crtIntervalStartLocalIndex,
        startLocalIndex,
        crtIntervalEndLocalIndex,
    )
    try:
        crt90_10 = calcCRT90_10Gaussian(
            timesScdsArr,
            avgAArr,
            localIndices,
            maxFitSeconds=MAX_CRT90_10_FIT_SECONDS,
            gaussianSigmaSeconds=CRT90_10_GAUSSIAN_SIGMA_SECONDS,
        )
    except Exception as err:
        print(f"{videoPath.stem}: CRT 90-10 calculation failed: {err}")
        crt90_10 = None
    else:
        try:
            crt90_10Uncertainty = calcCRT90_10BootstrapUncertainty(
                timesScdsArr,
                avgAArr,
                localIndices,
                maxFitSeconds=MAX_CRT90_10_FIT_SECONDS,
                gaussianSigmaSeconds=CRT90_10_GAUSSIAN_SIGMA_SECONDS,
                nBootstraps=CRT90_10_BOOTSTRAP_COUNT,
                randomSeed=CRT90_10_BOOTSTRAP_SEED,
            )
        except Exception as err:
            print(f"{videoPath.stem}: CRT 90-10 uncertainty failed: {err}")
            crt90_10Uncertainty = None

        axCRT90_10.plot(
            crt90_10["times"],
            crt90_10["values"],
            color="tab:orange",
            lw=2,
            label=(
                f"Gaussian-smoothed A (sigma={CRT90_10_GAUSSIAN_SIGMA_SECONDS:.3f}s)"
            ),
        )
        axCRT90_10.axvline(
            crt90_10["startTime"],
            color="tab:green",
            ls=":",
            label=f"smooth max: {crt90_10['startTime']:.3f}s, {crt90_10['startIndex']}",
        )
        axCRT90_10.axvline(
            crt90_10["time90"],
            color="tab:blue",
            ls="--",
            label=f"90%: {crt90_10['time90']:.3f}s",
        )
        if crt90_10Uncertainty is None:
            decayLabel = f"{crt90_10['crt90_10']:.3f}"
        else:
            ci95Low, ci95High = crt90_10Uncertainty["crt90_10_ci95"]
            ci95HalfWidth = (ci95High - ci95Low) / 2
            decayLabel = (
                f"{crt90_10Uncertainty['crt90_10_mean']:.3f}+-"
                f"{ci95HalfWidth:.3f}"
            )
        axCRT90_10.axvline(
            crt90_10["time10"],
            color="tab:red",
            ls="--",
            label=f"10%: {crt90_10['time10']:.3f}s, decay={decayLabel}",
        )

    pcrtFit = calcPCRTFit(timesScdsArr, avgAArr, localIndices)
    if pcrtFit is not None:
        axCRT.plot(
            pcrtFit["fitTimes"],
            pcrtFit["fitValues"],
            color="tab:purple",
            lw=1.4,
            label=(
                f"pCRT exp, pCRT={pcrtFit['pcrt']:.3f} +/- "
                f"{pcrtFit['pcrtUncertainty']:.3f}s"
            ),
        )
        axCRT.axvline(
            pcrtFit["criticalTime"],
            color="tab:purple",
            ls=":",
            label=f"criticalTime={pcrtFit['criticalTime']:.3f}s",
        )
    axCRT.legend(loc="lower left")
    axCRT90_10Handles, _ = axCRT90_10.get_legend_handles_labels()
    if axCRT90_10Handles:
        axCRT90_10.legend(loc="lower left")
    if avgADiagnostics is not None:
        avgAGrad = avgADiagnostics["signalGrad"]
        print(f"{videoPath.stem} {avgAGrad.min():.3f}, {avgAGrad.max():.3f}")
    if crt90_10 is not None:
        print(f"{videoPath.stem} crtDecay90_10={crt90_10['crt90_10']:.3f} s")
    if pcrtFit is not None:
        print(
            f"{videoPath.stem} pCRT={pcrtFit['pcrt']:.3f} +/- "
            f"{pcrtFit['pcrtUncertainty']:.3f} s, "
            f"criticalTime={pcrtFit['criticalTime']:.3f} s"
        )
    print()

    plt.tight_layout()
    plt.show()


# }}}


def main():
    for testVideo in iterRunOnVideoPaths(RUN_ON):
        # if testVideo.name in TENTATIVE_TIMESTAMPS:
        #     continue
        try:
            loadVideoRoi(testVideo, ROIS_PATH)
        except (KeyError, ValueError) as err:
            print(f"Skipping {testVideo.name}: {err}")
            continue

        if not cachePathForVideo(testVideo).exists():
            print(
                f"Skipping {testVideo.name}: "
                f"missing cache {cachePathForVideo(testVideo)}"
            )
            continue

        indices = selectFrames(
            testVideo,
            params,
            filterType=FILTER_TYPE,
            show=True,
        )
        if indices is None:
            continue

        crtIntervalStartIndex, startIndex, crtIntervalEndIndex = indices
        targetIndex = TARGETS.get(testVideo.name)
        if targetIndex is None:
            print(
                f"{testVideo.stem} ({metricLabel(FILTER_TYPE)}): "
                f"crtIntervalStart={crtIntervalStartIndex}, "
                f"start={startIndex}, crtIntervalEnd={crtIntervalEndIndex}"
            )
        else:
            frameError = crtIntervalStartIndex - targetIndex
            print(
                f"{testVideo.stem} ({metricLabel(FILTER_TYPE)}): "
                f"crtIntervalStart={crtIntervalStartIndex}, target={targetIndex}, "
                f"frameError={frameError:+d}, start={startIndex}, "
                f"crtIntervalEnd={crtIntervalEndIndex}"
            )


if __name__ == "__main__":
    main()
