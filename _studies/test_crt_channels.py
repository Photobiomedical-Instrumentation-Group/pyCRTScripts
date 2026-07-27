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
    calcSignalGradientDiagnostics,
)
from matplotlib import pyplot as plt
from release_frame_processing import (
    cannySumsFromLFrames,
    findReleaseIndex,
    laplacianSumsFromLFrames,
    minMaxNormalizeOrZeros,
    optimizableFindCrtIntervalIndices,
    releaseParamsFromLaplacianParams,
)

plt.style.use("bmh")

VIDEOS_DIR_LIST = [
    Path("TrainingVideos"),
]
RUN_ON = VIDEOS_DIR_LIST
FILTER_TYPE = "canny"
# FILTER_TYPE = "laplacian"
VIDEO_EXTENSIONS = [".MOV", ".wmv", ".mp4"]
CACHE_DIR = Path("Npz/Cache")
MAX_CRT90_10_FIT_SECONDS = DEFAULT_MAX_CRT90_10_FIT_SECONDS
CRT90_10_GAUSSIAN_SIGMA_SECONDS = DEFAULT_CRT90_10_GAUSSIAN_SIGMA_SECONDS
CRT90_10_BOOTSTRAP_COUNT = DEFAULT_CRT90_10_BOOTSTRAP_COUNT
CRT90_10_BOOTSTRAP_SEED = DEFAULT_CRT90_10_BOOTSTRAP_SEED
CHANNEL_COLORS = {
    "BGR G": "tab:green",
    "LAB A": "tab:blue",
}
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
FILTER_PARAMS = CANNY_PARAMS if FILTER_TYPE == "canny" else LAPLACIAN_PARAMS


def iterVideoPathsInDirectory(videoDir):
    for filePath in sorted(
        Path(videoDir).iterdir(), key=lambda path: path.name.lower()
    ):
        if filePath.suffix in VIDEO_EXTENSIONS:
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


def loadCachedVideo(videoPath):
    cachePath = cachePathForVideo(videoPath)
    if not cachePath.exists():
        raise FileNotFoundError(f"No cache found for {videoPath.name}: {cachePath}")

    with np.load(cachePath) as data:
        missingKeys = {"lFrames", "avgAArr", "avgGArr", "timesScdsArr"} - set(
            data.files
        )
        if missingKeys:
            raise KeyError(
                f"Cache {cachePath} is missing key(s): {', '.join(sorted(missingKeys))}"
            )

        return {
            "lFrames": data["lFrames"],
            "avgAArr": data["avgAArr"].astype(float),
            "avgGArr": data["avgGArr"].astype(float),
            "timesScdsArr": data["timesScdsArr"].astype(float),
        }


def releaseParamsForFilter(params, filterType):
    if filterType == "laplacian":
        return releaseParamsFromLaplacianParams(params)
    if filterType == "canny":
        return params
    raise ValueError(f"Unsupported FILTER_TYPE: {filterType}")


def metricArrFromLFrames(lFrames, params, filterType):
    if filterType == "laplacian":
        return laplacianSumsFromLFrames(lFrames, params)
    if filterType == "canny":
        return cannySumsFromLFrames(lFrames, params)
    raise ValueError(f"Unsupported FILTER_TYPE: {filterType}")


def evaluateSignal(signalArr, timeArr, metricArr, releaseParams):
    releaseIndex = findReleaseIndex(metricArr, timeArr, releaseParams)
    crtIndices = optimizableFindCrtIntervalIndices(
        signalArr,
        timeArr,
        releaseIndex,
        releaseParams["strictMinGrad"],
        releaseParams["strictMaxGrad"],
        releaseParams["relaxMinGrad"],
        releaseParams["relaxMaxGrad"],
    )
    crt90_10 = calcCRT90_10Gaussian(
        timeArr,
        signalArr,
        crtIndices,
        maxFitSeconds=MAX_CRT90_10_FIT_SECONDS,
        gaussianSigmaSeconds=CRT90_10_GAUSSIAN_SIGMA_SECONDS,
    )
    try:
        crt90_10Uncertainty = calcCRT90_10BootstrapUncertainty(
            timeArr,
            signalArr,
            crtIndices,
            maxFitSeconds=MAX_CRT90_10_FIT_SECONDS,
            gaussianSigmaSeconds=CRT90_10_GAUSSIAN_SIGMA_SECONDS,
            nBootstraps=CRT90_10_BOOTSTRAP_COUNT,
            randomSeed=CRT90_10_BOOTSTRAP_SEED,
        )
    except Exception as err:
        print(f"CRT90_10 uncertainty failed: {type(err).__name__}: {err}")
        crt90_10Uncertainty = None

    pcrtFit = calcPCRTFit(timeArr, signalArr, crtIndices)
    diagnostics = calcSignalGradientDiagnostics(
        signalArr,
        timeArr,
        crtIndices[0],
        crtIndices[2],
        releaseParams,
    )
    return crtIndices, crt90_10, crt90_10Uncertainty, pcrtFit, diagnostics


def crt90_10Label(crt90_10, uncertainty):
    if uncertainty is None:
        return f"CRT90_10={crt90_10['crt90_10']:.3f}s"

    ci95Low, ci95High = uncertainty["crt90_10_ci95"]
    ci95HalfWidth = (ci95High - ci95Low) / 2
    return (
        f"CRT90_10={uncertainty['crt90_10_mean']:.3f} +/- "
        f"{ci95HalfWidth:.3f}s"
    )


def plotSignalResult(
    contextAx,
    detailAx,
    channelLabel,
    contextTitle,
    signalArr,
    timeArr,
    releaseIndex,
    crtIndices,
    crt90_10,
    crt90_10Uncertainty,
    pcrtFit,
    diagnostics,
):
    crtIntervalStartIndex, _, crtIntervalEndIndex = crtIndices
    rawSignalColor = CHANNEL_COLORS.get(channelLabel, "tab:blue")
    signalNorm = minMaxNormalizeOrZeros(signalArr)
    intervalTimes = diagnostics["times"]
    signalGrad = diagnostics["signalGrad"]
    strictMinGrad = diagnostics["strictMinGrad"]
    strictMaxGrad = diagnostics["strictMaxGrad"]
    relaxMinGrad = diagnostics["relaxMinGrad"]
    relaxMaxGrad = diagnostics["relaxMaxGrad"]
    intervalSignalNorm = minMaxNormalizeOrZeros(
        signalArr[crtIntervalStartIndex:crtIntervalEndIndex]
    )

    contextAx.plot(
        timeArr,
        signalNorm,
        color=rawSignalColor,
        lw=1,
    )
    contextAx.axvspan(
        timeArr[crtIntervalStartIndex],
        timeArr[crtIntervalEndIndex],
        color="tab:gray",
        alpha=0.12,
    )
    contextAx.axvline(
        timeArr[releaseIndex],
        color="black",
        ls="--",
    )
    contextAx.axvline(
        timeArr[crtIntervalStartIndex],
        color="black",
        ls=":",
    )
    contextAx.axvline(
        timeArr[crtIntervalEndIndex],
        color="black",
        ls="-.",
    )
    contextAx.axvline(
        crt90_10["time90"],
        color="tab:green",
        ls="--",
        label=f"90%={crt90_10['time90']:.3f}s",
    )
    contextAx.axvline(
        crt90_10["time10"],
        color="tab:red",
        ls="--",
        label=f"10%={crt90_10['time10']:.3f}s",
    )
    if pcrtFit is not None:
        contextAx.axvline(
            pcrtFit["criticalTime"],
            color="tab:purple",
            ls=":",
            label=f"critical={pcrtFit['criticalTime']:.3f}s",
        )

    gradAx = contextAx.twinx()
    gradAx.plot(
        intervalTimes,
        signalGrad,
        color="tab:gray",
        lw=0.8,
        alpha=0.75,
    )
    gradAx.axhline(relaxMinGrad, color="tab:red", lw=0.8, ls=":")
    gradAx.axhline(relaxMaxGrad, color="tab:red", lw=0.8, ls=":")
    gradAx.axhline(strictMinGrad, color="tab:orange", lw=0.8, ls=":")
    gradAx.axhline(strictMaxGrad, color="tab:orange", lw=0.8, ls=":")

    contextAx.set_title(contextTitle)
    contextAx.grid(True)
    contextAx.legend(loc="lower left", fontsize="small")

    detailAx.plot(
        intervalTimes,
        intervalSignalNorm,
        color=rawSignalColor,
        lw=1,
    )
    detailAx.plot(
        crt90_10["times"],
        crt90_10["values"],
        color="tab:orange",
        lw=1.4,
        label=crt90_10Label(crt90_10, crt90_10Uncertainty),
    )
    if pcrtFit is not None:
        referenceValues = signalArr[crtIntervalStartIndex:crtIntervalEndIndex]
        referenceMin = referenceValues.min()
        referenceRange = referenceValues.max() - referenceMin
        fitValues = (pcrtFit["fitValues"] - referenceMin) / referenceRange
        detailAx.plot(
            pcrtFit["fitTimes"],
            fitValues,
            color="tab:purple",
            lw=1.4,
            label=(f"pCRT={pcrtFit['pcrt']:.3f} +/- {pcrtFit['pcrtUncertainty']:.3f}s"),
        )
        detailAx.axvline(
            pcrtFit["criticalTime"],
            color="tab:purple",
            ls=":",
            label=f"critical={pcrtFit['criticalTime']:.3f}s",
        )
    detailAx.axvline(
        crt90_10["startTime"],
        color="tab:green",
        ls=":",
    )
    detailAx.axvline(
        crt90_10["time90"],
        color="tab:green",
        ls="--",
        label=f"90%={crt90_10['time90']:.3f}s",
    )
    detailAx.axvline(
        crt90_10["time10"],
        color="tab:red",
        ls="--",
        label=f"10%={crt90_10['time10']:.3f}s",
    )
    detailAx.grid(True)
    detailAx.legend(loc="lower left", fontsize="small")


def processVideo(videoPath):
    cached = loadCachedVideo(videoPath)
    timeArr = cached["timesScdsArr"]
    releaseParams = releaseParamsForFilter(FILTER_PARAMS, FILTER_TYPE)
    metricArr = metricArrFromLFrames(
        cached["lFrames"],
        FILTER_PARAMS,
        FILTER_TYPE,
    )
    releaseIndex = findReleaseIndex(metricArr, timeArr, releaseParams)

    channelResults = {
        "BGR G": (
            cached["avgGArr"],
            evaluateSignal(cached["avgGArr"], timeArr, metricArr, releaseParams),
            "BGR G full array",
        ),
        "LAB A": (
            cached["avgAArr"],
            evaluateSignal(cached["avgAArr"], timeArr, metricArr, releaseParams),
            "LAB A full array",
        ),
    }

    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(12, 6),
        num=f"CRT channels - {videoPath.name}",
    )

    for columnIndex, channelLabel in enumerate(("BGR G", "LAB A")):
        signalArr, result, contextTitle = channelResults[channelLabel]
        crtIndices, crt90_10, crt90_10Uncertainty, pcrtFit, diagnostics = result
        plotSignalResult(
            axes[0, columnIndex],
            axes[1, columnIndex],
            channelLabel,
            contextTitle,
            signalArr,
            timeArr,
            releaseIndex,
            crtIndices,
            crt90_10,
            crt90_10Uncertainty,
            pcrtFit,
            diagnostics,
        )

    fig.suptitle(videoPath.name)
    fig.supxlabel("Time (s)")
    fig.text(0.005, 0.5, "Normalized Intensities", va="center", rotation="vertical")
    fig.text(0.975, 0.5, "Gradient", va="center", rotation=270)
    fig.tight_layout(rect=(0.025, 0.035, 0.975, 0.955))
    plt.show()


def main():
    for videoPath in iterRunOnVideoPaths(RUN_ON):
        try:
            processVideo(videoPath)
        except Exception as err:
            print(f"Skipping {videoPath}: {type(err).__name__}: {err}")


if __name__ == "__main__":
    main()
