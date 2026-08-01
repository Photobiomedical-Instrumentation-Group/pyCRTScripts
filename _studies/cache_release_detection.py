import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-pycrt")

import cv2 as cv
import numpy as np
import tomllib
from funcs import cannyEdgeDetector, laplacianEdgeDetector
from release_frame_processing import (
    cannySumsFromLFrames,
    findReleaseIndex,
    laplacianSumsFromLFrames,
    minMaxNormalizeOrZeros,
    releaseParamsFromLaplacianParams,
)

CACHE_DIR = Path("Npz/Cache")

# Set to "canny" or "laplacian".
FILTER_TYPE = "laplacian"

# Accepts a cache file, a directory of cache files, or a nested list of either.
RUN_ON = CACHE_DIR

OPTIMIZED_PARAMS_PATHS = {
    "canny": Path("optimized_params_canny.toml"),
    "laplacian": Path("optimized_params_laplacian.toml"),
}
OPTIMIZED_CRT_INTERVAL_PARAMS_PATH = Path("optimized_params_crt_interval.toml")
OPTIMIZED_PARAMS_CACHE = {}
OPTIMIZED_CRT_INTERVAL_PARAMS_CACHE = None

SHOW_LIVE_WINDOW = False
WAIT_KEY_MS = 1
OVERWRITE_EXISTING = False
VERBOSE = True


def loadOptimizedParamsFile(path):
    with path.open("rb") as file:
        config = tomllib.load(file)

    params = config.get("params")
    if not isinstance(params, dict) or not params:
        raise ValueError(f"No [params] table found in {path}.")
    return dict(params)


def optimizedParamsPathForFilter(filterType):
    path = OPTIMIZED_PARAMS_PATHS[filterType]
    if not path.exists():
        raise FileNotFoundError(f"Optimized {filterType} params TOML not found: {path}")
    return path


def loadOptimizedParams(filterType):
    if filterType in OPTIMIZED_PARAMS_CACHE:
        return dict(OPTIMIZED_PARAMS_CACHE[filterType])

    path = optimizedParamsPathForFilter(filterType)
    params = loadOptimizedParamsFile(path)
    OPTIMIZED_PARAMS_CACHE[filterType] = params
    if VERBOSE:
        print(f"loaded {filterType} params from {path}", flush=True)
    return dict(params)


def loadOptimizedCrtIntervalParams():
    global OPTIMIZED_CRT_INTERVAL_PARAMS_CACHE
    if OPTIMIZED_CRT_INTERVAL_PARAMS_CACHE is not None:
        return dict(OPTIMIZED_CRT_INTERVAL_PARAMS_CACHE)

    if not OPTIMIZED_CRT_INTERVAL_PARAMS_PATH.exists():
        raise FileNotFoundError(
            "Optimized CRT interval params TOML not found: "
            f"{OPTIMIZED_CRT_INTERVAL_PARAMS_PATH}"
        )
    params = loadOptimizedParamsFile(OPTIMIZED_CRT_INTERVAL_PARAMS_PATH)
    OPTIMIZED_CRT_INTERVAL_PARAMS_CACHE = params
    if VERBOSE:
        print(
            f"loaded CRT interval params from {OPTIMIZED_CRT_INTERVAL_PARAMS_PATH}",
            flush=True,
        )
    return dict(params)


def iterCachePaths(runOn):
    if isinstance(runOn, (list, tuple, set)):
        for item in runOn:
            yield from iterCachePaths(item)
        return

    runOnPath = Path(runOn)
    if runOnPath.is_dir():
        for cachePath in sorted(
            runOnPath.glob("*.npz"), key=lambda path: path.name.lower()
        ):
            yield cachePath
        return

    yield runOnPath


def paramsForFilter(filterType):
    if filterType == "canny":
        filterParams = loadOptimizedParams(filterType)
        return filterParams, filterParams
    if filterType == "laplacian":
        filterParams = loadOptimizedParams(filterType)
        return filterParams, releaseParamsFromLaplacianParams(filterParams)
    raise ValueError(f"Unsupported FILTER_TYPE: {filterType}")


def processedFrameFromLFrame(lFrame, filterType, filterParams):
    if filterType == "canny":
        return cannyEdgeDetector(
            lFrame,
            filterParams["thresh1"],
            filterParams["thresh2"],
            filterParams["blurKernel"],
            filterParams["l2grad"],
        )

    if filterType == "laplacian":
        laplacianFrame = laplacianEdgeDetector(
            lFrame,
            filterParams["ksize"],
            filterParams["blurKernel"],
            filterParams["scale"],
            filterParams["delta"],
        )
        return cv.convertScaleAbs(255 * minMaxNormalizeOrZeros(laplacianFrame))

    raise ValueError(f"Unsupported FILTER_TYPE: {filterType}")


def metricArrFromLFrames(
    lFrames, filterType, filterParams, show=False, windowName=None
):
    if not show:
        if filterType == "canny":
            return cannySumsFromLFrames(lFrames, filterParams)
        if filterType == "laplacian":
            return laplacianSumsFromLFrames(lFrames, filterParams)
        raise ValueError(f"Unsupported FILTER_TYPE: {filterType}")

    liveWindowOpen = True
    for frameIndex, lFrame in enumerate(lFrames):
        if not liveWindowOpen:
            break

        processedFrame = processedFrameFromLFrame(lFrame, filterType, filterParams)
        combined = cv.hconcat([lFrame, processedFrame])
        cv.imshow(windowName, combined)
        key = cv.waitKey(WAIT_KEY_MS) & 0xFF
        if key == ord("q"):
            liveWindowOpen = False
            cv.destroyWindow(windowName)

        if VERBOSE:
            print(f"processed frame {frameIndex}", flush=True)

    if liveWindowOpen:
        cv.destroyWindow(windowName)

    if filterType == "canny":
        return cannySumsFromLFrames(lFrames, filterParams)
    return laplacianSumsFromLFrames(lFrames, filterParams)


def releaseFieldNames(filterType):
    prefix = f"{filterType}Release"
    return {
        "index": f"{prefix}Index",
        "time": f"{prefix}TimeScds",
        "filter": f"{prefix}FilterType",
        "params": f"{prefix}ParamsJson",
    }


def cacheHasReleaseFields(cacheData, filterType):
    fieldNames = releaseFieldNames(filterType)
    return fieldNames["index"] in cacheData and fieldNames["time"] in cacheData


def loadCacheData(cachePath):
    with np.load(cachePath) as data:
        return {key: data[key] for key in data.files}


def saveCacheData(cachePath, cacheData):
    temporaryPath = cachePath.with_name(f"{cachePath.name}.tmp.npz")
    np.savez_compressed(temporaryPath, **cacheData)
    temporaryPath.replace(cachePath)


def updateCacheReleaseDetection(cachePath, filterType=FILTER_TYPE):
    filterParams, releaseParams = paramsForFilter(filterType)
    crtIntervalParams = loadOptimizedCrtIntervalParams()
    cacheData = loadCacheData(cachePath)

    if cacheHasReleaseFields(cacheData, filterType) and not OVERWRITE_EXISTING:
        print(f"skipping {cachePath.name}: {filterType} release fields already exist")
        return None

    missingKeys = {"lFrames", "timesScdsArr"} - set(cacheData)
    if missingKeys:
        raise KeyError(f"{cachePath.name} is missing keys: {sorted(missingKeys)}")

    lFrames = cacheData["lFrames"]
    timeArr = cacheData["timesScdsArr"].astype(float)
    if len(lFrames) != len(timeArr):
        raise ValueError(
            f"{cachePath.name}: lFrames and timesScdsArr lengths differ: "
            f"{len(lFrames)} != {len(timeArr)}."
        )

    windowName = f"{filterType}: {cachePath.stem}" if SHOW_LIVE_WINDOW else None
    metricArr = metricArrFromLFrames(
        lFrames,
        filterType,
        filterParams,
        show=SHOW_LIVE_WINDOW,
        windowName=windowName,
    )
    releaseIndex = findReleaseIndex(metricArr, timeArr, releaseParams)
    releaseTimeScds = float(timeArr[releaseIndex])

    fieldNames = releaseFieldNames(filterType)
    cacheData[fieldNames["index"]] = np.array(releaseIndex, dtype=np.int64)
    cacheData[fieldNames["time"]] = np.array(releaseTimeScds, dtype=np.float64)
    cacheData[fieldNames["filter"]] = np.array(filterType)
    cacheData[fieldNames["params"]] = np.array(json.dumps(filterParams, sort_keys=True))
    cacheData["crtIntervalParamsJson"] = np.array(
        json.dumps(crtIntervalParams, sort_keys=True)
    )

    saveCacheData(cachePath, cacheData)
    print(
        f"updated {cachePath.name}: {filterType}ReleaseIndex={releaseIndex}, "
        f"{filterType}ReleaseTimeScds={releaseTimeScds:.6f}",
        flush=True,
    )
    return releaseIndex, releaseTimeScds


def main():
    cachePaths = list(iterCachePaths(RUN_ON))
    if not cachePaths:
        raise ValueError(f"No cache files found in {RUN_ON}.")

    failed = []
    for cachePath in cachePaths:
        if not cachePath.exists():
            print(f"skipping {cachePath}: file not found")
            failed.append((cachePath, "file not found"))
            continue

        try:
            updateCacheReleaseDetection(cachePath)
        except Exception as err:
            reason = f"{type(err).__name__}: {err}"
            print(f"failed {cachePath.name}: {reason}", flush=True)
            failed.append((cachePath, reason))

    if failed:
        print("\nFailed cache files:")
        for cachePath, reason in failed:
            print(f"  {cachePath}: {reason}")

    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
