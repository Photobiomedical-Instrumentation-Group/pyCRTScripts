import argparse
import json
import multiprocessing as mp
import os
import queue
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import tomllib

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import optuna
from release_frame_processing import optimizableFindCrtIntervalIndices

SCRIPT_DIR = Path(__file__).resolve().parent
CACHE_DIR = SCRIPT_DIR / "Npz/Cache"
RELEASE_TARGETS_PATH = SCRIPT_DIR / "target_frames.toml"
INTERVAL_TARGETS_PATH = SCRIPT_DIR / "crt_interval_targets.toml"
OUTPUT_PATH = SCRIPT_DIR / "optimized_params_crt_interval.toml"

RAQUEL_MASTERS_DATASET = "raquelMasters"
OTHER_DATASET = "other"
VIDEO_EXTENSIONS = [".MOV", ".wmv", ".mp4"]
VIDEO_EXTENSIONS_LOWER = {suffix.lower() for suffix in VIDEO_EXTENSIONS}

# Top-level knobs for quick experimentation.
N_REPEATS = 50
N_TRIALS_PER_REPEAT = 200
VALIDATION_VIDEO_RATIO = 50 / 191
OPTIMIZATION_VIDEO_RATIO = 66 / 191
OPTUNA_TIMEOUT = None
TRIAL_TIMEOUT_SECONDS = 5 * 60
OPTUNA_SEED = 1009
SHOW_PROGRESS_BAR = True
VALIDATION_METRIC = "endpoint_rmse_s"

STRICT_MIN_GRAD_MIN = -2.0
STRICT_MIN_GRAD_MAX = 0.5
STRICT_MAX_GRAD_MIN = -0.5
STRICT_MAX_GRAD_MAX = 2.0
RELAX_MIN_GRAD_MIN = -10.0
RELAX_MIN_GRAD_MAX = 0.5
RELAX_MAX_GRAD_MIN = -0.5
RELAX_MAX_GRAD_MAX = 10.0
OFFSET_TIME_MIN = -0.5
OFFSET_TIME_MAX = 0.5
MIN_GRAD_SPAN = 1e-6

DEFAULT_PARAMS = {
    "strictMinGrad": -0.5,
    "strictMaxGrad": 1.0,
    "relaxMinGrad": -3.0,
    "relaxMaxGrad": 1.0,
    "offsetTime": 0.0,
}

LARGE_PENALTY = 1_000_000.0


@dataclass(frozen=True)
class TargetVideo:
    videoPath: Path
    releaseIndex: int
    crtIntervalStartIndex: int
    crtIntervalEndIndex: int
    datasetName: str


@dataclass(frozen=True)
class CachedVideo:
    videoPath: Path
    releaseIndex: int
    targetStartIndex: int
    targetEndIndex: int
    targetStartTimeScds: float
    targetEndTimeScds: float
    intensArr: np.ndarray
    timeArr: np.ndarray
    datasetName: str


def loadReleaseIndices():
    with RELEASE_TARGETS_PATH.open("rb") as file:
        targetConfig = tomllib.load(file)

    return {
        str(videoName): int(releaseIndex)
        for videoName, releaseIndex in targetConfig.get("targets", {}).items()
    }


def loadIntervalTargets():
    with INTERVAL_TARGETS_PATH.open("rb") as file:
        targetConfig = tomllib.load(file)

    intervalTargets = {}
    for videoName, target in targetConfig.get("targets", {}).items():
        intervalTargets[str(videoName)] = {
            "crtIntervalStartIndex": int(target["crtIntervalStartIndex"]),
            "crtIntervalEndIndex": int(target["crtIntervalEndIndex"]),
        }
    return intervalTargets


def cachePathForVideo(videoPath):
    return CACHE_DIR / f"{videoPath.stem}.npz"


def relativeToScript(path):
    path = Path(path)
    try:
        return path.relative_to(SCRIPT_DIR)
    except ValueError:
        return path


def datasetNameForVideo(videoPath):
    if videoPath.suffix.lower() == ".wmv" and videoPath.name.startswith("P"):
        return RAQUEL_MASTERS_DATASET
    return OTHER_DATASET


def validateTargetVideo(videoPath, releaseIndex, intervalTarget):
    cachePath = cachePathForVideo(videoPath)
    if not cachePath.exists():
        return f"cache not found at {relativeToScript(cachePath)}"

    try:
        with np.load(cachePath) as data:
            if "avgAArr" not in data or "timesScdsArr" not in data:
                return "cache missing avgAArr or timesScdsArr"
            frameCount = len(data["timesScdsArr"])
            if len(data["avgAArr"]) != frameCount:
                return (
                    "cache avgAArr and timesScdsArr length mismatch: "
                    f"{len(data['avgAArr'])} != {frameCount}"
                )
    except Exception as err:
        return f"cache not usable: {type(err).__name__}: {err}"

    targetStartIndex = intervalTarget["crtIntervalStartIndex"]
    targetEndIndex = intervalTarget["crtIntervalEndIndex"]

    if releaseIndex < 0 or releaseIndex >= frameCount:
        return f"release frame {releaseIndex} outside cache range [0, {frameCount - 1}]"
    if targetStartIndex < 0 or targetStartIndex >= frameCount:
        return (
            f"target start frame {targetStartIndex} outside cache range "
            f"[0, {frameCount - 1}]"
        )
    if targetEndIndex < 0 or targetEndIndex >= frameCount:
        return (
            f"target end frame {targetEndIndex} outside cache range "
            f"[0, {frameCount - 1}]"
        )
    if targetEndIndex <= targetStartIndex:
        return (
            f"target end frame {targetEndIndex} must be after "
            f"target start frame {targetStartIndex}"
        )

    return None


def discoverTargetVideos(verbose=True):
    releaseIndices = loadReleaseIndices()
    intervalTargets = loadIntervalTargets()
    targetVideos = []
    skipped = []
    cacheStems = (
        {cachePath.stem for cachePath in CACHE_DIR.glob("*.npz")}
        if CACHE_DIR.is_dir()
        else set()
    )

    for videoName in sorted(intervalTargets, key=str.lower):
        videoPath = Path(videoName)
        if videoPath.suffix.lower() not in VIDEO_EXTENSIONS_LOWER:
            skipped.append((videoName, "", "unsupported video extension"))
            continue

        datasetName = datasetNameForVideo(videoPath)
        if videoName not in releaseIndices:
            skipped.append((videoName, datasetName, "release target not found"))
            continue
        if videoPath.stem not in cacheStems:
            skipped.append((videoName, datasetName, "cache not found"))
            continue

        releaseIndex = releaseIndices[videoName]
        intervalTarget = intervalTargets[videoName]
        skipReason = validateTargetVideo(videoPath, releaseIndex, intervalTarget)
        if skipReason is not None:
            skipped.append((videoName, datasetName, skipReason))
            continue

        targetVideos.append(
            TargetVideo(
                videoPath=videoPath,
                releaseIndex=releaseIndex,
                crtIntervalStartIndex=intervalTarget["crtIntervalStartIndex"],
                crtIntervalEndIndex=intervalTarget["crtIntervalEndIndex"],
                datasetName=datasetName,
            )
        )

    targetVideos = sorted(
        targetVideos,
        key=lambda video: (video.datasetName.lower(), video.videoPath.name.lower()),
    )

    if verbose:
        print(
            f"usable_targets={len(targetVideos)} skipped_targets={len(skipped)} "
            f"datasets={json.dumps(countByDataset(targetVideos), sort_keys=True)}",
            flush=True,
        )
        for videoName, datasetName, reason in skipped[:20]:
            datasetLabel = f"{datasetName}: " if datasetName else ""
            print(f"skipped_target {datasetLabel}{videoName}: {reason}", flush=True)
        if len(skipped) > 20:
            print(f"skipped_target ... {len(skipped) - 20} more", flush=True)

    return targetVideos


def filterTargetVideos(targetVideos, videoNames):
    if not videoNames:
        return list(targetVideos)

    requested = set(videoNames)
    filtered = [
        targetVideo
        for targetVideo in targetVideos
        if targetVideo.videoPath.name in requested
        or targetVideo.videoPath.stem in requested
    ]
    matched = {video.videoPath.name for video in filtered} | {
        video.videoPath.stem for video in filtered
    }
    missing = requested - matched
    if missing:
        raise ValueError(f"Unknown target video(s): {', '.join(sorted(missing))}")
    return filtered


def countByDataset(targetVideos):
    counts = defaultdict(int)
    for targetVideo in targetVideos:
        counts[targetVideo.datasetName] += 1
    return dict(sorted(counts.items()))


def loadCachedVideo(targetVideo, verbose=False):
    cachePath = cachePathForVideo(targetVideo.videoPath)
    if not cachePath.exists():
        raise FileNotFoundError(
            f"No cache found for {targetVideo.videoPath.name}: "
            f"{relativeToScript(cachePath)}"
        )

    with np.load(cachePath) as data:
        intensArr = data["avgAArr"].astype(float)
        timeArr = data["timesScdsArr"].astype(float)

    if len(intensArr) != len(timeArr):
        raise ValueError(
            f"Cached avgAArr and timesScdsArr length mismatch for "
            f"{targetVideo.videoPath.name}: {len(intensArr)} != {len(timeArr)}."
        )
    if len(timeArr) < 3:
        raise ValueError(f"Not enough cached frames for {targetVideo.videoPath.name}.")

    releaseIndex = int(targetVideo.releaseIndex)
    targetStartIndex = int(targetVideo.crtIntervalStartIndex)
    targetEndIndex = int(targetVideo.crtIntervalEndIndex)
    for label, index in (
        ("release", releaseIndex),
        ("target start", targetStartIndex),
        ("target end", targetEndIndex),
    ):
        if index < 0 or index >= len(timeArr):
            raise ValueError(
                f"{label} frame {index} for {targetVideo.videoPath.name} is outside "
                f"the cached range [0, {len(timeArr) - 1}]."
            )

    cached = CachedVideo(
        videoPath=targetVideo.videoPath,
        releaseIndex=releaseIndex,
        targetStartIndex=targetStartIndex,
        targetEndIndex=targetEndIndex,
        targetStartTimeScds=float(timeArr[targetStartIndex]),
        targetEndTimeScds=float(timeArr[targetEndIndex]),
        intensArr=intensArr,
        timeArr=timeArr,
        datasetName=targetVideo.datasetName,
    )
    if verbose:
        print(
            f"loaded cache {relativeToScript(cachePath)}: frames={len(timeArr)}",
            flush=True,
        )
    return cached


def buildCache(targetVideos, verbose=False):
    return [
        loadCachedVideo(
            targetVideo,
            verbose=verbose,
        )
        for targetVideo in targetVideos
    ]


def paramsAreValid(params):
    return (
        params["strictMinGrad"] < params["strictMaxGrad"]
        and params["relaxMinGrad"] < params["relaxMaxGrad"]
        and params["relaxMinGrad"] <= params["strictMinGrad"]
        and params["relaxMaxGrad"] >= params["strictMaxGrad"]
    )


def selectIntervalCached(cachedVideo, params):
    crtIntervalStartIndex, _, crtIntervalEndIndex = optimizableFindCrtIntervalIndices(
        cachedVideo.intensArr,
        cachedVideo.timeArr,
        cachedVideo.releaseIndex,
        params["strictMinGrad"],
        params["strictMaxGrad"],
        params["relaxMinGrad"],
        params["relaxMaxGrad"],
        offsetTime=params["offsetTime"],
    )
    return int(crtIntervalStartIndex), int(crtIntervalEndIndex)


def failureResult(videoName, datasetName, targetVideo, err):
    if targetVideo is None:
        return {
            "video": videoName,
            "dataset": datasetName,
            "release": None,
            "target_start": None,
            "target_start_time_s": None,
            "predicted_start": None,
            "predicted_start_time_s": None,
            "start_frame_error": None,
            "start_time_error_s": None,
            "target_end": None,
            "target_end_time_s": None,
            "predicted_end": None,
            "predicted_end_time_s": None,
            "end_frame_error": None,
            "end_time_error_s": None,
            "duration_time_error_s": None,
            "exception": f"{type(err).__name__}: {err}",
        }

    return {
        "video": videoName,
        "dataset": datasetName,
        "release": targetVideo.releaseIndex,
        "target_start": targetVideo.targetStartIndex,
        "target_start_time_s": targetVideo.targetStartTimeScds,
        "predicted_start": None,
        "predicted_start_time_s": None,
        "start_frame_error": None,
        "start_time_error_s": None,
        "target_end": targetVideo.targetEndIndex,
        "target_end_time_s": targetVideo.targetEndTimeScds,
        "predicted_end": None,
        "predicted_end_time_s": None,
        "end_frame_error": None,
        "end_time_error_s": None,
        "duration_time_error_s": None,
        "exception": f"{type(err).__name__}: {err}",
    }


def evaluateCachedVideo(cachedVideo, params):
    try:
        predictedStartIndex, predictedEndIndex = selectIntervalCached(
            cachedVideo,
            params,
        )
        predictedStartTimeScds = float(cachedVideo.timeArr[predictedStartIndex])
        predictedEndTimeScds = float(cachedVideo.timeArr[predictedEndIndex])
        startFrameError = int(predictedStartIndex) - cachedVideo.targetStartIndex
        endFrameError = int(predictedEndIndex) - cachedVideo.targetEndIndex
        startTimeErrorScds = predictedStartTimeScds - cachedVideo.targetStartTimeScds
        endTimeErrorScds = predictedEndTimeScds - cachedVideo.targetEndTimeScds
        targetDurationScds = (
            cachedVideo.targetEndTimeScds - cachedVideo.targetStartTimeScds
        )
        predictedDurationScds = predictedEndTimeScds - predictedStartTimeScds
        durationTimeErrorScds = predictedDurationScds - targetDurationScds
        return {
            "video": cachedVideo.videoPath.name,
            "dataset": cachedVideo.datasetName,
            "release": cachedVideo.releaseIndex,
            "target_start": cachedVideo.targetStartIndex,
            "target_start_time_s": cachedVideo.targetStartTimeScds,
            "predicted_start": int(predictedStartIndex),
            "predicted_start_time_s": predictedStartTimeScds,
            "start_frame_error": int(startFrameError),
            "start_time_error_s": startTimeErrorScds,
            "target_end": cachedVideo.targetEndIndex,
            "target_end_time_s": cachedVideo.targetEndTimeScds,
            "predicted_end": int(predictedEndIndex),
            "predicted_end_time_s": predictedEndTimeScds,
            "end_frame_error": int(endFrameError),
            "end_time_error_s": endTimeErrorScds,
            "duration_time_error_s": durationTimeErrorScds,
        }
    except Exception as err:
        return failureResult(
            cachedVideo.videoPath.name,
            cachedVideo.datasetName,
            cachedVideo,
            err,
        )


def metricSummary(values):
    values = np.array(values, dtype=float)
    if len(values) == 0:
        raise ValueError("Cannot calculate metrics for an empty value set.")
    absoluteValues = np.abs(values)
    return {
        "mae": float(np.mean(absoluteValues)),
        "rmse": float(np.sqrt(np.mean(values**2))),
        "max_abs": float(np.max(absoluteValues)),
    }


def metricsFromResults(params, results):
    startTimeErrors = []
    endTimeErrors = []
    endpointTimeErrors = []
    durationTimeErrors = []
    startFrameErrors = []
    endFrameErrors = []
    endpointFrameErrors = []

    for result in results:
        if result["predicted_start"] is None or result["predicted_end"] is None:
            startTimeErrors.append(LARGE_PENALTY)
            endTimeErrors.append(LARGE_PENALTY)
            endpointTimeErrors.extend([LARGE_PENALTY, LARGE_PENALTY])
            durationTimeErrors.append(LARGE_PENALTY)
            startFrameErrors.append(LARGE_PENALTY)
            endFrameErrors.append(LARGE_PENALTY)
            endpointFrameErrors.extend([LARGE_PENALTY, LARGE_PENALTY])
            continue

        startTimeError = result["start_time_error_s"]
        endTimeError = result["end_time_error_s"]
        durationTimeError = result["duration_time_error_s"]
        startFrameError = result["start_frame_error"]
        endFrameError = result["end_frame_error"]

        startTimeErrors.append(startTimeError)
        endTimeErrors.append(endTimeError)
        endpointTimeErrors.extend([startTimeError, endTimeError])
        durationTimeErrors.append(durationTimeError)
        startFrameErrors.append(startFrameError)
        endFrameErrors.append(endFrameError)
        endpointFrameErrors.extend([startFrameError, endFrameError])

    if not results:
        raise ValueError("Cannot calculate metrics for an empty result set.")

    startTime = metricSummary(startTimeErrors)
    endTime = metricSummary(endTimeErrors)
    endpointTime = metricSummary(endpointTimeErrors)
    durationTime = metricSummary(durationTimeErrors)
    startFrame = metricSummary(startFrameErrors)
    endFrame = metricSummary(endFrameErrors)
    endpointFrame = metricSummary(endpointFrameErrors)
    failureCount = sum(
        result["predicted_start"] is None or result["predicted_end"] is None
        for result in results
    )

    return {
        "params": params,
        "video_count": len(results),
        "failure_count": int(failureCount),
        "success_rate": float((len(results) - failureCount) / len(results)),
        "start_mae_s": startTime["mae"],
        "start_rmse_s": startTime["rmse"],
        "start_max_abs_error_s": startTime["max_abs"],
        "end_mae_s": endTime["mae"],
        "end_rmse_s": endTime["rmse"],
        "end_max_abs_error_s": endTime["max_abs"],
        "endpoint_mae_s": endpointTime["mae"],
        "endpoint_rmse_s": endpointTime["rmse"],
        "endpoint_max_abs_error_s": endpointTime["max_abs"],
        "duration_mae_s": durationTime["mae"],
        "duration_rmse_s": durationTime["rmse"],
        "duration_max_abs_error_s": durationTime["max_abs"],
        "start_mae_frames": startFrame["mae"],
        "start_rmse_frames": startFrame["rmse"],
        "start_max_abs_error_frames": startFrame["max_abs"],
        "end_mae_frames": endFrame["mae"],
        "end_rmse_frames": endFrame["rmse"],
        "end_max_abs_error_frames": endFrame["max_abs"],
        "endpoint_mae_frames": endpointFrame["mae"],
        "endpoint_rmse_frames": endpointFrame["rmse"],
        "endpoint_max_abs_error_frames": endpointFrame["max_abs"],
        "results": results,
    }


def failedTrialMetrics(params, videoCount, exception):
    failure = {
        "video": "<trial>",
        "dataset": "",
        "release": None,
        "target_start": None,
        "target_start_time_s": None,
        "predicted_start": None,
        "predicted_start_time_s": None,
        "start_frame_error": None,
        "start_time_error_s": None,
        "target_end": None,
        "target_end_time_s": None,
        "predicted_end": None,
        "predicted_end_time_s": None,
        "end_frame_error": None,
        "end_time_error_s": None,
        "duration_time_error_s": None,
        "exception": exception,
    }
    return metricsFromResults(params, [failure] * int(videoCount))


def evaluateParams(params, cache, verbose=False):
    if not paramsAreValid(params):
        return failedTrialMetrics(
            params,
            len(cache),
            "InvalidParametersError: gradient bounds do not satisfy "
            "relaxMinGrad <= strictMinGrad < strictMaxGrad <= relaxMaxGrad",
        )

    results = [evaluateCachedVideo(cachedVideo, params) for cachedVideo in cache]
    if verbose:
        for result in results:
            print(json.dumps(result), flush=True)
    return metricsFromResults(params, results)


def evaluateTargetVideos(params, targetVideos, verbose=False):
    results = []
    for targetVideo in targetVideos:
        try:
            cachedVideo = loadCachedVideo(targetVideo, verbose=False)
            result = evaluateCachedVideo(cachedVideo, params)
        except Exception as err:
            result = failureResult(
                targetVideo.videoPath.name,
                targetVideo.datasetName,
                None,
                err,
            )
        results.append(result)
        if verbose:
            print(json.dumps(result), flush=True)

    return metricsFromResults(params, results)


def suggestParams(trial):
    return {
        "strictMinGrad": trial.suggest_float(
            "strictMinGrad",
            STRICT_MIN_GRAD_MIN,
            STRICT_MIN_GRAD_MAX,
        ),
        "strictMaxGrad": trial.suggest_float(
            "strictMaxGrad",
            STRICT_MAX_GRAD_MIN,
            STRICT_MAX_GRAD_MAX,
        ),
        "relaxMinGrad": trial.suggest_float(
            "relaxMinGrad",
            RELAX_MIN_GRAD_MIN,
            RELAX_MIN_GRAD_MAX,
        ),
        "relaxMaxGrad": trial.suggest_float(
            "relaxMaxGrad",
            RELAX_MAX_GRAD_MIN,
            RELAX_MAX_GRAD_MAX,
        ),
        "offsetTime": trial.suggest_float(
            "offsetTime",
            OFFSET_TIME_MIN,
            OFFSET_TIME_MAX,
        ),
    }


def randomUniform(rng, low, high):
    return float(rng.uniform(low, high))


def randomParams(rng):
    strictMinGrad = randomUniform(rng, STRICT_MIN_GRAD_MIN, STRICT_MIN_GRAD_MAX)
    strictMaxLow = max(STRICT_MAX_GRAD_MIN, strictMinGrad + MIN_GRAD_SPAN)
    strictMaxGrad = randomUniform(rng, strictMaxLow, STRICT_MAX_GRAD_MAX)
    relaxMinGrad = randomUniform(rng, RELAX_MIN_GRAD_MIN, strictMinGrad)
    relaxMaxLow = max(RELAX_MAX_GRAD_MIN, strictMaxGrad)
    relaxMaxGrad = randomUniform(rng, relaxMaxLow, RELAX_MAX_GRAD_MAX)
    return {
        "strictMinGrad": strictMinGrad,
        "strictMaxGrad": strictMaxGrad,
        "relaxMinGrad": relaxMinGrad,
        "relaxMaxGrad": relaxMaxGrad,
        "offsetTime": randomUniform(rng, OFFSET_TIME_MIN, OFFSET_TIME_MAX),
    }


def evaluateParamsWorker(cache, requestQueue, responseQueue):
    while True:
        payload = requestQueue.get()
        if payload is None:
            return

        trialNumber, params, verbose = payload
        try:
            metrics = evaluateParams(params, cache, verbose=verbose)
        except BaseException as err:
            responseQueue.put(
                (
                    trialNumber,
                    False,
                    f"{type(err).__name__}: {err}",
                )
            )
        else:
            responseQueue.put((trialNumber, True, metrics))


class TimedTrialEvaluator:
    def __init__(self, cache, timeoutSeconds):
        self.cache = cache
        self.timeoutSeconds = timeoutSeconds
        self.context = mp.get_context("spawn")
        self.requestQueue = None
        self.responseQueue = None
        self.process = None
        self.startWorker()

    def startWorker(self):
        self.requestQueue = self.context.Queue()
        self.responseQueue = self.context.Queue()
        self.process = self.context.Process(
            target=evaluateParamsWorker,
            args=(self.cache, self.requestQueue, self.responseQueue),
        )
        self.process.daemon = True
        self.process.start()

    def closeQueues(self):
        for queueObj in (self.requestQueue, self.responseQueue):
            if queueObj is None:
                continue
            queueObj.close()
            queueObj.join_thread()
        self.requestQueue = None
        self.responseQueue = None

    def close(self):
        if self.process is not None and self.process.is_alive():
            try:
                self.requestQueue.put(None)
            except Exception:
                pass
            self.process.join(timeout=2)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=2)
        self.process = None
        self.closeQueues()

    def restartWorker(self):
        if self.process is not None and self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=2)
        self.process = None
        self.closeQueues()
        self.startWorker()

    def evaluate(self, params, trialNumber, verbose=False):
        if self.process is None or not self.process.is_alive():
            self.restartWorker()

        self.requestQueue.put((trialNumber, params, verbose))
        deadline = time.monotonic() + self.timeoutSeconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                print(
                    f"trial={trialNumber} timed out after "
                    f"{self.timeoutSeconds:.0f} seconds; assigning penalty",
                    flush=True,
                )
                self.restartWorker()
                return failedTrialMetrics(
                    params,
                    len(self.cache),
                    f"TimeoutError: trial exceeded {self.timeoutSeconds:.0f} seconds",
                )

            try:
                responseTrialNumber, ok, payload = self.responseQueue.get(
                    timeout=min(1.0, remaining)
                )
            except queue.Empty:
                if self.process is not None and not self.process.is_alive():
                    exitCode = self.process.exitcode
                    self.restartWorker()
                    return failedTrialMetrics(
                        params,
                        len(self.cache),
                        f"WorkerExitError: trial worker exited with code {exitCode}",
                    )
                continue

            if responseTrialNumber != trialNumber:
                return failedTrialMetrics(
                    params,
                    len(self.cache),
                    f"WorkerProtocolError: expected trial {trialNumber}, "
                    f"got {responseTrialNumber}",
                )
            if not ok:
                return failedTrialMetrics(params, len(self.cache), payload)
            return payload


def makeObjective(cache, repeatIndex, verbose=False):
    bestValue = np.inf
    evaluator = TimedTrialEvaluator(cache, TRIAL_TIMEOUT_SECONDS)

    def objective(trial):
        nonlocal bestValue
        params = suggestParams(trial)
        metrics = evaluator.evaluate(params, trial.number, verbose=verbose)
        value = metrics[VALIDATION_METRIC]

        trial.set_user_attr("params", params)
        trial.set_user_attr("start_mae_s", metrics["start_mae_s"])
        trial.set_user_attr("start_rmse_s", metrics["start_rmse_s"])
        trial.set_user_attr("end_mae_s", metrics["end_mae_s"])
        trial.set_user_attr("end_rmse_s", metrics["end_rmse_s"])
        trial.set_user_attr("endpoint_mae_s", metrics["endpoint_mae_s"])
        trial.set_user_attr(
            "endpoint_max_abs_error_s", metrics["endpoint_max_abs_error_s"]
        )
        trial.set_user_attr("endpoint_mae_frames", metrics["endpoint_mae_frames"])
        trial.set_user_attr(
            "endpoint_max_abs_error_frames",
            metrics["endpoint_max_abs_error_frames"],
        )
        trial.set_user_attr("failure_count", metrics["failure_count"])
        trial.set_user_attr("success_rate", metrics["success_rate"])
        trial.set_user_attr("results", metrics["results"])

        if value < bestValue:
            bestValue = value
            print(
                f"repeat={repeatIndex} "
                f"new_best_{VALIDATION_METRIC}={value:.6f} "
                f"endpoint_mae_s={metrics['endpoint_mae_s']:.6f} "
                f"start_rmse_s={metrics['start_rmse_s']:.6f} "
                f"end_rmse_s={metrics['end_rmse_s']:.6f} "
                f"endpoint_max_abs_error_s={metrics['endpoint_max_abs_error_s']:.6f} "
                f"endpoint_rmse_frames={metrics['endpoint_rmse_frames']:.3f} "
                f"trial={trial.number} "
                f"params={json.dumps(params)}",
                flush=True,
            )

        return value

    objective.close = evaluator.close
    return objective


def validateVideoRatio(name, ratio):
    if not 0 < ratio < 1:
        raise ValueError(f"{name} must be between 0 and 1. Got {ratio}.")


def videoCountFromRatio(videoCount, ratio):
    if videoCount <= 0:
        return 0
    return max(1, int(round(videoCount * ratio)))


def splitVideoCounts(targetVideos):
    totalCount = len(targetVideos)
    validationCount = videoCountFromRatio(totalCount, VALIDATION_VIDEO_RATIO)
    optimizationCount = videoCountFromRatio(totalCount, OPTIMIZATION_VIDEO_RATIO)
    return validationCount, optimizationCount


def selectRandomSubset(candidates, count, rng):
    candidates = list(candidates)
    if count > len(candidates):
        raise ValueError(
            f"Cannot select {count} videos from {len(candidates)} candidates."
        )

    shuffled = list(candidates)
    rng.shuffle(shuffled)
    selected = shuffled[:count]
    rng.shuffle(selected)
    return selected


def validateSplitFeasibility(targetVideos):
    validateVideoRatio("VALIDATION_VIDEO_RATIO", VALIDATION_VIDEO_RATIO)
    validateVideoRatio("OPTIMIZATION_VIDEO_RATIO", OPTIMIZATION_VIDEO_RATIO)
    validationCount, optimizationCount = splitVideoCounts(targetVideos)
    requiredCount = validationCount + optimizationCount
    if len(targetVideos) < requiredCount:
        raise ValueError(
            f"Requested split is infeasible: ratios select {requiredCount} usable "
            f"target videos ({validationCount} validation, {optimizationCount} "
            f"optimization), but only {len(targetVideos)} were found."
        )


def selectValidationSet(targetVideos, rng):
    validationCount, _ = splitVideoCounts(targetVideos)
    return selectRandomSubset(targetVideos, validationCount, rng)


def selectOptimizationSet(targetVideos, validationSet, rng):
    validationNames = {targetVideo.videoPath.name for targetVideo in validationSet}
    candidates = [
        targetVideo
        for targetVideo in targetVideos
        if targetVideo.videoPath.name not in validationNames
    ]
    _, optimizationCount = splitVideoCounts(targetVideos)
    return selectRandomSubset(candidates, optimizationCount, rng)


def runRepeat(
    repeatIndex,
    targetVideos,
    validationSet,
    validationCache,
    rng,
    args,
):
    optimizationSet = selectOptimizationSet(
        targetVideos,
        validationSet,
        rng,
    )
    optimizationCache = buildCache(optimizationSet, verbose=False)
    initialParams = randomParams(rng)
    samplerSeed = int(rng.integers(0, np.iinfo(np.uint32).max))

    print(
        f"repeat={repeatIndex}/{args.repeats} "
        f"optimization_videos={len(optimizationSet)} "
        f"optimization_datasets={json.dumps(countByDataset(optimizationSet), sort_keys=True)} "
        f"initial_params={json.dumps(initialParams)}",
        flush=True,
    )

    sampler = optuna.samplers.TPESampler(seed=samplerSeed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.enqueue_trial(initialParams)
    objective = makeObjective(optimizationCache, repeatIndex, verbose=args.verbose)
    try:
        study.optimize(
            objective,
            n_trials=args.n_trials,
            timeout=args.timeout,
            show_progress_bar=not args.no_progress_bar and SHOW_PROGRESS_BAR,
        )
    finally:
        objective.close()

    completedTrials = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    if not completedTrials:
        raise RuntimeError(f"Repeat {repeatIndex} completed no trials.")

    bestTrial = None
    bestParams = None
    validationMetrics = None
    bestValidationValue = np.inf
    for candidateTrial in completedTrials:
        candidateParams = dict(
            candidateTrial.user_attrs.get("params", candidateTrial.params)
        )
        candidateValidationMetrics = evaluateParams(
            candidateParams,
            validationCache,
            verbose=False,
        )
        candidateValidationValue = candidateValidationMetrics[VALIDATION_METRIC]
        if candidateValidationValue < bestValidationValue:
            bestValidationValue = candidateValidationValue
            bestTrial = candidateTrial
            bestParams = candidateParams
            validationMetrics = candidateValidationMetrics

    optimizationMetrics = evaluateParams(bestParams, optimizationCache, verbose=False)
    validationValue = validationMetrics[VALIDATION_METRIC]

    print(
        f"repeat={repeatIndex} "
        f"best_trial={bestTrial.number} "
        f"best_trial_selection=validation "
        f"optimization_{VALIDATION_METRIC}={optimizationMetrics[VALIDATION_METRIC]:.6f} "
        f"validation_{VALIDATION_METRIC}={validationValue:.6f} "
        f"validation_endpoint_mae_s={validationMetrics['endpoint_mae_s']:.6f} "
        f"validation_success_rate={validationMetrics['success_rate']:.6f} "
        f"params={json.dumps(bestParams)}",
        flush=True,
    )

    return {
        "repeat": repeatIndex,
        "study": study,
        "best_trial": int(bestTrial.number),
        "optimization_set": optimizationSet,
        "initial_params": initialParams,
        "params": bestParams,
        "optimization_metrics": optimizationMetrics,
        "validation_metrics": validationMetrics,
    }


def evaluateAllDatasets(params, targetVideos, verbose=False):
    allMetrics = evaluateTargetVideos(params, targetVideos, verbose=verbose)
    perDataset = {}
    grouped = defaultdict(list)
    for targetVideo in targetVideos:
        grouped[targetVideo.datasetName].append(targetVideo)

    for datasetName in sorted(grouped):
        perDataset[datasetName] = evaluateTargetVideos(
            params,
            grouped[datasetName],
            verbose=verbose,
        )

    return {"all": allMetrics, "datasets": perDataset}


def printMetrics(metrics):
    print("params:")
    for key, value in metrics["params"].items():
        print(f"  {key}: {value}")
    print(
        f"videos={metrics['video_count']}, "
        f"failures={metrics['failure_count']}, "
        f"success_rate={metrics['success_rate']:.6f}"
    )
    print(
        f"start_mae={metrics['start_mae_s']:.6f} s, "
        f"start_rmse={metrics['start_rmse_s']:.6f} s, "
        f"start_max_abs_error={metrics['start_max_abs_error_s']:.6f} s"
    )
    print(
        f"end_mae={metrics['end_mae_s']:.6f} s, "
        f"end_rmse={metrics['end_rmse_s']:.6f} s, "
        f"end_max_abs_error={metrics['end_max_abs_error_s']:.6f} s"
    )
    print(
        f"endpoint_mae={metrics['endpoint_mae_s']:.6f} s, "
        f"endpoint_rmse={metrics['endpoint_rmse_s']:.6f} s, "
        f"endpoint_max_abs_error={metrics['endpoint_max_abs_error_s']:.6f} s"
    )
    print(
        f"duration_mae={metrics['duration_mae_s']:.6f} s, "
        f"duration_rmse={metrics['duration_rmse_s']:.6f} s, "
        f"duration_max_abs_error={metrics['duration_max_abs_error_s']:.6f} s"
    )
    print(
        f"endpoint_mae={metrics['endpoint_mae_frames']:.3f} frames, "
        f"endpoint_rmse={metrics['endpoint_rmse_frames']:.3f} frames, "
        f"endpoint_max_abs_error={metrics['endpoint_max_abs_error_frames']:.3f} frames"
    )
    print("per-video:")
    for result in metrics["results"]:
        if result["predicted_start"] is None or result["predicted_end"] is None:
            print(f"  {result['video']}: failed ({result['exception']})")
        else:
            print(
                f"  {result['video']}: "
                f"target=({result['target_start']}, {result['target_end']}), "
                f"predicted=({result['predicted_start']}, {result['predicted_end']}), "
                f"start_frame_error={result['start_frame_error']:+d}, "
                f"end_frame_error={result['end_frame_error']:+d}, "
                f"start_time_error={result['start_time_error_s']:+.6f} s, "
                f"end_time_error={result['end_time_error_s']:+.6f} s"
            )


def tomlValue(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if np.isfinite(value):
            return repr(float(value))
        return json.dumps(str(float(value)))
    if isinstance(value, Path):
        return json.dumps(str(value))
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(tomlValue(item) for item in value) + "]"
    raise TypeError(f"Cannot serialize {type(value).__name__} to TOML.")


def appendSection(lines, sectionName):
    if lines and lines[-1] != "":
        lines.append("")
    lines.append(f"[{sectionName}]")


def appendTableArray(lines, sectionName):
    if lines and lines[-1] != "":
        lines.append("")
    lines.append(f"[[{sectionName}]]")


def appendStringArray(lines, key, values):
    lines.append(f"{key} = [")
    for value in values:
        lines.append(f"  {tomlValue(value)},")
    lines.append("]")


def appendMetrics(lines, metrics):
    for key in (
        "video_count",
        "failure_count",
        "success_rate",
        "start_mae_s",
        "start_rmse_s",
        "start_max_abs_error_s",
        "end_mae_s",
        "end_rmse_s",
        "end_max_abs_error_s",
        "endpoint_mae_s",
        "endpoint_rmse_s",
        "endpoint_max_abs_error_s",
        "duration_mae_s",
        "duration_rmse_s",
        "duration_max_abs_error_s",
        "start_mae_frames",
        "start_rmse_frames",
        "start_max_abs_error_frames",
        "end_mae_frames",
        "end_rmse_frames",
        "end_max_abs_error_frames",
        "endpoint_mae_frames",
        "endpoint_rmse_frames",
        "endpoint_max_abs_error_frames",
    ):
        lines.append(f"{key} = {tomlValue(metrics[key])}")


def writeOptimizedParamsToml(
    outputPath,
    bestRecord,
    validationSet,
    finalPerformance,
    repeatRecords,
    seed,
):
    lines = []

    appendSection(lines, "run")
    lines.append(f"seed = {tomlValue(seed)}")
    lines.append(f"validation_metric = {tomlValue(VALIDATION_METRIC)}")
    lines.append(f"repeats = {tomlValue(len(repeatRecords))}")
    lines.append(f"trials_per_repeat = {tomlValue(N_TRIALS_PER_REPEAT)}")
    lines.append(f"validation_video_ratio = {tomlValue(VALIDATION_VIDEO_RATIO)}")
    lines.append(f"optimization_video_ratio = {tomlValue(OPTIMIZATION_VIDEO_RATIO)}")
    lines.append(f"cache_dir = {tomlValue(relativeToScript(CACHE_DIR))}")
    lines.append(
        f"release_targets_path = {tomlValue(relativeToScript(RELEASE_TARGETS_PATH))}"
    )
    lines.append(
        f"interval_targets_path = {tomlValue(relativeToScript(INTERVAL_TARGETS_PATH))}"
    )
    lines.append(f"trial_timeout_seconds = {tomlValue(TRIAL_TIMEOUT_SECONDS)}")

    appendSection(lines, "best")
    lines.append(f"repeat = {tomlValue(bestRecord['repeat'])}")
    lines.append(
        f"validation_{VALIDATION_METRIC} = "
        f"{tomlValue(bestRecord['validation_metrics'][VALIDATION_METRIC])}"
    )
    lines.append(
        f"all_{VALIDATION_METRIC} = "
        f"{tomlValue(finalPerformance['all'][VALIDATION_METRIC])}"
    )

    appendSection(lines, "params")
    for key, value in bestRecord["params"].items():
        lines.append(f"{key} = {tomlValue(value)}")

    appendSection(lines, "performance.validation")
    appendMetrics(lines, bestRecord["validation_metrics"])

    appendSection(lines, "performance.all")
    appendMetrics(lines, finalPerformance["all"])

    for datasetName, metrics in finalPerformance["datasets"].items():
        appendSection(lines, f"performance.datasets.{datasetName}")
        appendMetrics(lines, metrics)

    appendSection(lines, "selection.validation")
    appendStringArray(
        lines,
        "videos",
        [targetVideo.videoPath.name for targetVideo in validationSet],
    )

    appendSection(lines, "selection.best_optimization")
    appendStringArray(
        lines,
        "videos",
        [targetVideo.videoPath.name for targetVideo in bestRecord["optimization_set"]],
    )

    for record in repeatRecords:
        appendTableArray(lines, "repeats_summary")
        lines.append(f"repeat = {tomlValue(record['repeat'])}")
        lines.append(f"best_trial = {tomlValue(record['best_trial'])}")
        lines.append(
            f"optimization_{VALIDATION_METRIC} = "
            f"{tomlValue(record['optimization_metrics'][VALIDATION_METRIC])}"
        )
        lines.append(
            f"validation_{VALIDATION_METRIC} = "
            f"{tomlValue(record['validation_metrics'][VALIDATION_METRIC])}"
        )
        lines.append(
            f"all_{VALIDATION_METRIC} = "
            f"{tomlValue(record['all_metrics'][VALIDATION_METRIC])}"
        )
        lines.append(
            f"validation_success_rate = "
            f"{tomlValue(record['validation_metrics']['success_rate'])}"
        )

    outputPath.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parseArgs():
    parser = argparse.ArgumentParser(
        description=(
            "Optimize CRT interval detection gradient bounds and offset with "
            "repeated random optimization splits and one fixed validation split."
        )
    )
    parser.add_argument("--repeats", type=int, default=N_REPEATS)
    parser.add_argument("--n-trials", type=int, default=N_TRIALS_PER_REPEAT)
    parser.add_argument("--timeout", type=float, default=OPTUNA_TIMEOUT)
    parser.add_argument("--seed", type=int, default=OPTUNA_SEED)
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Evaluate default params against all discovered interval target videos.",
    )
    parser.add_argument(
        "--videos",
        nargs="*",
        help="Optional subset of target video stems or filenames.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-video results for every evaluation.",
    )
    parser.add_argument(
        "--no-progress-bar",
        action="store_true",
        help="Disable Optuna's progress bar.",
    )
    return parser.parse_args()


def main():
    mp.freeze_support()
    global N_REPEATS
    global N_TRIALS_PER_REPEAT

    args = parseArgs()
    N_REPEATS = args.repeats
    N_TRIALS_PER_REPEAT = args.n_trials

    if args.repeats < 1:
        raise ValueError("--repeats must be positive.")
    if args.n_trials < 1:
        raise ValueError("--n-trials must be positive.")

    targetVideos = filterTargetVideos(
        discoverTargetVideos(verbose=True),
        args.videos,
    )
    if not targetVideos:
        raise ValueError("No usable interval target videos were found.")

    if args.evaluate_only:
        metrics = evaluateTargetVideos(
            DEFAULT_PARAMS,
            targetVideos,
            verbose=args.verbose,
        )
        printMetrics(metrics)
        return

    validateSplitFeasibility(targetVideos)

    rng = np.random.default_rng(args.seed)
    validationSet = selectValidationSet(targetVideos, rng)
    validationCache = buildCache(validationSet, verbose=False)
    print(
        f"validation_videos={len(validationSet)} "
        f"validation_datasets={json.dumps(countByDataset(validationSet), sort_keys=True)}",
        flush=True,
    )

    bestRecord = None
    bestAllValue = np.inf
    repeatRecords = []
    for repeatIndex in range(1, args.repeats + 1):
        record = runRepeat(
            repeatIndex,
            targetVideos,
            validationSet,
            validationCache,
            rng,
            args,
        )
        record["all_metrics"] = evaluateTargetVideos(
            record["params"],
            targetVideos,
            verbose=False,
        )
        repeatRecords.append(record)
        allValue = record["all_metrics"][VALIDATION_METRIC]
        if allValue < bestAllValue:
            bestAllValue = allValue
            bestRecord = record
            print(
                f"new_best_repeat={repeatIndex} "
                f"all_{VALIDATION_METRIC}={allValue:.6f}",
                flush=True,
            )

    if bestRecord is None:
        raise RuntimeError("No repeat completed successfully.")

    finalPerformance = evaluateAllDatasets(
        bestRecord["params"],
        targetVideos,
        verbose=args.verbose,
    )

    writeOptimizedParamsToml(
        OUTPUT_PATH,
        bestRecord,
        validationSet,
        finalPerformance,
        repeatRecords,
        args.seed,
    )

    print(f"best_repeat={bestRecord['repeat']}")
    print("validation performance:")
    printMetrics(bestRecord["validation_metrics"])
    print("all-datasets performance:")
    printMetrics(finalPerformance["all"])
    for datasetName, metrics in finalPerformance["datasets"].items():
        print(f"{datasetName} performance:")
        printMetrics(metrics)
    print(f"wrote {relativeToScript(OUTPUT_PATH)}")


if __name__ == "__main__":
    main()
