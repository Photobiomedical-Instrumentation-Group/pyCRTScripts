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
from release_frame_processing import (
    findReleaseIndex,
    laplacianSumsFromLFrames,
    releaseParamsFromLaplacianParams,
)

ROIS_PATH = "rois_full.toml"
CACHE_DIR = Path("Npz/Cache")
TARGETS_PATH = Path("target_frames.toml")
OUTPUT_PATH = Path("optimized_params_laplacian.toml")

RAQUEL_MASTERS_DATASET = "raquelMasters"
OTHER_DATASET = "other"
VIDEO_EXTENSIONS = [".MOV", ".wmv", ".mp4"]
VIDEO_EXTENSIONS_LOWER = {suffix.lower() for suffix in VIDEO_EXTENSIONS}

# Top-level knobs for quick experimentation.
N_RESTARTS = 20
N_TRIALS_PER_RESTART = 100
VALIDATION_VIDEO_RATIO = 0.2
OPTIMIZATION_VIDEO_RATIO = 0.8
OPTUNA_TIMEOUT = None
TRIAL_TIMEOUT_SECONDS = 5 * 60
OPTUNA_SEED = 988
SHOW_PROGRESS_BAR = True
VALIDATION_METRIC = "weighted_rmse_s"

KSIZE_MIN = 0
KSIZE_MAX = 3
BLUR_KERNEL_MIN = 0
BLUR_KERNEL_MAX = 8
LAPLACIAN_PLATEAU_MIN = 0.01
LAPLACIAN_PLATEAU_MAX = 2.0
LAPLACIAN_SMOOTHING_KERNEL_CHOICES = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
LAPLACIAN_GRADIENT_SMOOTHING_KERNEL_CHOICES = [1, 3, 5]
INDEX_OFFSET_MIN = -5
INDEX_OFFSET_MAX = 5

FIXED_LAPLACIAN_PARAMS = {
    "scale": 1,
    "delta": 0,
}

DEFAULT_PARAMS = {
    "ksize": 3,
    "blurKernel": 7,
    **FIXED_LAPLACIAN_PARAMS,
    "laplacianPlateau": 2.0,
    "laplacianSmoothingKernel": 19,
    "laplacianGradientSmoothingKernel": 3,
    "indexOffset": 0,
}

LARGE_PENALTY = 1_000_000.0
UNDERESTIMATE_LOSS_MULTIPLIER = 2.0


@dataclass(frozen=True)
class TargetVideo:
    videoPath: Path
    targetIndex: int
    datasetName: str


@dataclass(frozen=True)
class CachedVideo:
    videoPath: Path
    targetIndex: int
    targetTimeScds: float
    lowerAnalysisBound: int
    lFrames: tuple[np.ndarray, ...]
    timeArr: np.ndarray
    datasetName: str


def loadTargetFrameIndices():
    with TARGETS_PATH.open("rb") as file:
        targetConfig = tomllib.load(file)

    return {
        str(videoName): int(targetIndex)
        for videoName, targetIndex in targetConfig.get("targets", {}).items()
    }


def cachePathForVideo(videoPath):
    return CACHE_DIR / f"{videoPath.stem}.npz"


def datasetNameForVideo(videoPath):
    if videoPath.suffix.lower() == ".wmv" and videoPath.name.startswith("P"):
        return RAQUEL_MASTERS_DATASET
    return OTHER_DATASET


def validateTargetVideo(videoPath, targetIndex):
    cachePath = cachePathForVideo(videoPath)
    if not cachePath.exists():
        return f"cache not found at {cachePath}"

    try:
        with np.load(cachePath) as data:
            if "lFrames" not in data or "timesScdsArr" not in data:
                return "cache missing lFrames or timesScdsArr"
            frameCount = len(data["timesScdsArr"])
    except Exception as err:
        return f"cache not usable: {type(err).__name__}: {err}"

    if targetIndex < 0 or targetIndex >= frameCount:
        return f"target frame {targetIndex} outside cache range [0, {frameCount - 1}]"

    return None


def discoverTargetVideos(verbose=True):
    targetFrameIndices = loadTargetFrameIndices()
    targetVideos = []
    skipped = []
    cacheStems = (
        {cachePath.stem for cachePath in CACHE_DIR.glob("*.npz")}
        if CACHE_DIR.is_dir()
        else set()
    )

    for videoName in sorted(targetFrameIndices, key=str.lower):
        videoPath = Path(videoName)
        if videoPath.suffix.lower() not in VIDEO_EXTENSIONS_LOWER:
            skipped.append((videoName, "", "unsupported video extension"))
            continue

        datasetName = datasetNameForVideo(videoPath)
        if videoPath.stem not in cacheStems:
            skipped.append((videoName, datasetName, "cache not found"))
            continue

        targetIndex = targetFrameIndices[videoName]
        skipReason = validateTargetVideo(videoPath, targetIndex)
        if skipReason is not None:
            skipped.append((videoName, datasetName, skipReason))
            continue

        targetVideos.append(
            TargetVideo(
                videoPath=videoPath,
                targetIndex=targetIndex,
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
            f"No cache found for {targetVideo.videoPath.name}: {cachePath}"
        )

    with np.load(cachePath) as data:
        lFrames = data["lFrames"]
        timeArr = data["timesScdsArr"].astype(float)

    if len(lFrames) < 3:
        raise ValueError(f"Not enough cached frames for {targetVideo.videoPath.name}.")

    targetIndex = int(targetVideo.targetIndex)
    if targetIndex < 0 or targetIndex >= len(timeArr):
        raise ValueError(
            f"Target frame {targetIndex} for {targetVideo.videoPath.name} is outside "
            f"the cached range [0, {len(timeArr) - 1}]."
        )

    cached = CachedVideo(
        videoPath=targetVideo.videoPath,
        targetIndex=targetIndex,
        targetTimeScds=float(timeArr[targetIndex]),
        lowerAnalysisBound=0,
        lFrames=tuple(lFrames),
        timeArr=timeArr,
        datasetName=targetVideo.datasetName,
    )
    if verbose:
        print(
            f"loaded cache {cachePath}: frames={len(lFrames)}",
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


def selectFrameCached(cachedVideo, params):
    laplacianArr = laplacianSumsFromLFrames(cachedVideo.lFrames, params)
    localReleaseIndex = findReleaseIndex(
        laplacianArr,
        cachedVideo.timeArr,
        releaseParamsFromLaplacianParams(params),
    )
    return localReleaseIndex + cachedVideo.lowerAnalysisBound


def weightedTimeError(timeErrorScds):
    multiplier = UNDERESTIMATE_LOSS_MULTIPLIER if timeErrorScds < 0 else 1.0
    return multiplier * timeErrorScds


def failureResult(videoName, datasetName, targetIndex, targetTimeScds, err):
    return {
        "video": videoName,
        "dataset": datasetName,
        "target": targetIndex,
        "target_time_s": targetTimeScds,
        "predicted": None,
        "predicted_time_s": None,
        "frame_error": None,
        "time_error_s": None,
        "weighted_time_error_s": None,
        "exception": f"{type(err).__name__}: {err}",
    }


def evaluateCachedVideo(cachedVideo, params):
    try:
        predictedIndex = selectFrameCached(cachedVideo, params)
        predictedLocalIndex = int(predictedIndex) - cachedVideo.lowerAnalysisBound
        predictedTimeScds = float(cachedVideo.timeArr[predictedLocalIndex])
        frameError = int(predictedIndex) - cachedVideo.targetIndex
        timeErrorScds = predictedTimeScds - cachedVideo.targetTimeScds
        weightedErrorScds = weightedTimeError(timeErrorScds)
        return {
            "video": cachedVideo.videoPath.name,
            "dataset": cachedVideo.datasetName,
            "target": cachedVideo.targetIndex,
            "target_time_s": cachedVideo.targetTimeScds,
            "predicted": int(predictedIndex),
            "predicted_time_s": predictedTimeScds,
            "frame_error": int(frameError),
            "time_error_s": timeErrorScds,
            "weighted_time_error_s": weightedErrorScds,
        }
    except Exception as err:
        return failureResult(
            cachedVideo.videoPath.name,
            cachedVideo.datasetName,
            cachedVideo.targetIndex,
            cachedVideo.targetTimeScds,
            err,
        )


def metricsFromResults(params, results):
    squaredTimeErrors = []
    absoluteTimeErrors = []
    squaredWeightedTimeErrors = []
    absoluteWeightedTimeErrors = []
    squaredFrameErrors = []
    absoluteFrameErrors = []

    for result in results:
        if result["predicted"] is None:
            squaredTimeErrors.append(LARGE_PENALTY)
            absoluteTimeErrors.append(LARGE_PENALTY)
            squaredWeightedTimeErrors.append(LARGE_PENALTY)
            absoluteWeightedTimeErrors.append(LARGE_PENALTY)
            squaredFrameErrors.append(LARGE_PENALTY)
            absoluteFrameErrors.append(LARGE_PENALTY)
            continue

        timeErrorScds = result["time_error_s"]
        weightedErrorScds = result["weighted_time_error_s"]
        frameError = result["frame_error"]
        squaredTimeErrors.append(timeErrorScds**2)
        absoluteTimeErrors.append(abs(timeErrorScds))
        squaredWeightedTimeErrors.append(weightedErrorScds**2)
        absoluteWeightedTimeErrors.append(abs(weightedErrorScds))
        squaredFrameErrors.append(frameError**2)
        absoluteFrameErrors.append(abs(frameError))

    if not results:
        raise ValueError("Cannot calculate metrics for an empty result set.")

    failureCount = sum(result["predicted"] is None for result in results)
    return {
        "params": params,
        "video_count": len(results),
        "failure_count": int(failureCount),
        "success_rate": float((len(results) - failureCount) / len(results)),
        "mae_s": float(np.mean(absoluteTimeErrors)),
        "rmse_s": float(np.sqrt(np.mean(squaredTimeErrors))),
        "max_abs_error_s": float(np.max(absoluteTimeErrors)),
        "weighted_mae_s": float(np.mean(absoluteWeightedTimeErrors)),
        "weighted_rmse_s": float(np.sqrt(np.mean(squaredWeightedTimeErrors))),
        "weighted_max_abs_error_s": float(np.max(absoluteWeightedTimeErrors)),
        "mae_frames": float(np.mean(absoluteFrameErrors)),
        "rmse_frames": float(np.sqrt(np.mean(squaredFrameErrors))),
        "max_abs_error_frames": float(np.max(absoluteFrameErrors)),
        "results": results,
    }


def evaluateParams(params, cache, verbose=False):
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
                targetVideo.targetIndex,
                None,
                err,
            )
        results.append(result)
        if verbose:
            print(json.dumps(result), flush=True)

    return metricsFromResults(params, results)


def suggestParams(trial):
    return {
        "ksize": trial.suggest_int("ksize", KSIZE_MIN, KSIZE_MAX),
        "blurKernel": trial.suggest_int("blurKernel", BLUR_KERNEL_MIN, BLUR_KERNEL_MAX),
        **FIXED_LAPLACIAN_PARAMS,
        "laplacianPlateau": trial.suggest_float(
            "laplacianPlateau",
            LAPLACIAN_PLATEAU_MIN,
            LAPLACIAN_PLATEAU_MAX,
            log=True,
        ),
        "laplacianSmoothingKernel": trial.suggest_categorical(
            "laplacianSmoothingKernel",
            LAPLACIAN_SMOOTHING_KERNEL_CHOICES,
        ),
        "laplacianGradientSmoothingKernel": trial.suggest_categorical(
            "laplacianGradientSmoothingKernel",
            LAPLACIAN_GRADIENT_SMOOTHING_KERNEL_CHOICES,
        ),
        "indexOffset": trial.suggest_int(
            "indexOffset", INDEX_OFFSET_MIN, INDEX_OFFSET_MAX
        ),
    }


def randomLogUniform(rng, low, high):
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def randomChoice(rng, choices):
    return choices[int(rng.integers(0, len(choices)))]


def randomParams(rng):
    return {
        "ksize": int(rng.integers(KSIZE_MIN, KSIZE_MAX + 1)),
        "blurKernel": int(rng.integers(BLUR_KERNEL_MIN, BLUR_KERNEL_MAX + 1)),
        **FIXED_LAPLACIAN_PARAMS,
        "laplacianPlateau": randomLogUniform(
            rng,
            LAPLACIAN_PLATEAU_MIN,
            LAPLACIAN_PLATEAU_MAX,
        ),
        "laplacianSmoothingKernel": int(
            randomChoice(rng, LAPLACIAN_SMOOTHING_KERNEL_CHOICES)
        ),
        "laplacianGradientSmoothingKernel": int(
            randomChoice(rng, LAPLACIAN_GRADIENT_SMOOTHING_KERNEL_CHOICES)
        ),
        "indexOffset": int(rng.integers(INDEX_OFFSET_MIN, INDEX_OFFSET_MAX + 1)),
    }


def failedTrialMetrics(params, videoCount, exception):
    failureResult = {
        "video": "<trial>",
        "dataset": "",
        "target": None,
        "target_time_s": None,
        "predicted": None,
        "predicted_time_s": None,
        "frame_error": None,
        "time_error_s": None,
        "weighted_time_error_s": None,
        "exception": exception,
    }
    return {
        "params": params,
        "video_count": int(videoCount),
        "failure_count": int(videoCount),
        "success_rate": 0.0,
        "mae_s": LARGE_PENALTY,
        "rmse_s": LARGE_PENALTY,
        "max_abs_error_s": LARGE_PENALTY,
        "weighted_mae_s": LARGE_PENALTY,
        "weighted_rmse_s": LARGE_PENALTY,
        "weighted_max_abs_error_s": LARGE_PENALTY,
        "mae_frames": LARGE_PENALTY,
        "rmse_frames": LARGE_PENALTY,
        "max_abs_error_frames": LARGE_PENALTY,
        "results": [failureResult],
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


def makeObjective(cache, restartIndex, verbose=False):
    bestValue = np.inf
    evaluator = TimedTrialEvaluator(cache, TRIAL_TIMEOUT_SECONDS)

    def objective(trial):
        nonlocal bestValue
        params = suggestParams(trial)
        metrics = evaluator.evaluate(params, trial.number, verbose=verbose)
        value = metrics[VALIDATION_METRIC]

        trial.set_user_attr("mae_s", metrics["mae_s"])
        trial.set_user_attr("max_abs_error_s", metrics["max_abs_error_s"])
        trial.set_user_attr("weighted_mae_s", metrics["weighted_mae_s"])
        trial.set_user_attr(
            "weighted_max_abs_error_s",
            metrics["weighted_max_abs_error_s"],
        )
        trial.set_user_attr("mae_frames", metrics["mae_frames"])
        trial.set_user_attr("max_abs_error_frames", metrics["max_abs_error_frames"])
        trial.set_user_attr("failure_count", metrics["failure_count"])
        trial.set_user_attr("success_rate", metrics["success_rate"])
        trial.set_user_attr("results", metrics["results"])

        if value < bestValue:
            bestValue = value
            print(
                f"restart={restartIndex} "
                f"new_best_{VALIDATION_METRIC}={value:.6f} "
                f"weighted_mae_s={metrics['weighted_mae_s']:.6f} "
                f"raw_rmse_s={metrics['rmse_s']:.6f} "
                f"raw_mae_s={metrics['mae_s']:.6f} "
                f"max_abs_error_s={metrics['max_abs_error_s']:.6f} "
                f"rmse_frames={metrics['rmse_frames']:.3f} "
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


def runRestart(restartIndex, targetVideos, validationSet, validationCache, rng, args):
    optimizationSet = selectOptimizationSet(targetVideos, validationSet, rng)
    optimizationCache = buildCache(optimizationSet, verbose=False)
    initialParams = randomParams(rng)
    samplerSeed = int(rng.integers(0, np.iinfo(np.uint32).max))

    print(
        f"restart={restartIndex}/{args.restarts} "
        f"optimization_videos={len(optimizationSet)} "
        f"optimization_datasets={json.dumps(countByDataset(optimizationSet), sort_keys=True)} "
        f"initial_params={json.dumps(initialParams)}",
        flush=True,
    )

    sampler = optuna.samplers.TPESampler(seed=samplerSeed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.enqueue_trial(initialParams)
    objective = makeObjective(optimizationCache, restartIndex, verbose=args.verbose)
    try:
        study.optimize(
            objective,
            n_trials=args.n_trials,
            timeout=args.timeout,
            show_progress_bar=not args.no_progress_bar and SHOW_PROGRESS_BAR,
        )
    finally:
        objective.close()

    bestParams = {**FIXED_LAPLACIAN_PARAMS, **dict(study.best_trial.params)}
    optimizationMetrics = evaluateParams(bestParams, optimizationCache, verbose=False)
    validationMetrics = evaluateParams(bestParams, validationCache, verbose=False)
    validationValue = validationMetrics[VALIDATION_METRIC]

    print(
        f"restart={restartIndex} "
        f"best_trial={study.best_trial.number} "
        f"optimization_{VALIDATION_METRIC}={optimizationMetrics[VALIDATION_METRIC]:.6f} "
        f"validation_{VALIDATION_METRIC}={validationValue:.6f} "
        f"validation_mae_s={validationMetrics['mae_s']:.6f} "
        f"validation_success_rate={validationMetrics['success_rate']:.6f} "
        f"params={json.dumps(bestParams)}",
        flush=True,
    )

    return {
        "restart": restartIndex,
        "study": study,
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
        f"mae={metrics['mae_s']:.6f} s, "
        f"rmse={metrics['rmse_s']:.6f} s, "
        f"max_abs_error={metrics['max_abs_error_s']:.6f} s"
    )
    print(
        f"weighted_mae={metrics['weighted_mae_s']:.6f} s, "
        f"weighted_rmse={metrics['weighted_rmse_s']:.6f} s, "
        f"weighted_max_abs_error={metrics['weighted_max_abs_error_s']:.6f} s"
    )
    print(
        f"mae={metrics['mae_frames']:.3f} frames, "
        f"rmse={metrics['rmse_frames']:.3f} frames, "
        f"max_abs_error={metrics['max_abs_error_frames']:.3f} frames"
    )
    print("per-video:")
    for result in metrics["results"]:
        if result["predicted"] is None:
            print(f"  {result['video']}: failed ({result['exception']})")
        else:
            print(
                f"  {result['video']}: target={result['target']}, "
                f"predicted={result['predicted']}, "
                f"frame_error={result['frame_error']:+d}, "
                f"time_error={result['time_error_s']:+.6f} s, "
                f"weighted_time_error={result['weighted_time_error_s']:+.6f} s"
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
        "mae_s",
        "rmse_s",
        "max_abs_error_s",
        "weighted_mae_s",
        "weighted_rmse_s",
        "weighted_max_abs_error_s",
        "mae_frames",
        "rmse_frames",
        "max_abs_error_frames",
    ):
        lines.append(f"{key} = {tomlValue(metrics[key])}")


def writeOptimizedParamsToml(
    outputPath,
    bestRecord,
    validationSet,
    finalPerformance,
    restartRecords,
    seed,
):
    lines = []

    appendSection(lines, "run")
    lines.append(f"seed = {tomlValue(seed)}")
    lines.append(f"validation_metric = {tomlValue(VALIDATION_METRIC)}")
    lines.append(f"restarts = {tomlValue(len(restartRecords))}")
    lines.append(f"trials_per_restart = {tomlValue(N_TRIALS_PER_RESTART)}")
    lines.append(f"validation_video_ratio = {tomlValue(VALIDATION_VIDEO_RATIO)}")
    lines.append(f"optimization_video_ratio = {tomlValue(OPTIMIZATION_VIDEO_RATIO)}")
    lines.append(f"cache_dir = {tomlValue(CACHE_DIR)}")
    lines.append(f"trial_timeout_seconds = {tomlValue(TRIAL_TIMEOUT_SECONDS)}")

    appendSection(lines, "best")
    lines.append(f"restart = {tomlValue(bestRecord['restart'])}")
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

    for record in restartRecords:
        appendTableArray(lines, "restarts_summary")
        lines.append(f"restart = {tomlValue(record['restart'])}")
        lines.append(f"best_trial = {tomlValue(record['study'].best_trial.number)}")
        lines.append(
            f"optimization_{VALIDATION_METRIC} = "
            f"{tomlValue(record['optimization_metrics'][VALIDATION_METRIC])}"
        )
        lines.append(
            f"validation_{VALIDATION_METRIC} = "
            f"{tomlValue(record['validation_metrics'][VALIDATION_METRIC])}"
        )
        lines.append(
            f"validation_success_rate = "
            f"{tomlValue(record['validation_metrics']['success_rate'])}"
        )

    outputPath.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parseArgs():
    parser = argparse.ArgumentParser(
        description=(
            "Optimize Laplacian release detection parameters with repeated random "
            "optimization splits and one fixed validation split."
        )
    )
    parser.add_argument("--restarts", type=int, default=N_RESTARTS)
    parser.add_argument("--n-trials", type=int, default=N_TRIALS_PER_RESTART)
    parser.add_argument("--timeout", type=float, default=OPTUNA_TIMEOUT)
    parser.add_argument("--seed", type=int, default=OPTUNA_SEED)
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Evaluate default params against all discovered target videos.",
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
    global N_RESTARTS
    global N_TRIALS_PER_RESTART

    args = parseArgs()
    N_RESTARTS = args.restarts
    N_TRIALS_PER_RESTART = args.n_trials

    if args.restarts < 1:
        raise ValueError("--restarts must be positive.")
    if args.n_trials < 1:
        raise ValueError("--n-trials must be positive.")

    targetVideos = filterTargetVideos(
        discoverTargetVideos(verbose=True),
        args.videos,
    )
    if not targetVideos:
        raise ValueError("No usable target videos were found.")

    if args.evaluate_only:
        metrics = evaluateTargetVideos(
            DEFAULT_PARAMS, targetVideos, verbose=args.verbose
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
    bestValidationValue = np.inf
    restartRecords = []
    for restartIndex in range(1, args.restarts + 1):
        record = runRestart(
            restartIndex,
            targetVideos,
            validationSet,
            validationCache,
            rng,
            args,
        )
        restartRecords.append(record)
        validationValue = record["validation_metrics"][VALIDATION_METRIC]
        if validationValue < bestValidationValue:
            bestValidationValue = validationValue
            bestRecord = record
            print(
                f"new_best_restart={restartIndex} "
                f"validation_{VALIDATION_METRIC}={validationValue:.6f}",
                flush=True,
            )

    if bestRecord is None:
        raise RuntimeError("No restart completed successfully.")

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
        restartRecords,
        args.seed,
    )

    print(f"best_restart={bestRecord['restart']}")
    print("validation performance:")
    printMetrics(bestRecord["validation_metrics"])
    print("all-datasets performance:")
    printMetrics(finalPerformance["all"])
    for datasetName, metrics in finalPerformance["datasets"].items():
        print(f"{datasetName} performance:")
        printMetrics(metrics)
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
