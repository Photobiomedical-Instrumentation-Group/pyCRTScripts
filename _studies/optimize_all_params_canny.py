import argparse
import json
import os
from copy import deepcopy
from pathlib import Path

import numpy as np
import optuna
import tomllib

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-pycrt")

from crt_pipeline_analysis import run_crt_pipeline_analysis
from plot_crt_pipeline_vs_reference import (
    loadReferenceMeasurements,
)

REFERENCE_CSV_PATH = Path("raquel_masters_roi_pcrt.csv")

GIVEN_PARAMETERS = {
    "rois_path": Path("rois_full.toml"),
    "cache_dir": Path("Npz/Cache"),
    "output_csv_path": Path("crt_pipeline_comparison.csv"),
    "video_extensions": [".MOV", ".wmv", ".mp4"],
    "write_csv": True,
    "verbose": False,
    "filter_type": "canny",
    "max_crt90_10_fit_seconds": 10.0,
    "crt90_10_gaussian_sigma_seconds": 0.1,
    "crt90_10_bootstrap_count": 500,
    "crt90_10_bootstrap_seed": 0,
    "crt90_10_min": 0.0,
    "crt90_10_max": 10.0,
    "crt90_10_relative_uncertainty_min": 0.0,
    "crt90_10_relative_uncertainty_max": 0.5,
}

PARAMETERS_TO_OPTIMIZE_DEFAULTS = {
    "relax_max_grad": 2.0,
    "relax_min_grad": -3.0,
    "strict_max_grad": 1,
    "strict_min_grad": -0.5,
    "canny_params": {
        "thresh_1": 56,
        "thresh_2": 85,
        "blur_kernel": 6,
        "l2_grad": False,
        "canny_plateau": 0.1,
        "canny_smoothing_kernel": 9,
        "canny_gradient_smoothing_kernel": 1,
        "index_offset": 0,
    },
}

N_TRIALS = 100
OPTUNA_TIMEOUT = None
OPTUNA_SEED = 2468
SHOW_PROGRESS_BAR = True
MIN_R2_REFERENCE_VIDEOS = 2
OPTIMIZATION_BOOTSTRAP_COUNT = 50
STAGE1_REFERENCE_VIDEO_COUNT = 40
STAGE1_NON_REFERENCE_VIDEO_COUNT = 20
STAGE2_TRIAL_COUNT = 10
PROGRESSIVE_BATCH_SIZE = 20
PRUNER_STARTUP_TRIALS = 10
PRUNER_WARMUP_STEPS = 1
DATASET_SELECTION_SEED = 2468


def runOnFromCache(parameters):
    cacheDir = Path(parameters["cache_dir"])
    roisPath = Path(parameters["rois_path"])
    with roisPath.open("rb") as file:
        roiConfig = tomllib.load(file)

    cacheStems = {cachePath.stem for cachePath in cacheDir.glob("*.npz")}
    videoNames = sorted(roiConfig.get("roi", {}).keys(), key=str.lower)
    runOn = [
        Path(videoName)
        for videoName in videoNames
        if Path(videoName).stem in cacheStems
    ]
    if not runOn:
        raise ValueError(
            f"No cached videos with matching ROI names found in {cacheDir} "
            f"and {roisPath}."
        )
    return runOn


def mergeDictionaries(base, overrides):
    merged = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = mergeDictionaries(merged[key], value)
        else:
            merged[key] = value
    return merged


def composeParameters(optimizedParameters, writeCsv=False, verbose=False):
    return mergeDictionaries(
        GIVEN_PARAMETERS,
        {
            **optimizedParameters,
            "write_csv": writeCsv,
            "verbose": verbose,
        },
    )


def flattenOptimizedParameters(parameters):
    cannyParams = parameters["canny_params"]
    return {
        "relax_max_grad": parameters["relax_max_grad"],
        "relax_min_grad": parameters["relax_min_grad"],
        "strict_max_grad": parameters["strict_max_grad"],
        "strict_min_grad": parameters["strict_min_grad"],
        "thresh_1": cannyParams["thresh_1"],
        "thresh_2": cannyParams["thresh_2"],
        "blur_kernel": cannyParams["blur_kernel"],
        "l2_grad": cannyParams["l2_grad"],
        "canny_plateau": cannyParams["canny_plateau"],
        "canny_smoothing_kernel": cannyParams["canny_smoothing_kernel"],
        "canny_gradient_smoothing_kernel": cannyParams[
            "canny_gradient_smoothing_kernel"
        ],
        "index_offset": cannyParams["index_offset"],
    }


def nestedOptimizedParameters(flatParameters):
    return {
        "relax_max_grad": flatParameters["relax_max_grad"],
        "relax_min_grad": flatParameters["relax_min_grad"],
        "strict_max_grad": flatParameters["strict_max_grad"],
        "strict_min_grad": flatParameters["strict_min_grad"],
        "canny_params": {
            "thresh_1": int(flatParameters["thresh_1"]),
            "thresh_2": int(flatParameters["thresh_2"]),
            "blur_kernel": int(flatParameters["blur_kernel"]),
            "l2_grad": bool(flatParameters["l2_grad"]),
            "canny_plateau": flatParameters["canny_plateau"],
            "canny_smoothing_kernel": int(flatParameters["canny_smoothing_kernel"]),
            "canny_gradient_smoothing_kernel": int(
                flatParameters["canny_gradient_smoothing_kernel"]
            ),
            "index_offset": int(flatParameters["index_offset"]),
        },
    }


def suggestGradientParameters(trial):
    strictMinGrad = trial.suggest_float("strict_min_grad", -5.0, 0.0)
    relaxMinGrad = trial.suggest_float("relax_min_grad", -10.0, strictMinGrad)

    relaxMaxGrad = trial.suggest_float("relax_max_grad", 2.0, 10.0)
    strictMaxGrad = trial.suggest_float(
        "strict_max_grad",
        max(-0.1, strictMinGrad),
        relaxMaxGrad,
    )

    return {
        "relax_max_grad": relaxMaxGrad,
        "relax_min_grad": relaxMinGrad,
        "strict_max_grad": strictMaxGrad,
        "strict_min_grad": strictMinGrad,
    }


def suggestOptimizedParameters(trial):
    thresh1 = trial.suggest_int("thresh_1", 1, 100)
    thresh2 = trial.suggest_int("thresh_2", thresh1 + 1, 180)
    flatParameters = {
        **suggestGradientParameters(trial),
        "thresh_1": thresh1,
        "thresh_2": thresh2,
        "blur_kernel": trial.suggest_int("blur_kernel", 0, 8),
        "l2_grad": trial.suggest_categorical("l2_grad", [False, True]),
        "canny_plateau": trial.suggest_float(
            "canny_plateau",
            0.01,
            20.0,
            log=True,
        ),
        "canny_smoothing_kernel": trial.suggest_categorical(
            "canny_smoothing_kernel",
            [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21],
        ),
        "canny_gradient_smoothing_kernel": trial.suggest_categorical(
            "canny_gradient_smoothing_kernel",
            [1, 3, 5],
        ),
        "index_offset": trial.suggest_int("index_offset", -5, 5),
    }
    return nestedOptimizedParameters(flatParameters)


def isFinitePositive(value):
    return np.isfinite(value) and value > 0


def isFiniteNonnegative(value):
    return np.isfinite(value) and value >= 0


def hasValidReference(videoPath, referenceMeasurements):
    reference = referenceMeasurements.get(Path(videoPath).name)
    if reference is None:
        return False
    return isFinitePositive(reference["reference_pcrt"]) and isFiniteNonnegative(
        reference["reference_pcrt_uncertainty"]
    )


def selectFixedSubset(paths, count, rng):
    paths = list(paths)
    if count >= len(paths):
        return paths
    selectedIndices = rng.choice(len(paths), size=count, replace=False)
    return [paths[int(index)] for index in selectedIndices]


def selectStage1RunOn(
    runOn,
    referenceMeasurements,
    referenceVideoCount,
    nonReferenceVideoCount,
    seed=DATASET_SELECTION_SEED,
):
    referencePaths = [
        videoPath
        for videoPath in runOn
        if hasValidReference(videoPath, referenceMeasurements)
    ]
    nonReferencePaths = [
        videoPath
        for videoPath in runOn
        if not hasValidReference(videoPath, referenceMeasurements)
    ]

    rng = np.random.default_rng(seed)
    selectedPaths = [
        *selectFixedSubset(referencePaths, referenceVideoCount, rng),
        *selectFixedSubset(nonReferencePaths, nonReferenceVideoCount, rng),
    ]
    rng.shuffle(selectedPaths)
    return selectedPaths


def crt90_10PassesFilters(value, uncertainty, parameters):
    if not (isFinitePositive(value) and isFiniteNonnegative(uncertainty)):
        return False
    relativeUncertainty = uncertainty / value
    return (
        parameters["crt90_10_min"] <= value <= parameters["crt90_10_max"]
        and parameters["crt90_10_relative_uncertainty_min"]
        <= relativeUncertainty
        <= parameters["crt90_10_relative_uncertainty_max"]
    )


def fitLineAdjustedR2(xValues, yValues):
    if len(xValues) < MIN_R2_REFERENCE_VIDEOS:
        return None

    slope, intercept = np.polyfit(xValues, yValues, 1)
    fittedValues = slope * xValues + intercept
    ssResidual = np.sum((yValues - fittedValues) ** 2)
    ssTotal = np.sum((yValues - np.mean(yValues)) ** 2)
    if ssTotal == 0:
        return None
    r2 = float(1 - ssResidual / ssTotal)
    sampleCount = len(xValues)
    predictorCount = 1
    if sampleCount <= predictorCount + 1:
        return None
    adjustedR2 = 1 - (1 - r2) * (sampleCount - 1) / (
        sampleCount - predictorCount - 1
    )
    return float(adjustedR2)


def labACrt90_10ReferenceAdjustedR2(rows, referenceMeasurements, parameters):
    xValues = []
    yValues = []
    validReferenceCount = 0
    for row in rows:
        reference = referenceMeasurements.get(row["video"])
        if reference is None:
            continue

        referenceValue = reference["reference_pcrt"]
        referenceUncertainty = reference["reference_pcrt_uncertainty"]
        if not (
            isFinitePositive(referenceValue)
            and isFiniteNonnegative(referenceUncertainty)
        ):
            continue

        validReferenceCount += 1
        metricValue = float(row["a_crt90_10_s"])
        metricUncertainty = float(row["a_crt90_10_uncertainty_s"])
        if not crt90_10PassesFilters(metricValue, metricUncertainty, parameters):
            continue

        xValues.append(referenceValue)
        yValues.append(metricValue)

    adjustedR2 = fitLineAdjustedR2(np.asarray(xValues), np.asarray(yValues))
    referenceVideoCount = len(xValues)
    return {
        "adjusted_r2": (
            0.0
            if adjustedR2 is None or not np.isfinite(adjustedR2)
            else max(0.0, adjustedR2)
        ),
        "reference_video_count": referenceVideoCount,
        "valid_reference_count": validReferenceCount,
        "reference_coverage": (
            referenceVideoCount / validReferenceCount if validReferenceCount else 0.0
        ),
    }


def scoreRows(rows, referenceMeasurements, parameters):
    allVideoSuccessCount = sum(
        crt90_10PassesFilters(
            float(row["a_crt90_10_s"]),
            float(row["a_crt90_10_uncertainty_s"]),
            parameters,
        )
        for row in rows
    )
    allVideoSuccessRate = (
        allVideoSuccessCount / len(rows) if len(rows) else 0.0
    )
    r2Result = labACrt90_10ReferenceAdjustedR2(
        rows,
        referenceMeasurements,
        parameters,
    )
    score = (
        allVideoSuccessRate
        * r2Result["reference_coverage"]
        * r2Result["adjusted_r2"]
    )
    return {
        "score": score,
        "all_video_success_count": allVideoSuccessCount,
        "all_video_success_rate": allVideoSuccessRate,
        "reference_coverage": r2Result["reference_coverage"],
        "adjusted_r2": r2Result["adjusted_r2"],
        "reference_video_count": r2Result["reference_video_count"],
        "valid_reference_count": r2Result["valid_reference_count"],
    }


def scoreResult(result, referenceMeasurements, parameters):
    return scoreRows(result["rows"], referenceMeasurements, parameters)


def composeEvaluationParameters(
    optimizedParameters,
    writeCsv=False,
    verbose=False,
    bootstrapCount=None,
):
    parameters = composeParameters(
        optimizedParameters,
        writeCsv=writeCsv,
        verbose=verbose,
    )
    if bootstrapCount is not None:
        parameters["crt90_10_bootstrap_count"] = bootstrapCount
    return parameters


def evaluateOptimizedParameters(
    optimizedParameters,
    referenceMeasurements,
    writeCsv,
    runOn=None,
    bootstrapCount=None,
):
    parameters = composeEvaluationParameters(
        optimizedParameters,
        writeCsv=writeCsv,
        verbose=False,
        bootstrapCount=bootstrapCount,
    )
    if runOn is None:
        runOn = runOnFromCache(parameters)
    result = run_crt_pipeline_analysis(
        runOn,
        parameters,
        show_plot=False,
    )
    score = scoreResult(result, referenceMeasurements, parameters)
    return result, score


def iterBatches(values, batchSize):
    if batchSize < 1:
        raise ValueError(f"batchSize must be positive, got {batchSize}.")
    for startIndex in range(0, len(values), batchSize):
        yield values[startIndex : startIndex + batchSize]


def optimisticScoreUpperBound(
    score,
    processedVideoCount,
    totalVideoCount,
    totalValidReferenceCount,
):
    remainingVideoCount = totalVideoCount - processedVideoCount
    maximumSuccessRate = (
        score["all_video_success_count"] + remainingVideoCount
    ) / totalVideoCount

    remainingReferenceCount = (
        totalValidReferenceCount - score["valid_reference_count"]
    )
    maximumReferenceCoverage = (
        score["reference_video_count"] + remainingReferenceCount
    ) / totalValidReferenceCount
    return maximumSuccessRate * maximumReferenceCoverage


def evaluateOptimizedParametersProgressively(
    optimizedParameters,
    referenceMeasurements,
    runOn,
    bootstrapCount,
    batchSize,
    trial=None,
    scoreToBeat=None,
):
    parameters = composeEvaluationParameters(
        optimizedParameters,
        writeCsv=False,
        verbose=False,
        bootstrapCount=bootstrapCount,
    )
    runOn = list(runOn)
    totalVideoCount = len(runOn)
    totalValidReferenceCount = sum(
        hasValidReference(videoPath, referenceMeasurements) for videoPath in runOn
    )
    rows = []

    for batchIndex, batchRunOn in enumerate(
        iterBatches(runOn, batchSize),
        start=1,
    ):
        batchResult = run_crt_pipeline_analysis(
            batchRunOn,
            parameters,
            show_plot=False,
        )
        rows.extend(batchResult["rows"])
        score = scoreRows(rows, referenceMeasurements, parameters)

        if trial is not None:
            trial.report(score["score"], step=batchIndex)

        if totalValidReferenceCount:
            upperBound = optimisticScoreUpperBound(
                score,
                len(rows),
                totalVideoCount,
                totalValidReferenceCount,
            )
            if (
                scoreToBeat is not None
                and np.isfinite(scoreToBeat)
                and upperBound <= scoreToBeat
            ):
                raise optuna.TrialPruned(
                    f"optimistic score {upperBound:.6f} cannot beat "
                    f"{scoreToBeat:.6f}"
                )

        if trial is not None and trial.should_prune():
            raise optuna.TrialPruned(
                f"median pruner stopped trial after {len(rows)} videos"
            )

    return {"rows": rows}, scoreRows(rows, referenceMeasurements, parameters)


def makeObjective(
    referenceMeasurements,
    runOn,
    bootstrapCount,
    batchSize,
):
    bestScore = -np.inf

    def objective(trial):
        nonlocal bestScore
        try:
            optimizedParameters = suggestOptimizedParameters(trial)
            _, score = evaluateOptimizedParametersProgressively(
                optimizedParameters,
                referenceMeasurements,
                runOn,
                bootstrapCount,
                batchSize,
                trial=trial,
                scoreToBeat=bestScore,
            )
        except optuna.TrialPruned:
            raise
        except Exception as err:
            trial.set_user_attr("exception", f"{type(err).__name__}: {err}")
            return -1.0

        trial.set_user_attr(
            "all_video_success_rate",
            score["all_video_success_rate"],
        )
        trial.set_user_attr("reference_coverage", score["reference_coverage"])
        trial.set_user_attr("adjusted_r2", score["adjusted_r2"])
        trial.set_user_attr("reference_video_count", score["reference_video_count"])
        trial.set_user_attr("valid_reference_count", score["valid_reference_count"])
        trial.set_user_attr("optimized_parameters", optimizedParameters)

        if score["score"] > bestScore:
            bestScore = score["score"]
            print(
                f"new_best_score={score['score']:.6f} "
                f"all_video_success_rate={score['all_video_success_rate']:.6f} "
                f"reference_coverage={score['reference_coverage']:.6f} "
                f"adjusted_r2={score['adjusted_r2']:.6f} "
                f"reference_videos={score['reference_video_count']}/"
                f"{score['valid_reference_count']} "
                f"trial={trial.number} "
                f"params={json.dumps(optimizedParameters)}",
                flush=True,
            )

        return score["score"]

    return objective


def topCompletedTrials(study, count):
    completedTrials = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
        and "optimized_parameters" in trial.user_attrs
    ]
    return sorted(
        completedTrials,
        key=lambda trial: trial.value,
        reverse=True,
    )[:count]


def evaluateStage2Candidates(
    stage1Trials,
    referenceMeasurements,
    runOn,
    bootstrapCount,
    batchSize,
):
    bestTrial = None
    bestScore = -np.inf

    for candidateIndex, trial in enumerate(stage1Trials, start=1):
        try:
            _, score = evaluateOptimizedParametersProgressively(
                trial.user_attrs["optimized_parameters"],
                referenceMeasurements,
                runOn,
                bootstrapCount,
                batchSize,
                scoreToBeat=bestScore,
            )
        except optuna.TrialPruned as err:
            print(
                f"stage2_pruned trial={trial.number} "
                f"candidate={candidateIndex}/{len(stage1Trials)} reason={err}",
                flush=True,
            )
            continue

        print(
            f"stage2_score={score['score']:.6f} "
            f"all_video_success_rate={score['all_video_success_rate']:.6f} "
            f"reference_coverage={score['reference_coverage']:.6f} "
            f"adjusted_r2={score['adjusted_r2']:.6f} "
            f"trial={trial.number} "
            f"candidate={candidateIndex}/{len(stage1Trials)}",
            flush=True,
        )
        if score["score"] > bestScore:
            bestTrial = trial
            bestScore = score["score"]

    if bestTrial is None:
        raise RuntimeError("No stage-two candidate completed full-dataset evaluation.")
    return bestTrial


def printFinalResult(selectedTrial, result, score):
    print(f"selected_stage1_trial={selectedTrial.number}")
    print(f"best_score={score['score']:.6f}")
    print(f"all_video_success_rate={score['all_video_success_rate']:.6f}")
    print(f"reference_coverage={score['reference_coverage']:.6f}")
    print(f"adjusted_r2={score['adjusted_r2']:.6f}")
    print(
        f"reference_video_count={score['reference_video_count']}/"
        f"{score['valid_reference_count']}"
    )
    print("best_parameters:")
    print(json.dumps(selectedTrial.user_attrs["optimized_parameters"], indent=2))
    print("filtered_success_rates:")
    print(json.dumps(result["filtered_success_rates"], indent=2))
    print("r2_values:")
    print(json.dumps(result["r2_values"], indent=2))


def parseArgs():
    parser = argparse.ArgumentParser(
        description=(
            "Optimize all Canny release-detection parameters for LAB A CRT90_10 "
            "success plus correlation with reference pCRT values."
        )
    )
    parser.add_argument("--n-trials", type=int, default=N_TRIALS)
    parser.add_argument("--timeout", type=float, default=OPTUNA_TIMEOUT)
    parser.add_argument("--seed", type=int, default=OPTUNA_SEED)
    parser.add_argument(
        "--optimization-bootstraps",
        type=int,
        default=OPTIMIZATION_BOOTSTRAP_COUNT,
        help="Bootstrap count used during both optimization stages.",
    )
    parser.add_argument(
        "--stage1-reference-videos",
        type=int,
        default=STAGE1_REFERENCE_VIDEO_COUNT,
        help="Number of reference videos in the fixed stage-one subset.",
    )
    parser.add_argument(
        "--stage1-non-reference-videos",
        type=int,
        default=STAGE1_NON_REFERENCE_VIDEO_COUNT,
        help="Number of non-reference videos in the fixed stage-one subset.",
    )
    parser.add_argument(
        "--stage2-trials",
        type=int,
        default=STAGE2_TRIAL_COUNT,
        help="Number of top stage-one trials evaluated on the full dataset.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=PROGRESSIVE_BATCH_SIZE,
        help="Videos evaluated between Optuna pruning decisions.",
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Evaluate PARAMETERS_TO_OPTIMIZE_DEFAULTS without running Optuna.",
    )
    parser.add_argument(
        "--no-progress-bar",
        action="store_true",
        help="Disable Optuna's progress bar.",
    )
    return parser.parse_args()


def main():
    args = parseArgs()
    referenceMeasurements = loadReferenceMeasurements(REFERENCE_CSV_PATH)

    if args.evaluate_only:
        result, score = evaluateOptimizedParameters(
            PARAMETERS_TO_OPTIMIZE_DEFAULTS,
            referenceMeasurements,
            writeCsv=True,
        )
        print("evaluated default parameters")
        print(f"score={score['score']:.6f}")
        print(f"all_video_success_rate={score['all_video_success_rate']:.6f}")
        print(f"reference_coverage={score['reference_coverage']:.6f}")
        print(f"adjusted_r2={score['adjusted_r2']:.6f}")
        print(
            f"reference_video_count={score['reference_video_count']}/"
            f"{score['valid_reference_count']}"
        )
        print(json.dumps(result["filtered_success_rates"], indent=2))
        return

    if args.n_trials < 1:
        raise ValueError("--n-trials must be positive.")
    if args.optimization_bootstraps < 1:
        raise ValueError("--optimization-bootstraps must be positive.")
    if args.stage1_reference_videos < MIN_R2_REFERENCE_VIDEOS + 1:
        raise ValueError(
            "--stage1-reference-videos must provide at least three references "
            "for adjusted R2."
        )
    if args.stage1_non_reference_videos < 0:
        raise ValueError("--stage1-non-reference-videos cannot be negative.")
    if args.stage2_trials < 1:
        raise ValueError("--stage2-trials must be positive.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive.")

    allRunOn = runOnFromCache(GIVEN_PARAMETERS)
    stage1RunOn = selectStage1RunOn(
        allRunOn,
        referenceMeasurements,
        args.stage1_reference_videos,
        args.stage1_non_reference_videos,
    )
    stage1ReferenceCount = sum(
        hasValidReference(videoPath, referenceMeasurements)
        for videoPath in stage1RunOn
    )
    print(
        f"stage1_dataset={len(stage1RunOn)} "
        f"reference_videos={stage1ReferenceCount} "
        f"non_reference_videos={len(stage1RunOn) - stage1ReferenceCount} "
        f"optimization_bootstraps={args.optimization_bootstraps}",
        flush=True,
    )

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=PRUNER_STARTUP_TRIALS,
        n_warmup_steps=PRUNER_WARMUP_STEPS,
        interval_steps=1,
    )
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )
    study.enqueue_trial(flattenOptimizedParameters(PARAMETERS_TO_OPTIMIZE_DEFAULTS))
    study.optimize(
        makeObjective(
            referenceMeasurements,
            stage1RunOn,
            args.optimization_bootstraps,
            args.batch_size,
        ),
        n_trials=args.n_trials,
        timeout=args.timeout,
        show_progress_bar=not args.no_progress_bar and SHOW_PROGRESS_BAR,
    )

    stage1Trials = topCompletedTrials(study, args.stage2_trials)
    if not stage1Trials:
        raise RuntimeError("No stage-one trial completed successfully.")
    print(
        f"stage2_candidates={len(stage1Trials)} "
        f"full_dataset_videos={len(allRunOn)}",
        flush=True,
    )
    selectedTrial = evaluateStage2Candidates(
        stage1Trials,
        referenceMeasurements,
        allRunOn,
        args.optimization_bootstraps,
        args.batch_size,
    )

    bestOptimizedParameters = selectedTrial.user_attrs["optimized_parameters"]
    result, score = evaluateOptimizedParameters(
        bestOptimizedParameters,
        referenceMeasurements,
        writeCsv=True,
        runOn=allRunOn,
    )
    printFinalResult(selectedTrial, result, score)


if __name__ == "__main__":
    main()
