import csv
import os
from copy import deepcopy
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

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

CSV_FIELD_NAMES = [
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

ERROR_ROI = -1
ERROR_CACHE = -2
ERROR_METRIC = -3
ERROR_CRT_INTERVAL = -4
ERROR_CRT90_10 = -5
ERROR_CRT90_10_UNCERTAINTY = -6
ERROR_PCRT = -7

METRICS = {
    "g_crt90_10": {
        "label": "BGR G CRT90_10",
        "kind": "crt90_10",
    },
    "g_pcrt": {
        "label": "BGR G pCRT",
        "kind": "pcrt",
    },
    "a_crt90_10": {
        "label": "LAB A CRT90_10",
        "kind": "crt90_10",
    },
    "a_pcrt": {
        "label": "LAB A pCRT",
        "kind": "pcrt",
    },
}

PLOT_SPECS = [
    {
        "key": "bgr_g_pcrt_vs_crt90_10",
        "title": "BGR G",
        "x_metric": "g_crt90_10",
        "y_metric": "g_pcrt",
    },
    {
        "key": "lab_a_pcrt_vs_crt90_10",
        "title": "LAB A",
        "x_metric": "a_crt90_10",
        "y_metric": "a_pcrt",
    },
    {
        "key": "lab_a_crt90_10_vs_bgr_g_crt90_10",
        "title": "CRT90_10 LAB A vs BGR G",
        "x_metric": "g_crt90_10",
        "y_metric": "a_crt90_10",
    },
    {
        "key": "bgr_g_pcrt_vs_lab_a_pcrt",
        "title": "pCRT BGR G vs LAB A",
        "x_metric": "a_pcrt",
        "y_metric": "g_pcrt",
    },
]


DEFAULT_PARAMETERS = {
    "rois_path": "rois_full.toml",
    "cache_dir": Path("Npz/Cache"),
    "output_csv_path": Path("crt_pipeline_comparison.csv"),
    "video_extensions": [".MOV", ".wmv", ".mp4"],
    "write_csv": True,
    "verbose": True,
    "filter_type": "canny",
    "relax_max_grad": 1.0,
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
        "strict_min_grad": -0.5,
        "strict_max_grad": 1,
        "relax_min_grad": -3.0,
        "relax_max_grad": 1.0,
        "index_offset": 0,
    },
    "laplacian_params": {
        "ksize": 3,
        "blur_kernel": 7,
        "scale": 1,
        "delta": 0,
        "laplacian_plateau": 2.0,
        "laplacian_smoothing_kernel": 19,
        "laplacian_gradient_smoothing_kernel": 3,
        "strict_min_grad": -0.5,
        "strict_max_grad": 1,
        "relax_min_grad": -3.0,
        "relax_max_grad": 1.0,
        "index_offset": 0,
    },
    "max_crt90_10_fit_seconds": DEFAULT_MAX_CRT90_10_FIT_SECONDS,
    "crt90_10_gaussian_sigma_seconds": DEFAULT_CRT90_10_GAUSSIAN_SIGMA_SECONDS,
    "crt90_10_bootstrap_count": DEFAULT_CRT90_10_BOOTSTRAP_COUNT,
    "crt90_10_bootstrap_seed": DEFAULT_CRT90_10_BOOTSTRAP_SEED,
    "crt90_10_min": 0.0,
    "crt90_10_max": 10.0,
    "pcrt_min": 0.0,
    "pcrt_max": 10.0,
    "crt90_10_relative_uncertainty_min": 0.0,
    "crt90_10_relative_uncertainty_max": 1.0,
    "pcrt_relative_uncertainty_min": 0.0,
    "pcrt_relative_uncertainty_max": 1.0,
}


def mergeParameters(defaults, overrides):
    merged = deepcopy(defaults)
    if not overrides:
        return merged

    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = mergeParameters(merged[key], value)
        else:
            merged[key] = value
    return merged


def applyTopLevelGradientParameters(parameters, userParameters):
    userParameters = userParameters or {}
    for filterKey in ("canny_params", "laplacian_params"):
        userFilterParams = userParameters.get(filterKey, {})
        for key in ("strict_min_grad", "strict_max_grad", "relax_min_grad", "relax_max_grad"):
            if key in userFilterParams:
                continue
            parameters[filterKey][key] = parameters[key]


def normalizeParameters(parameters):
    normalized = deepcopy(parameters)
    normalized["cache_dir"] = Path(normalized["cache_dir"])
    normalized["output_csv_path"] = Path(normalized["output_csv_path"])
    normalized["video_extensions"] = {
        suffix.lower() for suffix in normalized["video_extensions"]
    }
    normalized["filter_type"] = normalized["filter_type"].lower()
    return normalized


def buildParameters(userParameters=None):
    parameters = mergeParameters(DEFAULT_PARAMETERS, userParameters)
    applyTopLevelGradientParameters(parameters, userParameters)
    return normalizeParameters(parameters)


def iterVideoPathsInDirectory(videoDir, videoExtensions):
    for filePath in sorted(
        Path(videoDir).iterdir(), key=lambda path: path.name.lower()
    ):
        if filePath.suffix.lower() in videoExtensions:
            yield filePath


def iterRunOnVideoPaths(runOn, videoExtensions):
    if isinstance(runOn, (list, tuple, set)):
        for item in runOn:
            yield from iterRunOnVideoPaths(item, videoExtensions)
        return

    runOnPath = Path(runOn)
    if runOnPath.is_dir():
        yield from iterVideoPathsInDirectory(runOnPath, videoExtensions)
        return

    yield runOnPath


def cachePathForVideo(videoPath, cacheDir):
    return cacheDir / f"{videoPath.stem}.npz"


def roiToString(roi):
    if roi is None:
        return str(ERROR_ROI)
    return ",".join(str(value) for value in roi)


def loadCachedVideo(videoPath, cacheDir):
    cachePath = cachePathForVideo(videoPath, cacheDir)
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


def cannyParamsForRelease(params):
    return {
        "thresh1": params["thresh_1"],
        "thresh2": params["thresh_2"],
        "blurKernel": params["blur_kernel"],
        "l2grad": params["l2_grad"],
        "cannyPlateau": params["canny_plateau"],
        "cannySmoothingKernel": params["canny_smoothing_kernel"],
        "cannyGradientSmoothingKernel": params["canny_gradient_smoothing_kernel"],
        "strictMinGrad": params["strict_min_grad"],
        "strictMaxGrad": params["strict_max_grad"],
        "relaxMinGrad": params["relax_min_grad"],
        "relaxMaxGrad": params["relax_max_grad"],
        "indexOffset": params["index_offset"],
    }


def laplacianParamsForRelease(params):
    return {
        "ksize": params["ksize"],
        "blurKernel": params["blur_kernel"],
        "scale": params["scale"],
        "delta": params["delta"],
        "laplacianPlateau": params["laplacian_plateau"],
        "laplacianSmoothingKernel": params["laplacian_smoothing_kernel"],
        "laplacianGradientSmoothingKernel": params[
            "laplacian_gradient_smoothing_kernel"
        ],
        "strictMinGrad": params["strict_min_grad"],
        "strictMaxGrad": params["strict_max_grad"],
        "relaxMinGrad": params["relax_min_grad"],
        "relaxMaxGrad": params["relax_max_grad"],
        "indexOffset": params["index_offset"],
    }


def paramsForFilter(parameters):
    if parameters["filter_type"] == "canny":
        cannyParams = cannyParamsForRelease(parameters["canny_params"])
        return cannyParams, cannyParams
    if parameters["filter_type"] == "laplacian":
        laplacianParams = laplacianParamsForRelease(parameters["laplacian_params"])
        return laplacianParams, releaseParamsFromLaplacianParams(laplacianParams)
    raise ValueError(f"Unsupported filter_type: {parameters['filter_type']}")


def metricArrFromCachedFrames(lFrames, filterParams, parameters):
    if parameters["filter_type"] == "canny":
        return cannySumsFromLFrames(lFrames, filterParams)
    if parameters["filter_type"] == "laplacian":
        return laplacianSumsFromLFrames(lFrames, filterParams)
    raise ValueError(f"Unsupported filter_type: {parameters['filter_type']}")


def emptyMeasurement(errorCode):
    return {
        "crt90_10": float(errorCode),
        "crt90_10_uncertainty": float(errorCode),
        "pcrt": float(errorCode),
        "pcrt_uncertainty": float(errorCode),
    }


def measureChannel(signalArr, timeArr, metricArr, releaseParams, parameters):
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
    except Exception:
        return emptyMeasurement(ERROR_CRT_INTERVAL)

    try:
        calcCRT90_10Gaussian(
            timeArr,
            signalArr,
            indices,
            maxFitSeconds=parameters["max_crt90_10_fit_seconds"],
            gaussianSigmaSeconds=parameters["crt90_10_gaussian_sigma_seconds"],
        )
    except Exception:
        crt90_10Value = ERROR_CRT90_10
        crt90_10UncertaintyValue = ERROR_CRT90_10
    else:
        try:
            crt90_10Uncertainty = calcCRT90_10BootstrapUncertainty(
                timeArr,
                signalArr,
                indices,
                maxFitSeconds=parameters["max_crt90_10_fit_seconds"],
                gaussianSigmaSeconds=parameters["crt90_10_gaussian_sigma_seconds"],
                nBootstraps=parameters["crt90_10_bootstrap_count"],
                randomSeed=parameters["crt90_10_bootstrap_seed"],
            )
        except Exception:
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


def measureVideo(videoPath, parameters):
    try:
        roi = loadVideoRoi(videoPath, parameters["rois_path"])
    except (KeyError, ValueError):
        return (
            roiToString(None),
            emptyMeasurement(ERROR_ROI),
            emptyMeasurement(ERROR_ROI),
        )

    try:
        cachedVideo = loadCachedVideo(videoPath, parameters["cache_dir"])
    except Exception:
        return (
            roiToString(roi),
            emptyMeasurement(ERROR_CACHE),
            emptyMeasurement(ERROR_CACHE),
        )

    filterParams, releaseParams = paramsForFilter(parameters)
    try:
        metricArr = metricArrFromCachedFrames(
            cachedVideo["lFrames"],
            filterParams,
            parameters,
        )
    except Exception:
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
        parameters,
    )
    aMeasurement = measureChannel(
        cachedVideo["avgAArr"],
        cachedVideo["timesScdsArr"],
        metricArr,
        releaseParams,
        parameters,
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


def formatMeasurement(value, uncertainty):
    if value < 0 or uncertainty < 0:
        return str(int(value if value < 0 else uncertainty))
    return f"{value:.3f}+-{uncertainty:.3f}"


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


def measurementFromRow(row):
    return {
        "video": row["video"],
        "g_crt90_10": float(row["g_crt90_10_s"]),
        "g_crt90_10_uncertainty": float(row["g_crt90_10_uncertainty_s"]),
        "g_pcrt": float(row["g_pcrt_s"]),
        "g_pcrt_uncertainty": float(row["g_pcrt_uncertainty_s"]),
        "a_crt90_10": float(row["a_crt90_10_s"]),
        "a_crt90_10_uncertainty": float(row["a_crt90_10_uncertainty_s"]),
        "a_pcrt": float(row["a_pcrt_s"]),
        "a_pcrt_uncertainty": float(row["a_pcrt_uncertainty_s"]),
    }


def inRange(value, minValue, maxValue):
    return minValue <= value <= maxValue


def relativeUncertainty(value, uncertainty):
    return uncertainty / value


def metricRanges(metric, parameters):
    kind = METRICS[metric]["kind"]
    if kind == "crt90_10":
        return (
            parameters["crt90_10_min"],
            parameters["crt90_10_max"],
            parameters["crt90_10_relative_uncertainty_min"],
            parameters["crt90_10_relative_uncertainty_max"],
        )
    if kind == "pcrt":
        return (
            parameters["pcrt_min"],
            parameters["pcrt_max"],
            parameters["pcrt_relative_uncertainty_min"],
            parameters["pcrt_relative_uncertainty_max"],
        )
    raise ValueError(f"Unsupported metric kind: {kind}")


def metricFailed(measurement, metric):
    value = measurement[metric]
    uncertainty = measurement[f"{metric}_uncertainty"]
    return not (
        np.isfinite(value)
        and value > 0
        and np.isfinite(uncertainty)
        and uncertainty >= 0
    )


def metricFiltered(measurement, metric, parameters):
    value = measurement[metric]
    uncertainty = measurement[f"{metric}_uncertainty"]
    valueMin, valueMax, relativeUncertaintyMin, relativeUncertaintyMax = metricRanges(
        metric,
        parameters,
    )
    return not (
        inRange(value, valueMin, valueMax)
        and inRange(
            relativeUncertainty(value, uncertainty),
            relativeUncertaintyMin,
            relativeUncertaintyMax,
        )
    )


def filterMeasurementsForPlot(measurements, xMetric, yMetric, parameters):
    kept = []
    failedX = 0
    failedY = 0
    filteredX = 0
    filteredY = 0

    for measurement in measurements:
        xFailed = metricFailed(measurement, xMetric)
        yFailed = metricFailed(measurement, yMetric)
        if xFailed:
            failedX += 1
        if yFailed:
            failedY += 1
        if xFailed or yFailed:
            continue

        xFiltered = metricFiltered(measurement, xMetric, parameters)
        yFiltered = metricFiltered(measurement, yMetric, parameters)
        if xFiltered:
            filteredX += 1
        if yFiltered:
            filteredY += 1
        if xFiltered or yFiltered:
            continue

        kept.append(measurement)

    totalCount = len(measurements)
    keptCount = len(kept)
    counts = {
        "total_count": totalCount,
        "kept_count": keptCount,
        "success_rate": keptCount / totalCount if totalCount else 0.0,
        "failed_x": failedX,
        "failed_y": failedY,
        "filtered_x": filteredX,
        "filtered_y": filteredY,
        "x_metric": xMetric,
        "y_metric": yMetric,
    }
    return kept, counts


def fitLineAndR2(xValues, yValues):
    if len(xValues) < 2:
        return None

    slope, intercept = np.polyfit(xValues, yValues, 1)
    fittedValues = slope * xValues + intercept
    ssResidual = np.sum((yValues - fittedValues) ** 2)
    ssTotal = np.sum((yValues - np.mean(yValues)) ** 2)
    r2 = np.nan if ssTotal == 0 else 1 - ssResidual / ssTotal
    return slope, intercept, r2


def plotMeasurements(ax, measurements, counts, xMetric, yMetric, title):
    xLabel = f"{METRICS[xMetric]['label']} (s)"
    yLabel = f"{METRICS[yMetric]['label']} (s)"
    xValues = np.array([measurement[xMetric] for measurement in measurements])
    yValues = np.array([measurement[yMetric] for measurement in measurements])
    xErrors = np.array(
        [measurement[f"{xMetric}_uncertainty"] for measurement in measurements]
    )
    yErrors = np.array(
        [measurement[f"{yMetric}_uncertainty"] for measurement in measurements]
    )

    fit = fitLineAndR2(xValues, yValues)
    if fit is None:
        r2Text = "R2=n/a"
    else:
        slope, intercept, r2 = fit
        xFit = np.linspace(np.min(xValues), np.max(xValues), 100)
        yFit = slope * xFit + intercept
        ax.plot(
            xFit,
            yFit,
            color="tab:red",
            lw=1.4,
            label=f"linear fit, R2={r2:.3f}",
        )
        r2Text = f"R2={r2:.3f}"

    if len(measurements):
        ax.errorbar(
            xValues,
            yValues,
            xerr=xErrors,
            yerr=yErrors,
            fmt=".",
            ms=6,
            color="tab:blue",
            ecolor="tab:gray",
            elinewidth=0.8,
            capsize=2,
            label=(
                f"videos={len(measurements)}, {r2Text}\n"
                f"failed x={counts['failed_x']}, filtered x={counts['filtered_x']}\n"
                f"failed y={counts['failed_y']}, filtered y={counts['filtered_y']}"
            ),
        )
    else:
        ax.plot(
            [],
            [],
            label=(
                "videos=0, R2=n/a\n"
                f"failed x={counts['failed_x']}, filtered x={counts['filtered_x']}\n"
                f"failed y={counts['failed_y']}, filtered y={counts['filtered_y']}"
            ),
        )

    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)
    ax.set_title(title)
    ax.legend(loc="best")


def plotComparison(measurements, plotStats):
    if plt is None:
        raise ImportError("matplotlib is required to show the CRT comparison plot.")

    plt.style.use("bmh")
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, plotSpec in zip(axes.flat, PLOT_SPECS):
        stats = plotStats[plotSpec["key"]]
        plotMeasurements(
            ax,
            stats["measurements"],
            stats,
            plotSpec["x_metric"],
            plotSpec["y_metric"],
            plotSpec["title"],
        )

    fig.suptitle("CRT pipeline comparison")
    fig.tight_layout()
    plt.show()


def summarizeMetricSuccess(measurements):
    metricStats = {}
    for metric in METRICS:
        successCount = sum(
            not metricFailed(measurement, metric) for measurement in measurements
        )
        totalCount = len(measurements)
        metricStats[metric] = {
            "success_count": successCount,
            "failure_count": totalCount - successCount,
            "success_rate": successCount / totalCount if totalCount else 0.0,
        }
    return metricStats


def summarizeFilteredMetricSuccess(measurements, parameters):
    metricStats = {}
    totalCount = len(measurements)
    for metric in METRICS:
        failedCount = 0
        filteredCount = 0
        successCount = 0
        for measurement in measurements:
            if metricFailed(measurement, metric):
                failedCount += 1
                continue
            if metricFiltered(measurement, metric, parameters):
                filteredCount += 1
                continue
            successCount += 1

        metricStats[metric] = {
            "success_count": successCount,
            "failure_count": failedCount,
            "filtered_count": filteredCount,
            "success_rate": successCount / totalCount if totalCount else 0.0,
        }
    return metricStats


def writeCsv(rows, csvPath):
    with Path(csvPath).open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELD_NAMES)
        writer.writeheader()
        writer.writerows(rows)


def analyzePlots(measurements, parameters):
    plotStats = {}
    for plotSpec in PLOT_SPECS:
        keptMeasurements, counts = filterMeasurementsForPlot(
            measurements,
            plotSpec["x_metric"],
            plotSpec["y_metric"],
            parameters,
        )
        xValues = np.array(
            [measurement[plotSpec["x_metric"]] for measurement in keptMeasurements]
        )
        yValues = np.array(
            [measurement[plotSpec["y_metric"]] for measurement in keptMeasurements]
        )
        fit = fitLineAndR2(xValues, yValues)
        counts.update(
            {
                "title": plotSpec["title"],
                "r2": None if fit is None else float(fit[2]),
                "slope": None if fit is None else float(fit[0]),
                "intercept": None if fit is None else float(fit[1]),
                "measurements": keptMeasurements,
            }
        )
        plotStats[plotSpec["key"]] = counts
    return plotStats


def run_crt_pipeline_analysis(run_on, parameters=None, show_plot=True):
    """
    Run the full CRT comparison pipeline on one or more videos.

    This function is the importable equivalent of running
    compare_crt_pipelines.py followed by plot_crt_pipeline_comparison.py. It
    iterates over video files, loads ROIs and cached arrays from Npz/Cache,
    performs release-frame detection from cached LAB L-channel ROI frames, uses
    derivative-based slicing to find the CRT interval, measures CRT90_10 and
    pCRT on both the BGR G and LAB A average intensity arrays, optionally writes
    a CSV file, optionally shows the 2x2 comparison plot, and returns the
    success/filtering/R2 statistics.

    Parameters
    ----------
    run_on : str, pathlib.Path, list, tuple, or set
        Video source(s) to analyze. A file path analyzes that single video. A
        directory path analyzes all files in that directory whose extension is
        listed in ``parameters["video_extensions"]``. Lists, tuples, and sets
        can contain any mixture of files, directories, or nested collections.

    parameters : dict or None, default None
        Optional parameter overrides. Public parameter names use snake_case.
        Missing keys use the defaults below.

        General I/O parameters:

        ``rois_path`` : str or pathlib.Path, default ``"rois_full.toml"``
            TOML file containing video ROIs.

        ``cache_dir`` : str or pathlib.Path, default ``Path("Npz/Cache")``
            Directory containing cached ``<video_stem>.npz`` files. Each cache
            must contain ``lFrames``, ``timesScdsArr``, ``avgAArr``, and
            ``avgGArr``.

        ``output_csv_path`` : str or pathlib.Path, default
        ``Path("crt_pipeline_comparison.csv")``
            CSV path written when ``write_csv`` is true.

        ``video_extensions`` : list[str], default
        ``[".MOV", ".wmv", ".mp4"]``
            Extensions accepted when a directory is supplied in ``run_on``.

        ``write_csv`` : bool, default ``True``
            Whether to write the per-video result CSV.

        ``verbose`` : bool, default ``True``
            Whether to print one progress line per video in the format
            ``measured <video_name>: <pCRT G>, <CRT90_10 G>, <pCRT A>,
            <CRT90_10 A>, <video_num>/<total_videos>``.

        Release-detection parameters:

        ``filter_type`` : {"canny", "laplacian"}, default ``"canny"``
            Edge metric used for release-frame detection. CRT measurements are
            still calculated on BGR G and LAB A regardless of this value.

        ``relax_max_grad`` : float, default ``1.0``
            Relaxed upper derivative bound used for CRT interval end selection,
            unless overridden inside ``canny_params`` or ``laplacian_params``.

        ``relax_min_grad`` : float, default ``-3.0``
            Relaxed lower derivative bound used for CRT interval end selection,
            unless overridden inside ``canny_params`` or ``laplacian_params``.

        ``strict_max_grad`` : float, default ``1``
            Strict upper derivative bound used for CRT interval start selection,
            unless overridden inside ``canny_params`` or ``laplacian_params``.

        ``strict_min_grad`` : float, default ``-0.5``
            Strict lower derivative bound used for CRT interval start selection,
            unless overridden inside ``canny_params`` or ``laplacian_params``.

        ``canny_params`` : dict, default
        ``{"thresh_1": 56, "thresh_2": 85, "blur_kernel": 6,
        "l2_grad": False, "canny_plateau": 0.1,
        "canny_smoothing_kernel": 9,
        "canny_gradient_smoothing_kernel": 1,
        "strict_min_grad": -0.5, "strict_max_grad": 1,
        "relax_min_grad": -3.0, "relax_max_grad": 1.0, "index_offset": 0}``
            Canny release-detection parameters. These are translated internally
            to the existing ``release_frame_processing`` camelCase keys.

        ``laplacian_params`` : dict, default
        ``{"ksize": 3, "blur_kernel": 7, "scale": 1, "delta": 0,
        "laplacian_plateau": 2.0, "laplacian_smoothing_kernel": 19,
        "laplacian_gradient_smoothing_kernel": 3,
        "strict_min_grad": -0.5, "strict_max_grad": 1,
        "relax_min_grad": -3.0, "relax_max_grad": 1.0, "index_offset": 0}``
            Laplacian release-detection parameters. ``scale`` and ``delta`` are
            included for completeness even though the current optimizer fixes
            them at ``1`` and ``0``.

        CRT90_10 calculation parameters:

        ``max_crt90_10_fit_seconds`` : float, default ``10.0``
            Maximum duration, starting at the CRT interval start, used by the
            Gaussian-smoothed CRT90_10 calculation.

        ``crt90_10_gaussian_sigma_seconds`` : float, default ``0.1``
            Gaussian smoothing sigma in seconds for CRT90_10.

        ``crt90_10_bootstrap_count`` : int, default ``500``
            Number of bootstrap traces used to estimate CRT90_10 uncertainty.

        ``crt90_10_bootstrap_seed`` : int, default ``0``
            Random seed used for CRT90_10 bootstrap uncertainty.

        Plot/filter parameters:

        ``crt90_10_min`` : float, default ``0.0``
            Minimum CRT90_10 value included in each plot and R2 calculation.

        ``crt90_10_max`` : float, default ``10.0``
            Maximum CRT90_10 value included in each plot and R2 calculation.

        ``pcrt_min`` : float, default ``0.0``
            Minimum pCRT value included in each plot and R2 calculation.

        ``pcrt_max`` : float, default ``10.0``
            Maximum pCRT value included in each plot and R2 calculation.

        ``crt90_10_relative_uncertainty_min`` : float, default ``0.0``
            Minimum allowed relative CRT90_10 uncertainty, calculated as
            ``crt90_10_uncertainty / crt90_10``.

        ``crt90_10_relative_uncertainty_max`` : float, default ``1.0``
            Maximum allowed relative CRT90_10 uncertainty, calculated as
            ``crt90_10_uncertainty / crt90_10``.

        ``pcrt_relative_uncertainty_min`` : float, default ``0.0``
            Minimum allowed relative pCRT uncertainty, calculated as
            ``pcrt_uncertainty / pcrt``.

        ``pcrt_relative_uncertainty_max`` : float, default ``1.0``
            Maximum allowed relative pCRT uncertainty, calculated as
            ``pcrt_uncertainty / pcrt``.

    show_plot : bool, default True
        Whether to show the 2x2 scatter-plot comparison figure. The plotted
        panels are BGR G pCRT vs BGR G CRT90_10, LAB A pCRT vs LAB A CRT90_10,
        LAB A CRT90_10 vs BGR G CRT90_10, and BGR G pCRT vs LAB A pCRT.

    Returns
    -------
    dict
        Dictionary with these keys:

        ``total_videos`` : int
            Number of video paths expanded from ``run_on``.

        ``output_csv_path`` : str
            CSV output path, even when ``write_csv`` is false.

        ``metric_success_rates`` : dict
            Per-metric raw measurement success rates before plot filtering.
            Keys are ``g_crt90_10``, ``g_pcrt``, ``a_crt90_10``, and
            ``a_pcrt``. Each value contains ``success_count``,
            ``failure_count``, and ``success_rate``.

        ``filtered_success_rates`` : dict
            Per-metric success rates after applying that metric's value and
            relative-uncertainty filters. Keys are ``g_crt90_10``, ``g_pcrt``,
            ``a_crt90_10``, and ``a_pcrt``. Each value contains
            ``success_count``, ``failure_count``, ``filtered_count``, and
            ``success_rate``.

        ``r2_values`` : dict
            Linear-fit coefficient of determination for each plot key. Values
            are floats or ``None`` when fewer than two videos remain after
            filtering.

        ``plots`` : dict
            Detailed per-plot statistics, including kept count, failed/filter
            counts for x and y metrics, slope, intercept, and R2.

        ``rows`` : list[dict]
            Per-video CSV rows. Measurement failures are encoded as negative
            error values: ``-1`` ROI failure, ``-2`` cache failure, ``-3`` edge
            metric failure, ``-4`` CRT interval failure, ``-5`` CRT90_10
            failure, ``-6`` CRT90_10 uncertainty failure, and ``-7`` pCRT
            failure.
    """
    parameters = buildParameters(parameters)
    videoPaths = list(iterRunOnVideoPaths(run_on, parameters["video_extensions"]))
    rows = []
    measurements = []
    totalVideos = len(videoPaths)

    for videoNum, videoPath in enumerate(videoPaths, start=1):
        roi, gMeasurement, aMeasurement = measureVideo(videoPath, parameters)
        row = rowForVideo(videoPath, roi, gMeasurement, aMeasurement)
        rows.append(row)
        measurements.append(measurementFromRow(row))

        if parameters["verbose"]:
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

    if parameters["write_csv"]:
        writeCsv(rows, parameters["output_csv_path"])

    plotStats = analyzePlots(measurements, parameters)
    if show_plot:
        plotComparison(measurements, plotStats)

    r2Values = {key: stats["r2"] for key, stats in plotStats.items()}
    return {
        "total_videos": totalVideos,
        "output_csv_path": str(parameters["output_csv_path"]),
        "metric_success_rates": summarizeMetricSuccess(measurements),
        "filtered_success_rates": summarizeFilteredMetricSuccess(
            measurements,
            parameters,
        ),
        "r2_values": r2Values,
        "plots": {
            key: {
                field: value
                for field, value in stats.items()
                if field != "measurements"
            }
            for key, stats in plotStats.items()
        },
        "rows": rows,
    }
