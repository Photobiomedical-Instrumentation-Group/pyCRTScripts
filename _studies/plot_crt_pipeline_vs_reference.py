import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-pycrt")

import matplotlib.pyplot as plt
import numpy as np

PIPELINE_CSV_PATH = Path("crt_pipeline_comparison.csv")
REFERENCE_CSV_PATH = Path("raquel_masters_roi_pcrt.csv")

REFERENCE_PCRT_MIN = 0.0
REFERENCE_PCRT_MAX = 10.0
REFERENCE_PCRT_RELATIVE_UNCERTAINTY_MIN = 0.0
REFERENCE_PCRT_RELATIVE_UNCERTAINTY_MAX = 1.0

CRT90_10_MIN = 0.0
CRT90_10_MAX = 10.0
PCRT_MIN = 0.0
PCRT_MAX = 10.0
CRT90_10_RELATIVE_UNCERTAINTY_MIN = 0.0
CRT90_10_RELATIVE_UNCERTAINTY_MAX = 1.0
PCRT_RELATIVE_UNCERTAINTY_MIN = 0.0
PCRT_RELATIVE_UNCERTAINTY_MAX = 1.0

plt.style.use("bmh")


METRICS = {
    "g_crt90_10": {
        "label": "BGR G CRT90_10",
        "kind": "crt90_10",
        "value_column": "g_crt90_10_s",
        "uncertainty_column": "g_crt90_10_uncertainty_s",
    },
    "g_pcrt": {
        "label": "BGR G pCRT",
        "kind": "pcrt",
        "value_column": "g_pcrt_s",
        "uncertainty_column": "g_pcrt_uncertainty_s",
    },
    "a_crt90_10": {
        "label": "LAB A CRT90_10",
        "kind": "crt90_10",
        "value_column": "a_crt90_10_s",
        "uncertainty_column": "a_crt90_10_uncertainty_s",
    },
    "a_pcrt": {
        "label": "LAB A pCRT",
        "kind": "pcrt",
        "value_column": "a_pcrt_s",
        "uncertainty_column": "a_pcrt_uncertainty_s",
    },
}

PLOT_SPECS = [
    {
        "title": "BGR G CRT90_10 vs reference pCRT",
        "metric": "g_crt90_10",
    },
    {
        "title": "BGR G pCRT vs reference pCRT",
        "metric": "g_pcrt",
    },
    {
        "title": "LAB A CRT90_10 vs reference pCRT",
        "metric": "a_crt90_10",
    },
    {
        "title": "LAB A pCRT vs reference pCRT",
        "metric": "a_pcrt",
    },
]


def parseFloat(row, key):
    value = row.get(key, "")
    if value == "":
        return np.nan
    return float(value)


def loadCsvRows(csvPath):
    with csvPath.open(newline="") as file:
        return list(csv.DictReader(file))


def inRange(value, minValue, maxValue):
    return minValue <= value <= maxValue


def relativeUncertainty(value, uncertainty):
    return uncertainty / value


def isValidValueWithUncertainty(value, uncertainty):
    return (
        np.isfinite(value)
        and value > 0
        and np.isfinite(uncertainty)
        and uncertainty >= 0
    )


def metricRanges(metric):
    kind = METRICS[metric]["kind"]
    if kind == "crt90_10":
        return (
            CRT90_10_MIN,
            CRT90_10_MAX,
            CRT90_10_RELATIVE_UNCERTAINTY_MIN,
            CRT90_10_RELATIVE_UNCERTAINTY_MAX,
        )
    if kind == "pcrt":
        return (
            PCRT_MIN,
            PCRT_MAX,
            PCRT_RELATIVE_UNCERTAINTY_MIN,
            PCRT_RELATIVE_UNCERTAINTY_MAX,
        )
    raise ValueError(f"Unsupported metric kind: {kind}")


def referenceFiltered(reference):
    value = reference["reference_pcrt"]
    uncertainty = reference["reference_pcrt_uncertainty"]
    return not (
        inRange(value, REFERENCE_PCRT_MIN, REFERENCE_PCRT_MAX)
        and inRange(
            relativeUncertainty(value, uncertainty),
            REFERENCE_PCRT_RELATIVE_UNCERTAINTY_MIN,
            REFERENCE_PCRT_RELATIVE_UNCERTAINTY_MAX,
        )
    )


def metricFiltered(measurement, metric):
    value = measurement[metric]
    uncertainty = measurement[f"{metric}_uncertainty"]
    valueMin, valueMax, relativeUncertaintyMin, relativeUncertaintyMax = metricRanges(
        metric
    )
    return not (
        inRange(value, valueMin, valueMax)
        and inRange(
            relativeUncertainty(value, uncertainty),
            relativeUncertaintyMin,
            relativeUncertaintyMax,
        )
    )


def loadReferenceMeasurements(csvPath):
    references = {}
    for row in loadCsvRows(csvPath):
        videoName = row["video_file_name"]
        references[videoName] = {
            "reference_pcrt": parseFloat(row, "pcrt"),
            "reference_pcrt_uncertainty": parseFloat(row, "uncertainty"),
        }
    return references


def loadPipelineMeasurements(csvPath):
    measurements = []
    for row in loadCsvRows(csvPath):
        measurement = {"video": row["video"]}
        for metric, metricInfo in METRICS.items():
            measurement[metric] = parseFloat(row, metricInfo["value_column"])
            measurement[f"{metric}_uncertainty"] = parseFloat(
                row,
                metricInfo["uncertainty_column"],
            )
        measurements.append(measurement)
    return measurements


def joinedMeasurements(pipelineMeasurements, referenceMeasurements):
    joined = []
    for measurement in pipelineMeasurements:
        reference = referenceMeasurements.get(measurement["video"])
        if reference is None:
            joined.append({**measurement, "missing_reference": True})
            continue
        joined.append({**measurement, **reference, "missing_reference": False})
    return joined


def filterMeasurementsForMetric(measurements, metric):
    kept = []
    counts = {
        "missing_reference": 0,
        "failed_reference": 0,
        "filtered_reference": 0,
        "failed_metric": 0,
        "filtered_metric": 0,
    }

    for measurement in measurements:
        if measurement["missing_reference"]:
            counts["missing_reference"] += 1
            continue

        referenceValue = measurement["reference_pcrt"]
        referenceUncertainty = measurement["reference_pcrt_uncertainty"]
        metricValue = measurement[metric]
        metricUncertainty = measurement[f"{metric}_uncertainty"]

        if not isValidValueWithUncertainty(referenceValue, referenceUncertainty):
            counts["failed_reference"] += 1
            continue
        if referenceFiltered(measurement):
            counts["filtered_reference"] += 1
            continue
        if not isValidValueWithUncertainty(metricValue, metricUncertainty):
            counts["failed_metric"] += 1
            continue
        if metricFiltered(measurement, metric):
            counts["filtered_metric"] += 1
            continue

        kept.append(measurement)

    counts["kept"] = len(kept)
    counts["total"] = len(measurements)
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


def plotMetric(ax, measurements, counts, metric, title):
    metricLabel = METRICS[metric]["label"]
    xValues = np.array(
        [measurement["reference_pcrt"] for measurement in measurements]
    )
    yValues = np.array([measurement[metric] for measurement in measurements])
    xErrors = np.array(
        [measurement["reference_pcrt_uncertainty"] for measurement in measurements]
    )
    yErrors = np.array(
        [measurement[f"{metric}_uncertainty"] for measurement in measurements]
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

    label = (
        f"videos={counts['kept']}, {r2Text}\n"
        f"missing ref={counts['missing_reference']}, "
        f"failed ref={counts['failed_reference']}, "
        f"filtered ref={counts['filtered_reference']}\n"
        f"failed metric={counts['failed_metric']}, "
        f"filtered metric={counts['filtered_metric']}"
    )
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
            label=label,
        )
    else:
        ax.plot([], [], label=label)

    ax.set_xlabel("Reference pCRT (s)")
    ax.set_ylabel(f"{metricLabel} (s)")
    ax.set_title(title)
    ax.legend(loc="best")


def main():
    pipelineMeasurements = loadPipelineMeasurements(PIPELINE_CSV_PATH)
    referenceMeasurements = loadReferenceMeasurements(REFERENCE_CSV_PATH)
    measurements = joinedMeasurements(pipelineMeasurements, referenceMeasurements)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, plotSpec in zip(axes.flat, PLOT_SPECS):
        keptMeasurements, counts = filterMeasurementsForMetric(
            measurements,
            plotSpec["metric"],
        )
        plotMetric(
            ax,
            keptMeasurements,
            counts,
            plotSpec["metric"],
            plotSpec["title"],
        )

    fig.suptitle(f"{PIPELINE_CSV_PATH.name} vs {REFERENCE_CSV_PATH.name}")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
