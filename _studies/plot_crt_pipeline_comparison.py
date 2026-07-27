import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

INPUT_CSV_PATH = Path("crt_pipeline_comparison.csv")

CRT90_10_MIN = 0.0
CRT90_10_MAX = 10.0
PCRT_MIN = 0.0
PCRT_MAX = 10.0
CRT90_10_RELATIVE_UNCERTAINTY_MIN = 0.0
CRT90_10_RELATIVE_UNCERTAINTY_MAX = 1.0
PCRT_RELATIVE_UNCERTAINTY_MIN = 0.0
PCRT_RELATIVE_UNCERTAINTY_MAX = 1.0

plt.style.use("bmh")


def parseFloat(row, key):
    value = row.get(key, "")
    if value == "":
        return np.nan
    return float(value)


def loadRows(csvPath):
    with csvPath.open(newline="") as file:
        return list(csv.DictReader(file))


def inRange(value, minValue, maxValue):
    return minValue <= value <= maxValue


def relativeUncertainty(value, uncertainty):
    return uncertainty / value


def rowToMeasurement(row):
    return {
        "video": row["video"],
        "g_crt90_10": parseFloat(row, "g_crt90_10_s"),
        "g_crt90_10_uncertainty": parseFloat(row, "g_crt90_10_uncertainty_s"),
        "g_pcrt": parseFloat(row, "g_pcrt_s"),
        "g_pcrt_uncertainty": parseFloat(row, "g_pcrt_uncertainty_s"),
        "a_crt90_10": parseFloat(row, "a_crt90_10_s"),
        "a_crt90_10_uncertainty": parseFloat(row, "a_crt90_10_uncertainty_s"),
        "a_pcrt": parseFloat(row, "a_pcrt_s"),
        "a_pcrt_uncertainty": parseFloat(row, "a_pcrt_uncertainty_s"),
    }


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
        "title": "BGR G",
        "xMetric": "g_crt90_10",
        "yMetric": "g_pcrt",
    },
    {
        "title": "LAB A",
        "xMetric": "a_crt90_10",
        "yMetric": "a_pcrt",
    },
    {
        "title": "CRT90_10 LAB A vs BGR G",
        "xMetric": "g_crt90_10",
        "yMetric": "a_crt90_10",
    },
    {
        "title": "pCRT BGR G vs LAB A",
        "xMetric": "a_pcrt",
        "yMetric": "g_pcrt",
    },
]


def isFinitePositiveValue(value):
    return np.isfinite(value) and value > 0


def isFiniteNonnegativeUncertainty(value):
    return np.isfinite(value) and value >= 0


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


def metricFailed(measurement, metric):
    value = measurement[metric]
    uncertainty = measurement[f"{metric}_uncertainty"]
    return not (
        isFinitePositiveValue(value)
        and isFiniteNonnegativeUncertainty(uncertainty)
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


def filterMeasurementsForPlot(measurements, xMetric, yMetric):
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

        xFiltered = metricFiltered(measurement, xMetric)
        yFiltered = metricFiltered(measurement, yMetric)
        if xFiltered:
            filteredX += 1
        if yFiltered:
            filteredY += 1
        if xFiltered or yFiltered:
            continue

        kept.append(measurement)

    counts = {
        "failed_x": failedX,
        "failed_y": failedY,
        "filtered_x": filteredX,
        "filtered_y": filteredY,
    }
    return kept, counts


def fitLineAndR2(xValues, yValues):
    if len(xValues) < 2:
        return None

    slope, intercept = np.polyfit(xValues, yValues, 1)
    fittedValues = slope * xValues + intercept
    ssResidual = np.sum((yValues - fittedValues) ** 2)
    ssTotal = np.sum((yValues - np.mean(yValues)) ** 2)
    if ssTotal == 0:
        r2 = np.nan
    else:
        r2 = 1 - ssResidual / ssTotal

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


def main():
    rows = loadRows(INPUT_CSV_PATH)
    measurements = [rowToMeasurement(row) for row in rows]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, plotSpec in zip(axes.flat, PLOT_SPECS):
        keptMeasurements, counts = filterMeasurementsForPlot(
            measurements,
            plotSpec["xMetric"],
            plotSpec["yMetric"],
        )
        plotMeasurements(
            ax,
            keptMeasurements,
            counts,
            plotSpec["xMetric"],
            plotSpec["yMetric"],
            plotSpec["title"],
        )

    fig.suptitle(INPUT_CSV_PATH.name)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
