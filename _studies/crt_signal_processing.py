import numpy as np
from funcs import ensmoothenSignal
from pyCRT.curveFitting import (
    calcPCRTFirstPositivePeak,
    exponential,
    pCRTFromParameters,
)
from release_frame_processing import (
    DEFAULT_AVG_A_RELAX_MAX_GRAD,
    DEFAULT_AVG_A_RELAX_MIN_GRAD,
    DEFAULT_AVG_A_STRICT_MAX_GRAD,
    DEFAULT_AVG_A_STRICT_MIN_GRAD,
    calcGradientWithFilteredSamples,
    minMaxNormalizeOrZeros,
)

DEFAULT_MAX_CRT90_10_FIT_SECONDS = 10.0
DEFAULT_CRT90_10_GAUSSIAN_SIGMA_SECONDS = 0.1
DEFAULT_CRT90_10_BOOTSTRAP_COUNT = 500
DEFAULT_CRT90_10_BOOTSTRAP_SEED = 0


def calcSignalGradientDiagnostics(
    signalArr,
    timeArr,
    crtIntervalStartIndex,
    crtIntervalEndIndex,
    params,
):
    if crtIntervalEndIndex <= crtIntervalStartIndex + 1:
        raise ValueError(
            "Invalid signal-gradient diagnostic interval: "
            f"crtIntervalStartIndex={crtIntervalStartIndex}, "
            f"crtIntervalEndIndex={crtIntervalEndIndex}."
        )

    signalNorm = minMaxNormalizeOrZeros(signalArr)
    fullSignalGrad, fullSignalSmooth, fullTimes, _ = calcGradientWithFilteredSamples(
        signalNorm,
        timeArr,
    )
    intervalTimes = timeArr[crtIntervalStartIndex:crtIntervalEndIndex]
    intervalSignal = signalNorm[crtIntervalStartIndex:crtIntervalEndIndex]
    intervalGrad, intervalSignalSmooth, intervalTimes, _ = (
        calcGradientWithFilteredSamples(
            intervalSignal,
            intervalTimes,
        )
    )

    return {
        "times": intervalTimes,
        "signalInterval": intervalSignalSmooth,
        "signalGrad": intervalGrad,
        "fullTimes": fullTimes,
        "fullSignalSmooth": fullSignalSmooth,
        "fullSignalGrad": fullSignalGrad,
        "strictMinGrad": params.get("strictMinGrad", DEFAULT_AVG_A_STRICT_MIN_GRAD),
        "strictMaxGrad": params.get("strictMaxGrad", DEFAULT_AVG_A_STRICT_MAX_GRAD),
        "relaxMinGrad": params.get("relaxMinGrad", DEFAULT_AVG_A_RELAX_MIN_GRAD),
        "relaxMaxGrad": params.get("relaxMaxGrad", DEFAULT_AVG_A_RELAX_MAX_GRAD),
        "crtIntervalStartIndex": crtIntervalStartIndex,
        "crtIntervalEndIndex": crtIntervalEndIndex,
    }


def calcCRT90_10Gaussian(
    timeArr,
    signalArr,
    indices,
    maxFitSeconds=DEFAULT_MAX_CRT90_10_FIT_SECONDS,
    gaussianSigmaSeconds=DEFAULT_CRT90_10_GAUSSIAN_SIGMA_SECONDS,
):
    crtIntervalStartIndex, _, crtIntervalEndIndex = indices
    fullIntervalTimes = timeArr[crtIntervalStartIndex:crtIntervalEndIndex]
    if len(fullIntervalTimes) < 2:
        raise ValueError("Need at least two points to calculate Gaussian CRT90_10.")

    fitEndTime = min(
        fullIntervalTimes[0] + maxFitSeconds,
        fullIntervalTimes[-1],
    )
    fitEndOffset = int(np.searchsorted(fullIntervalTimes, fitEndTime, side="right"))
    fitEndOffset = max(fitEndOffset, 2)
    intervalTimes = fullIntervalTimes[:fitEndOffset]
    intervalValues = signalArr[
        crtIntervalStartIndex : crtIntervalStartIndex + fitEndOffset
    ]

    timeDiffs = np.diff(intervalTimes)
    positiveTimeDiffs = timeDiffs[timeDiffs > 0]
    if len(positiveTimeDiffs) == 0:
        raise ValueError("Need increasing timestamps to calculate Gaussian CRT90_10.")

    medianFrameTime = float(np.median(positiveTimeDiffs))
    sigmaSamples = gaussianSigmaSeconds / medianFrameTime
    if sigmaSamples > 0:
        smoothedValues = ensmoothenSignal(
            intervalValues,
            method="gaussian",
            sigma=sigmaSamples,
        )
    else:
        smoothedValues = intervalValues
    smoothedValues = minMaxNormalizeOrZeros(smoothedValues)

    startOffset = int(np.argmax(smoothedValues))
    startTime = intervalTimes[startOffset]
    startIndex = crtIntervalStartIndex + startOffset
    startValue = smoothedValues[startOffset]
    value90 = 0.9 * startValue
    value10 = 0.1 * startValue

    ninetyPercentCandidates = np.flatnonzero(smoothedValues[startOffset:] < value90)
    if len(ninetyPercentCandidates) == 0:
        raise ValueError(
            "No Gaussian-smoothed signal point fell below the 90% value."
        )
    time90Offset = startOffset + int(ninetyPercentCandidates[0])

    tenPercentCandidates = np.flatnonzero(smoothedValues[time90Offset:] < value10)
    if len(tenPercentCandidates) == 0:
        raise ValueError(
            "No Gaussian-smoothed signal point fell below the 10% value."
        )
    time10Offset = time90Offset + int(tenPercentCandidates[0])

    time90 = intervalTimes[time90Offset]
    time10 = intervalTimes[time10Offset]

    return {
        "crt90_10": time10 - time90,
        "startTime": startTime,
        "startIndex": startIndex,
        "time90": time90,
        "time10": time10,
        "times": intervalTimes,
        "values": smoothedValues,
        "maxFitSeconds": maxFitSeconds,
        "gaussianSigmaSeconds": gaussianSigmaSeconds,
    }


def calcCRT90_10BootstrapUncertainty(
    timeArr,
    signalArr,
    indices,
    maxFitSeconds=DEFAULT_MAX_CRT90_10_FIT_SECONDS,
    gaussianSigmaSeconds=DEFAULT_CRT90_10_GAUSSIAN_SIGMA_SECONDS,
    nBootstraps=DEFAULT_CRT90_10_BOOTSTRAP_COUNT,
    randomSeed=DEFAULT_CRT90_10_BOOTSTRAP_SEED,
):
    crtIntervalStartIndex, _, crtIntervalEndIndex = indices
    fullIntervalTimes = timeArr[crtIntervalStartIndex:crtIntervalEndIndex]
    if len(fullIntervalTimes) < 2:
        raise ValueError("Need at least two points to estimate CRT90_10 uncertainty.")

    fitEndTime = min(
        fullIntervalTimes[0] + maxFitSeconds,
        fullIntervalTimes[-1],
    )
    fitEndOffset = int(np.searchsorted(fullIntervalTimes, fitEndTime, side="right"))
    fitEndOffset = max(fitEndOffset, 2)
    intervalTimes = fullIntervalTimes[:fitEndOffset]
    intervalValues = np.asarray(
        signalArr[crtIntervalStartIndex : crtIntervalStartIndex + fitEndOffset],
        dtype=float,
    )

    timeDiffs = np.diff(intervalTimes)
    positiveTimeDiffs = timeDiffs[timeDiffs > 0]
    if len(positiveTimeDiffs) == 0:
        raise ValueError("Need increasing timestamps to estimate CRT90_10 uncertainty.")

    medianFrameTime = float(np.median(positiveTimeDiffs))
    sigmaSamples = gaussianSigmaSeconds / medianFrameTime
    if sigmaSamples > 0:
        smoothedValues = ensmoothenSignal(
            intervalValues,
            method="gaussian",
            sigma=sigmaSamples,
        )
    else:
        smoothedValues = intervalValues

    residuals = intervalValues - smoothedValues
    noiseSigma = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 0.0

    rng = np.random.default_rng(randomSeed)
    signalArr = np.asarray(signalArr, dtype=float)
    samples = []
    failureCount = 0
    for _ in range(nBootstraps):
        syntheticSignal = signalArr.copy()
        syntheticSignal[
            crtIntervalStartIndex : crtIntervalStartIndex + fitEndOffset
        ] = intervalValues + rng.normal(0.0, noiseSigma, size=len(intervalValues))
        try:
            sampleResult = calcCRT90_10Gaussian(
                timeArr,
                syntheticSignal,
                indices,
                maxFitSeconds=maxFitSeconds,
                gaussianSigmaSeconds=gaussianSigmaSeconds,
            )
        except Exception:
            failureCount += 1
            continue

        samples.append(sampleResult["crt90_10"])

    if len(samples) == 0:
        raise ValueError("All CRT90_10 bootstrap iterations failed.")

    samples = np.asarray(samples)
    ci95 = np.percentile(samples, [2.5, 97.5])
    return {
        "crt90_10_mean": float(np.mean(samples)),
        "crt90_10_std": float(np.std(samples, ddof=1)) if len(samples) > 1 else 0.0,
        "crt90_10_ci95": (float(ci95[0]), float(ci95[1])),
        "noiseSigma": noiseSigma,
        "successCount": int(len(samples)),
        "failureCount": int(failureCount),
        "samples": samples,
    }


def calcPCRTFit(timeArr, signalArr, indices):
    crtIntervalStartIndex, startIndex, crtIntervalEndIndex = indices
    fitTimes = timeArr[startIndex:crtIntervalEndIndex] - timeArr[startIndex]
    fitValues = signalArr[startIndex:crtIntervalEndIndex]
    referenceValues = signalArr[crtIntervalStartIndex:crtIntervalEndIndex]

    if len(fitTimes) < 4 or referenceValues.max() == referenceValues.min():
        return None

    referenceMin = referenceValues.min()
    referenceRange = referenceValues.max() - referenceMin
    fitValues = (fitValues - referenceMin) / referenceRange

    try:
        pcrtTuple, criticalTime = calcPCRTFirstPositivePeak(fitTimes, fitValues)
    except Exception as err:
        print(f"pCRT fit failed: {err}")
        return None

    pcrt, pcrtUncertainty = pCRTFromParameters(pcrtTuple)
    pcrtParams, _ = pcrtTuple
    pcrtValues = exponential(fitTimes, *pcrtParams) * referenceRange + referenceMin

    return {
        "pcrt": pcrt,
        "pcrtUncertainty": pcrtUncertainty,
        "criticalTime": criticalTime + timeArr[startIndex],
        "fitTimes": fitTimes + timeArr[startIndex],
        "fitValues": pcrtValues,
    }
