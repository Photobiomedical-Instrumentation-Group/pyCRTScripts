import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-pycrt")

import cv2 as cv
import numpy as np
from funcs import (
    applyMedian,
    cannyEdgeDetector,
    ensmoothenSignal,
    laplacianEdgeDetector,
    minMaxNormalize,
)
from pyCRT.frameOperations import cropFrame, rescaleFrame

DEFAULT_AVG_A_RELAX_MAX_GRAD = 0.2
DEFAULT_AVG_A_RELAX_MIN_GRAD = -DEFAULT_AVG_A_RELAX_MAX_GRAD
DEFAULT_AVG_A_STRICT_MAX_GRAD = DEFAULT_AVG_A_RELAX_MAX_GRAD
DEFAULT_AVG_A_STRICT_MIN_GRAD = DEFAULT_AVG_A_RELAX_MIN_GRAD
DEFAULT_GRADIENT_SAVGOL_WINDOW_SECONDS = 0.1
DEFAULT_GRADIENT_SAVGOL_POLYORDER = 3
DEFAULT_CANNY_GRADIENT_SMOOTHING_KERNEL = 1


def frameToLabRoi(frame, roi, medianKernelRadius=1, rescaleFactor=0.5):
    medianFrame = applyMedian(frame, medianKernelRadius)
    float32Frame = medianFrame.astype(np.float32) / 255
    labFrame = cv.cvtColor(float32Frame, cv.COLOR_BGR2LAB)
    rescaledLabFrame = rescaleFrame(labFrame, rescaleFactor)
    return cropFrame(rescaledLabFrame, roi)


def labRoiToLFrame(labRoi):
    return np.uint8(np.round(labRoi[..., 0] * 255 / 100))


def frameToLChannelRoi(frame, roi, medianKernelRadius=1, rescaleFactor=0.5):
    return labRoiToLFrame(
        frameToLabRoi(
            frame,
            roi,
            medianKernelRadius=medianKernelRadius,
            rescaleFactor=rescaleFactor,
        )
    )


def frameToCachedMetrics(frame, roi, medianKernelRadius=1, rescaleFactor=0.5):
    medianFrame = applyMedian(frame, medianKernelRadius)
    rescaledRgbFrame = rescaleFrame(medianFrame, rescaleFactor)
    croppedRgbFrame = cropFrame(rescaledRgbFrame, roi)

    float32Frame = medianFrame.astype(np.float32) / 255
    labFrame = cv.cvtColor(float32Frame, cv.COLOR_BGR2LAB)
    rescaledLabFrame = rescaleFrame(labFrame, rescaleFactor)
    croppedLabFrame = cropFrame(rescaledLabFrame, roi)

    lFrame = labRoiToLFrame(croppedLabFrame)
    avgA = np.mean(croppedLabFrame[..., 1])
    avgG = np.mean(croppedRgbFrame[..., 1])
    return lFrame, avgA, avgG


def cannySumFromLFrame(lFrame, params):
    cannyFrame = cannyEdgeDetector(
        lFrame,
        params["thresh1"],
        params["thresh2"],
        params["blurKernel"],
        params["l2grad"],
    )
    return np.sum(cannyFrame.astype(np.float32))


def normalizeCannyArr(cannyArr):
    cannyArr = np.array(cannyArr, dtype=float)
    if cannyArr.max() == cannyArr.min():
        return np.zeros_like(cannyArr)
    return 100 * minMaxNormalize(cannyArr)


def cannySumsFromLFrames(lFrames, params):
    cannySums = [cannySumFromLFrame(lFrame, params) for lFrame in lFrames]
    return normalizeCannyArr(cannySums)


def laplacianSumFromLFrame(lFrame, params):
    laplacianFrame = laplacianEdgeDetector(
        lFrame,
        params["ksize"],
        params["blurKernel"],
        params["scale"],
        params["delta"],
    )
    return np.sum(laplacianFrame.astype(np.float64) ** 2)


def laplacianSumsFromLFrames(lFrames, params):
    laplacianSums = [laplacianSumFromLFrame(lFrame, params) for lFrame in lFrames]
    return normalizeCannyArr(laplacianSums)


def releaseParamsFromLaplacianParams(params):
    return {
        **params,
        "cannyPlateau": params["laplacianPlateau"],
        "cannySmoothingKernel": params["laplacianSmoothingKernel"],
        "cannyGradientSmoothingKernel": params.get(
            "laplacianGradientSmoothingKernel",
            DEFAULT_CANNY_GRADIENT_SMOOTHING_KERNEL,
        ),
    }


def smoothCannyArr(cannyArr, params):
    cannyArrSmooth = ensmoothenSignal(
        cannyArr,
        method="median",
        kernelSize=params["cannySmoothingKernel"],
    )
    return normalizeCannyArr(cannyArrSmooth)


def removeNonIncreasingSamples(signalArr, timeArr):
    signalArr = np.asarray(signalArr)
    timeArr = np.asarray(timeArr)
    if len(signalArr) != len(timeArr):
        raise ValueError(
            f"signalArr and timeArr must have the same length: "
            f"{len(signalArr)=}, {len(timeArr)=}."
        )
    if len(signalArr) == 0:
        return signalArr, timeArr, np.array([], dtype=int)

    keepIndices = [0]
    lastTime = timeArr[0]
    for index in range(1, len(timeArr)):
        if timeArr[index] > lastTime:
            keepIndices.append(index)
            lastTime = timeArr[index]

    keepIndices = np.array(keepIndices, dtype=int)
    droppedCount = len(timeArr) - len(keepIndices)
    if droppedCount:
        print(
            f"Warning: removed {droppedCount} sample(s) with non-increasing "
            "timestamps before calculating gradient."
        )

    return signalArr[keepIndices], timeArr[keepIndices], keepIndices


def calcGradientWithFilteredSamples(
    signalArr,
    timeArr,
    smoothingWindowSeconds=DEFAULT_GRADIENT_SAVGOL_WINDOW_SECONDS,
    savgolPolyorder=DEFAULT_GRADIENT_SAVGOL_POLYORDER,
):
    signalArr, timeArr, sourceIndices = removeNonIncreasingSamples(signalArr, timeArr)
    if len(signalArr) < 2:
        raise ValueError("Need at least two points to calculate the Canny gradient.")

    if smoothingWindowSeconds > 0:
        medianFrameTime = float(np.median(np.diff(timeArr)))
        windowSamples = smoothingWindowSeconds / medianFrameTime
        windowLength = round(windowSamples)
        if windowLength % 2 == 0:
            windowLength += 1
        minimumWindowLength = savgolPolyorder + 2
        if minimumWindowLength % 2 == 0:
            minimumWindowLength += 1
        windowLength = max(windowLength, minimumWindowLength)

        if windowLength <= len(signalArr):
            signalArr = ensmoothenSignal(
                signalArr,
                method="savitzky-golay",
                windowLength=windowLength,
                polyorder=savgolPolyorder,
            )

    return np.gradient(signalArr, timeArr), signalArr, timeArr, sourceIndices


def calcGradient(
    signalArr,
    timeArr,
    smoothingWindowSeconds=DEFAULT_GRADIENT_SAVGOL_WINDOW_SECONDS,
    savgolPolyorder=DEFAULT_GRADIENT_SAVGOL_POLYORDER,
):
    gradient, _, _, _ = calcGradientWithFilteredSamples(
        signalArr,
        timeArr,
        smoothingWindowSeconds=smoothingWindowSeconds,
        savgolPolyorder=savgolPolyorder,
    )
    return gradient


def smoothCannyGradientArr(cannyArrGrad, params):
    kernelSize = int(
        params.get(
            "cannyGradientSmoothingKernel",
            DEFAULT_CANNY_GRADIENT_SMOOTHING_KERNEL,
        )
    )
    if kernelSize <= 1:
        return np.asarray(cannyArrGrad)

    return ensmoothenSignal(
        cannyArrGrad,
        method="median",
        kernelSize=kernelSize,
    )


def findReleaseIndex(cannyArr, timeArr, params):
    cannyArr = np.asarray(cannyArr)
    filteredCannyArr, filteredTimeArr, sourceIndices = removeNonIncreasingSamples(
        cannyArr,
        timeArr,
    )
    if len(filteredCannyArr) < 2:
        raise ValueError("Need at least two valid timestamped Canny samples.")

    cannyArrSmooth = smoothCannyArr(filteredCannyArr, params)
    cannyArrGrad = calcGradient(
        cannyArrSmooth,
        filteredTimeArr,
        smoothingWindowSeconds=0,
    )
    cannyArrGrad = smoothCannyGradientArr(cannyArrGrad, params)
    minGradOffset = int(np.argmin(cannyArrGrad))
    maxGradOffset = int(np.argmax(cannyArrGrad))
    if minGradOffset < maxGradOffset:
        return findReleaseIndex(cannyArr.max() - cannyArr, timeArr, params)

    releaseCandidates = np.argwhere(
        (np.arange(len(cannyArrSmooth)) >= minGradOffset)
        & (np.abs(cannyArrGrad) <= params["cannyPlateau"])
    )
    if len(releaseCandidates) == 0:
        raise ValueError("No release candidate satisfied the Canny plateau condition.")

    releaseIndex = sourceIndices[releaseCandidates[0, 0]] + params["indexOffset"]
    return int(np.clip(releaseIndex, 0, len(cannyArr) - 1))


def minMaxNormalizeOrZeros(array):
    array = np.array(array, dtype=float)
    if array.max() == array.min():
        return np.zeros_like(array)
    return minMaxNormalize(array)


def findLongestTrueRun(mask):
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        raise ValueError(
            "No A-channel gradient interval satisfied relaxMinGrad..relaxMaxGrad."
        )

    padded = np.concatenate(([False], mask, [False]))
    changes = np.diff(padded.astype(int))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    lengths = ends - starts
    bestRunIndex = int(np.argmax(lengths))
    return int(starts[bestRunIndex]), int(ends[bestRunIndex])


def findFirstStrictStartAndRelaxedEnd(relaxedMask, strictStartMask):
    relaxedMask = np.asarray(relaxedMask, dtype=bool)
    strictStartMask = np.asarray(strictStartMask, dtype=bool)
    if relaxedMask.shape != strictStartMask.shape:
        raise ValueError(
            "relaxedMask and strictStartMask must have the same shape: "
            f"{relaxedMask.shape=} {strictStartMask.shape=}."
        )

    strictOffsets = np.flatnonzero(strictStartMask)
    if len(strictOffsets) == 0:
        raise ValueError(
            "No A-channel gradient point satisfied the strict start range."
        )

    startOffset = int(strictOffsets[0])
    relaxedFailures = np.flatnonzero(~relaxedMask[startOffset:])
    if len(relaxedFailures) == 0:
        endOffset = len(relaxedMask)
    else:
        endOffset = startOffset + int(relaxedFailures[0])

    return startOffset, endOffset


def optimizableFindCrtIntervalIndices(
    intensArr,
    timeArr,
    releaseIndex,
    strictMinGrad,
    strictMaxGrad,
    relaxMinGrad,
    relaxMaxGrad,
    offsetTime=0.0,
):
    # {{{
    assert strictMinGrad < strictMaxGrad
    assert relaxMinGrad < relaxMaxGrad
    assert relaxMinGrad <= strictMinGrad
    assert relaxMaxGrad >= strictMaxGrad

    releaseTime = timeArr[releaseIndex]
    searchEndTime = min(releaseTime + 30, timeArr[-1])
    searchEndIndex = int(np.argmin(np.abs(timeArr - searchEndTime)))

    assert searchEndIndex > releaseIndex + 1

    intensArr = minMaxNormalizeOrZeros(intensArr)
    intensArrSearchInterval = intensArr[releaseIndex:searchEndIndex]

    intensGrad, _, _, sourceOffsets = calcGradientWithFilteredSamples(
        intensArrSearchInterval,
        timeArr[releaseIndex:searchEndIndex],
    )
    strictStartMask = (strictMinGrad <= intensGrad) & (intensGrad <= strictMaxGrad)
    relaxMask = (relaxMinGrad <= intensGrad) & (intensGrad <= relaxMaxGrad)
    crtIntervalStartOffset, crtIntervalEndOffset = findFirstStrictStartAndRelaxedEnd(
        relaxMask,
        strictStartMask,
    )
    assert crtIntervalEndOffset > crtIntervalStartOffset

    crtIntervalStartOffset = int(sourceOffsets[crtIntervalStartOffset])
    if crtIntervalEndOffset == len(sourceOffsets):
        crtIntervalEndOffset = int(sourceOffsets[-1]) + 1
    else:
        crtIntervalEndOffset = int(sourceOffsets[crtIntervalEndOffset])

    intervalTimes = timeArr[
        releaseIndex + crtIntervalStartOffset : releaseIndex + crtIntervalEndOffset
    ]
    endTimeGaussian = min(intervalTimes[0] + 1, intervalTimes[-1])
    endIndexGaussian = (
        crtIntervalStartOffset
        + int(np.argmin(np.abs(intervalTimes - endTimeGaussian)))
        + 1
    )
    endIndexGaussian = min(
        max(endIndexGaussian, crtIntervalStartOffset + 1),
        crtIntervalEndOffset,
    )
    intensStartIntervalSmooth = ensmoothenSignal(
        intensArrSearchInterval[crtIntervalStartOffset:endIndexGaussian],
        method="gaussian",
        sigma=3,
    )
    startIndex = (
        releaseIndex
        + crtIntervalStartOffset
        + int(np.argmax(intensStartIntervalSmooth))
    )
    crtIntervalStartIndex = releaseIndex + crtIntervalStartOffset
    crtIntervalEndIndex = releaseIndex + crtIntervalEndOffset
    if offsetTime != 0:
        offsetStartTime = timeArr[crtIntervalStartIndex] + offsetTime
        offsetStartIndex = int(np.argmin(np.abs(timeArr - offsetStartTime)))
        crtIntervalStartIndex = int(
            np.clip(offsetStartIndex, 0, crtIntervalEndIndex - 1)
        )

    return int(crtIntervalStartIndex), int(startIndex), int(crtIntervalEndIndex)


# }}}
