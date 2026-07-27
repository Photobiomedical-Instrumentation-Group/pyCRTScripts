from pathlib import Path

import cv2 as cv
import numpy as np
import tomli
from matplotlib import pyplot as plt
from pyCRT import frameOperations as fo
from pyCRT import videoReading as vr
from scipy.ndimage import gaussian_filter1d, median_filter, uniform_filter1d
from scipy.signal import savgol_filter
from skimage.restoration import denoise_tv_chambolle


def showFrameRoi(name, frame, roi):
    cv.imshow(name, fo.drawRoi(frame.copy(), roi))


def applyMedian(frame, kernel):
    return cv.medianBlur(frame, (2 * kernel) + 1)


def applyBlur(frame, kernel):
    # {{{
    kernel = (2 * kernel) + 1
    return cv.GaussianBlur(frame, (kernel, kernel), 0)


# }}}


def nothing(x):
    pass


def showBoolFrame(name, frameBool):
    cv.imshow(name, 255 * frameBool.astype(np.uint8))


def bgrToYCrCbPaper(frame):
    # {{{
    B, G, R = frame[..., 0], frame[..., 1], frame[..., 2]

    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = R - Y
    Cb = B - Y

    return np.stack((Y, Cr, Cb), axis=-1).astype(np.float32)


# }}}


def skinDetector(frame):
    # {{{
    assert frame.dtype == np.uint8

    frameHSV = fo.bgrToHsv(frame)
    frameBGR = frame.astype(np.float32)
    frameYCrCb = bgrToYCrCbPaper(frame)

    H, S, _ = frameHSV[..., 0], frameHSV[..., 1], frameHSV[..., 2]
    B, G, R = frameBGR[..., 0], frameBGR[..., 1], frameBGR[..., 2]
    Y, Cr, Cb = frameYCrCb[..., 0], frameYCrCb[..., 1], frameYCrCb[..., 2]

    case1 = (
        (H >= 0.0)
        & (H <= 65.0)
        & (S >= 0.10)
        & (S <= 0.85)
        & (R > 70)
        & (G > 25)
        & (B > 10)
        & (R >= G)
        & (R >= B)
        & (np.abs(R - G) > 5)
    )

    case2 = (
        (R > 80)
        & (G > 30)
        & (B > 15)
        & (R >= G)
        & (R >= B)
        & (np.abs(R - G) > 10)
        & (Cr > 115)
        & (Cb > 60)
        & (Y > 50)
        & (Cr <= 1.5862 * Cb + 30.0)
        & (Cr >= 0.3448 * Cb + 65.0)
        & (Cr >= -4.5652 * Cb + 220.0)
        & (Cr <= -1.15 * Cb + 315.0)
        & (Cr <= -2.2857 * Cb + 450.0)
    )

    return case1 | case2


# }}}


def cannyEdgeDetector(frame, thresh1, thresh2, blurKernel, l2grad):
    # {{{
    # assumes frame is uint8
    if blurKernel > 0:
        frame = applyBlur(frame, blurKernel)
    return cv.Canny(
        frame,
        threshold1=int(thresh1),
        threshold2=int(thresh2),
        L2gradient=l2grad,
    )


# }}}


def laplacianEdgeDetector(frame, ksize, blurKernel, scale, delta):
    # {{{
    # assumes frame is uint8
    if blurKernel > 0:
        frame = applyBlur(frame, blurKernel)
    return cv.Laplacian(
        frame,
        ddepth=cv.CV_64F,
        ksize=(2 * ksize) + 1,
        scale=float(scale),
        delta=float(delta),
    )


# }}}


def minMaxNormalize(array):
    # {{{
    arrMin = array.min()
    arrMax = array.max()
    return (array - arrMin) / (arrMax - arrMin)


# }}}


def movingAvg(array, k):
    # {{{
    return uniform_filter1d(array, size=2 * k + 1, mode="nearest")


# }}}


def measureMetrics(
    videoPath,
    savePath: str | None = None,
    scaleFactor=0.5,
    roi=None,
    showRescaled=False,
    showGray=False,
    showCanny=False,
    showLaplacian=False,
    showMedian=False,
    showControls=False,
    defaultParams=None,
):
    # {{{
    defaults = {
        "Median": 2,
        "Blur": 3,
        "Thresh1": 5,
        "Thresh2": 27,
        "L2Grad": 1,
        "Ksize": 2,
        "LapBlur": 5,
        "Scale": 1,
        "Delta": 0,
    }
    if defaultParams is not None:
        defaults.update(defaultParams)

    parameters = defaults.copy()

    if showControls:
        cv.namedWindow("Controls")

        trackbarMaxima = {
            "Median": 25,
            "Blur": 25,
            "Thresh1": 100,
            "Thresh2": 100,
            "L2Grad": 1,
            "Ksize": 3,
            "LapBlur": 25,
            "Scale": 20,
            "Delta": 100,
        }

        for name, defaultValue in defaults.items():
            cv.createTrackbar(
                name,
                "Controls",
                defaultValue,
                trackbarMaxima[name],
                nothing,
            )

    avgIntensList = []
    avgIntensLABList = []
    cannyList = []
    lapList = []
    timeList = []
    paused = False

    with vr.videoCapture(videoPath) as cap:
        for originalFrame in vr.frameReader(cap):
            if showControls:
                for name in parameters:
                    parameters[name] = cv.getTrackbarPos(name, "Controls")

            timeScds = cap.get(cv.CAP_PROP_POS_MSEC) / 1000.0
            timeList.append(timeScds)

            rescaledFrame = fo.rescaleFrame(originalFrame, scaleFactor)
            if showRescaled:
                showFrameRoi("rescaledFrame", rescaledFrame, roi)

            croppedFrame = fo.cropFrame(rescaledFrame, roi)
            avgIntensList.append(cv.mean(croppedFrame)[:3])

            labFrame = cv.cvtColor(croppedFrame, cv.COLOR_BGR2LAB)
            avgIntensLABList.append(cv.mean(labFrame)[:3])

            medianFrame = applyMedian(croppedFrame, parameters["Median"])
            if showMedian:
                cv.imshow("medianFrame", medianFrame)

            grayFrame = cv.cvtColor(medianFrame, cv.COLOR_BGR2GRAY)
            if showGray:
                cv.imshow("grayFrame", grayFrame)

            cannyFrame = cannyEdgeDetector(
                grayFrame,
                parameters["Thresh1"],
                parameters["Thresh2"],
                parameters["Blur"],
                parameters["L2Grad"],
            )
            cannyList.append(np.sum(cannyFrame.astype(np.float32)))
            if showCanny:
                cv.imshow("cannyFrame", cannyFrame)

            lapFrame = laplacianEdgeDetector(
                grayFrame,
                parameters["Ksize"],
                parameters["LapBlur"],
                parameters["Scale"],
                parameters["Delta"],
            )
            lapList.append(np.sum(lapFrame.astype(np.int32) ** 2))
            if showLaplacian:
                cv.imshow("lapFrame", cv.convertScaleAbs(lapFrame))

            key = cv.waitKey(0) if paused else cv.waitKey(1)

            if key == ord("q"):
                break
            if key == ord("p"):
                paused = True
            elif key == ord("o"):
                paused = False
            elif key == ord("i") and paused:
                paused = True
                continue

    cv.destroyAllWindows()
    avgIntensArr = np.array(avgIntensList)
    avgIntensLABArr = np.array(avgIntensLABList)
    cannyArr = np.array(cannyList)
    lapArr = np.array(lapList)
    timeArr = np.array(timeList)
    timeArr -= timeArr.min()
    cannyArr = minMaxNormalize(cannyArr)
    lapArr = minMaxNormalize(lapArr)

    if savePath is not None:
        np.savez(
            savePath,
            avgIntensArr=avgIntensArr,
            avgIntensLABArr=avgIntensLABArr,
            cannyArr=cannyArr,
            lapArr=lapArr,
            timeArr=timeArr,
        )

    return (
        avgIntensArr,
        avgIntensLABArr,
        cannyArr,
        lapArr,
        timeArr,
    )


# }}}


def plotMetrics(
    timeArr,
    avgIntensLABArrAvgd,
    avgIntensArrAvgd,
    avgIntensArrGrad,
    cannyArrAvgd,
    cannyArrGrad,
    lapArrAvgd,
    lapArrGrad,
    title="",
):
    # {{{
    fig, axes = plt.subplots(nrows=4, ncols=1, layout="tight", figsize=(10, 8))
    twinAxes = []

    bgrColors = ("b", "g", "r")
    labColors = ("black", "magenta", "orange")
    labLabels = ("L", "A", "B")

    for ax in axes:
        ax.margins(x=0)
        twinAxes.append(ax.twinx())

    # Normalize every LAB channel separately, only for plotting.
    avgIntensLABArrNorm = avgIntensLABArrAvgd.astype(np.float64).copy()

    for i in range(3):
        channel = avgIntensLABArrNorm[..., i]
        channelMin = channel.min()
        channelMax = channel.max()
        channelRange = channelMax - channelMin

        if channelRange > 0:
            avgIntensLABArrNorm[..., i] = (channel - channelMin) / channelRange
        else:
            avgIntensLABArrNorm[..., i] = 0.0

    for i in range(3):
        axes[0].plot(
            timeArr,
            avgIntensLABArrNorm[..., i],
            color=labColors[i],
            label=f"LAB {labLabels[i]}",
        )

    axes[0].set_ylabel("LAB avg\nnormalized")

    for i in range(3):
        axes[1].plot(
            timeArr,
            avgIntensArrAvgd[..., i],
            color=bgrColors[i],
            label=bgrColors[i].upper(),
        )

    twinAxes[1].plot(
        timeArr,
        avgIntensArrGrad[..., 1],
        color="g",
        label="G grad",
        lw=1,
        ls="--",
    )

    axes[1].set_ylabel("BGR avg")

    axes[2].plot(
        timeArr,
        cannyArrAvgd,
        label="Canny",
        color="blue",
    )

    twinAxes[2].plot(
        timeArr,
        cannyArrGrad,
        label="Canny Grad",
        color="gray",
        ls="--",
    )

    axes[2].set_ylabel("Canny")

    axes[3].plot(
        timeArr,
        lapArrAvgd,
        label="Laplace",
        color="orange",
    )

    twinAxes[3].plot(
        timeArr,
        lapArrGrad,
        label="Laplace Grad",
        color="gray",
        ls="--",
    )

    axes[3].set_ylabel("Laplacian")
    axes[3].set_xlabel("Time (s)")

    for ax, twinAx in zip(axes, twinAxes):
        ax.legend(loc="lower left")

        handles, _ = twinAx.get_legend_handles_labels()
        if handles:
            twinAx.legend(loc="lower right")

    fig.suptitle(title)


# }}}


def plotZoomInMetrics(
    timeArr,
    avgIntensArr,
    lapArr,
    lapArrGrad,
    cannyArr,
    cannyArrGrad,
    title,
):
    # {{{

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5), layout="tight")

    avgIntensG = avgIntensArr[:, 1]
    # prominence = 0.5
    # GPeaks, _ = find_peaks(avgIntensG, prominence=prominence)
    maxIndex = np.argmax(avgIntensG)
    maxTime = timeArr[maxIndex]

    minGradLapIndex = np.argmin(lapArrGrad)
    minGradLapTime = timeArr[minGradLapIndex]

    minGradCannyIndex = np.argmin(cannyArrGrad)
    minGradCannyTime = timeArr[minGradCannyIndex]

    time1 = maxTime - 3
    time2 = maxTime + 5
    # time2 = timeArr[-1]

    index1 = np.argmin(np.abs(timeArr - time1))
    index2 = np.argmin(np.abs(timeArr - time2))

    timeArr2 = timeArr[index1:index2]
    avgIntensG2 = minMaxNormalize(avgIntensG[index1:index2])

    cannyArr2 = minMaxNormalize(cannyArr[index1:index2])
    lapArr2 = minMaxNormalize(lapArr[index1:index2])

    ax1.plot(timeArr2, avgIntensG2, color="green")
    ax2.plot(timeArr2, cannyArr2, color="blue")
    ax2.plot(timeArr2, lapArr2, color="orange")

    ax1.axvline(maxTime, color="black", ls="--", label="maxIntens")
    ax1.axvline(minGradLapTime, color="orange", ls="--", label="minGradLapTime")
    ax1.axvline(minGradCannyTime, color="blue", ls="--", label="minGradCannyTime")

    ax2.axvline(maxTime, color="black", ls="--", label="maxIntens")
    ax2.axvline(minGradLapTime, color="orange", ls="--", label="minGradLapTime")
    ax2.axvline(minGradCannyTime, color="blue", ls="--", label="minGradCannyTime")

    ax1.legend(loc="upper right")
    ax1.margins(0, 0.1)
    ax2.margins(0, 0.1)
    fig.suptitle(title)


# }}}


def bilateralFilter1D(arr, spatialSigma, intensitySigma, radius=None):
    # {{{
    arr = np.asarray(arr, dtype=float)

    if radius is None:
        radius = int(np.ceil(3 * spatialSigma))

    filtered = np.empty_like(arr)

    for i in range(len(arr)):
        left = max(0, i - radius)
        right = min(len(arr), i + radius + 1)

        localValues = arr[left:right]
        localIndices = np.arange(left, right)

        spatialWeights = np.exp(-0.5 * ((localIndices - i) / spatialSigma) ** 2)

        intensityWeights = np.exp(-0.5 * ((localValues - arr[i]) / intensitySigma) ** 2)

        weights = spatialWeights * intensityWeights
        filtered[i] = np.sum(weights * localValues) / np.sum(weights)

    return filtered


# }}}


def ensmoothenSignal(arr, method="moving average", **kwargs):
    # {{{
    method = method.strip().lower()

    if method == "moving average":
        return movingAvg(arr, kwargs["avgKernel"])

    if method == "median":
        return median_filter(arr, size=kwargs["kernelSize"])

    if method == "gaussian":
        return gaussian_filter1d(arr, sigma=kwargs["sigma"])

    if method == "savitzky-golay":
        return savgol_filter(
            arr,
            window_length=kwargs["windowLength"],
            polyorder=kwargs["polyorder"],
            mode=kwargs.get("mode", "interp"),
        )

    if method == "total variation":
        return denoise_tv_chambolle(
            arr,
            weight=kwargs["weight"],
            eps=kwargs.get("eps", 2e-4),
            max_num_iter=kwargs.get("maxNumIter", 200),
        )

    if method == "bilateral":
        return bilateralFilter1D(
            arr,
            spatialSigma=kwargs["spatialSigma"],
            intensitySigma=kwargs["intensitySigma"],
            radius=kwargs.get("radius"),
        )

    if method == "nothing":
        return arr

    raise ValueError(f"Method {method} is not valid.")


# }}}


def findReleaseAndStartIndices(cannyArr, timeArr):
    # {{{

    cannyArrGrad = np.gradient(cannyArr, timeArr)
    minGradIndex = np.argmin(cannyArrGrad)
    releaseIndex = np.argwhere(
        (np.arange(len(cannyArr)) >= minGradIndex) * (np.abs(cannyArrGrad) <= 0.1)
    )[0][0]
    startTime = timeArr[releaseIndex] + 0.1
    startIndex = np.argmin(np.abs(timeArr - startTime))
    # print(releaseIndex)
    # print(startIndex)
    return releaseIndex, startIndex


# }}}


def findReleaseAndStartIndicesFromUnsmoothedArrays(
    cannyArr, timeArr, smoothingParams=None
):
    # {{{
    if smoothingParams is None:
        smoothingParams = {}

    cannySmoothingParams = smoothingParams.get(
        "canny", {"method": "median", "kernelSize": 15}
    )
    cannyArrSmooth = ensmoothenSignal(cannyArr, **cannySmoothingParams)

    releaseIndex, startIndex = findReleaseAndStartIndices(cannyArrSmooth, timeArr)

    return releaseIndex, startIndex


# }}}


def dictToString(dictio):
    # {{{
    stringList = []
    for key, value in dictio.items():
        if isinstance(value, (float, np.floating)):
            valueType = "float"
            value = float(value)
        elif isinstance(value, (int, np.integer)):
            valueType = "int"
            value = int(value)
        elif value is None:
            valueType = "None"
            value = "None"
        else:
            raise ValueError(
                "dictToString error: don't know how to deal with "
                f"{value} of type {type(value)}"
            )
        stringList.append(f"{key}: {valueType} {value}")
    return "; ".join(stringList)


# }}}


def stringToDict(string):
    # {{{
    stringList = string.split("; ")
    newDict = {}
    for subString in stringList:
        key, valueType, value = subString.split(" ")
        if valueType == "float":
            newDict[key[:-1]] = float(value)
        elif valueType == "int":
            newDict[key[:-1]] = int(value)
        elif valueType == "None":
            newDict[key[:-1]] = None
        else:
            raise ValueError(
                "stringToDict error: don't know how to deal with "
                f"{value} of type {valueType}"
            )
    return newDict


# }}}


def plotReleaseTimingForFile(filePath, smoothingParams, savePath=None, **kwargs):
    # {{{
    filePath = Path(filePath)

    if filePath.suffix == ".npz":
        arq = np.load(filePath)
        avgIntensArr = arq["avgIntensArr"]
        avgIntensLABArr = arq["avgIntensLABArr"]
        cannyArr = arq["cannyArr"]
        lapArr = arq["lapArr"]
        timeArr = arq["timeArr"]

    else:
        showEverything = kwargs.get("default", False)

        roi = kwargs.get("roi")
        showRescaled = kwargs.get("showRescaled", showEverything)
        showGray = kwargs.get("showGray", showEverything)
        showCanny = kwargs.get("showCanny", showEverything)
        showLaplacian = kwargs.get("showLaplacian", showEverything)
        showMedian = kwargs.get("showMedian", showEverything)
        showControls = kwargs.get("showControls", showEverything)

        defaultParams = kwargs.get("defaultParams")

        (
            avgIntensArr,
            avgIntensLABArr,
            cannyArr,
            lapArr,
            timeArr,
        ) = measureMetrics(
            str(filePath),
            scaleFactor=0.5,
            roi=roi,
            showRescaled=showRescaled,
            showGray=showGray,
            showCanny=showCanny,
            showLaplacian=showLaplacian,
            showMedian=showMedian,
            showControls=showControls,
            defaultParams=defaultParams,
            savePath=savePath,
        )

    avgIntensArrAvgd = avgIntensArr.copy()
    avgIntensArrGrad = avgIntensArr.copy()

    avgIntensLABArrAvgd = avgIntensLABArr.copy()

    for i in range(3):
        avgIntensArrAvgd[..., i] = ensmoothenSignal(
            avgIntensArr[..., i],
            **smoothingParams["avgIntens"],
        )
        avgIntensArrGrad[..., i] = np.gradient(
            avgIntensArr[..., i],
            timeArr,
        )

        avgIntensLABArrAvgd[..., i] = ensmoothenSignal(
            avgIntensLABArr[..., i],
            **smoothingParams["avgIntens"],
        )

    cannyArrAvgd = ensmoothenSignal(cannyArr, **smoothingParams["canny"])
    cannyArrGrad = np.gradient(cannyArrAvgd, timeArr)

    lapArrAvgd = ensmoothenSignal(lapArr, **smoothingParams["lap"])
    lapArrGrad = np.gradient(lapArrAvgd, timeArr)

    plotMetrics(
        timeArr,
        avgIntensLABArrAvgd,
        avgIntensArrAvgd,
        avgIntensArrGrad,
        cannyArrAvgd,
        cannyArrGrad,
        lapArrAvgd,
        lapArrGrad,
        title=filePath.stem,
    )

    plotZoomInMetrics(
        timeArr,
        avgIntensArrAvgd,
        lapArrAvgd,
        lapArrGrad,
        cannyArrAvgd,
        cannyArrGrad,
        filePath.stem,
    )

    greenIntensityNorm = minMaxNormalize(avgIntensArrAvgd[..., 1])
    maxIndex = np.argmax(greenIntensityNorm)
    maxTime = timeArr[maxIndex]

    releaseIndex = findReleaseAndStartIndices(cannyArrAvgd, timeArr)[0]
    releaseTime = timeArr[releaseIndex]

    plotStartTime = releaseTime - 2
    plotStartIndex = np.argmin(np.abs(timeArr - plotStartTime))

    plotEndTime = releaseTime + 5
    plotEndIndex = np.argmin(np.abs(timeArr - plotEndTime))

    greenIntensityNorm = greenIntensityNorm[plotStartIndex:plotEndIndex]
    timeArrZoomed = timeArr[plotStartIndex:plotEndIndex]
    cannyArrGradZoomed = cannyArrGrad[plotStartIndex:plotEndIndex]

    _, ax = plt.subplots(figsize=(11, 5.5), layout="constrained")
    ax.margins(x=0)

    ax.plot(
        timeArrZoomed,
        greenIntensityNorm,
        lw=2.2,
        color="tab:green",
        label="Normalized green intensity",
    )

    eventLines = [
        (releaseTime, "Release", "tab:blue"),
        # (startTime, "Start", "black"),
        (maxTime, "Peak", "tab:red"),
    ]

    for t, label, color in eventLines:
        if timeArrZoomed[0] <= t <= timeArrZoomed[-1]:
            ax.axvline(
                t,
                color=color,
                ls="--",
                lw=1.5,
                alpha=0.9,
                label=f"{label}: {t:.2f} s",
            )

    ax2 = ax.twinx()
    ax2.plot(
        timeArrZoomed,
        cannyArrGradZoomed,
        lw=1.3,
        color="0.35",
        alpha=0.75,
        label="Canny gradient",
    )

    ax.set_title(filePath.stem)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Green intensity, min-max normalized")
    ax2.set_ylabel("Canny gradient")

    ax.grid(True, which="major", alpha=0.25)
    ax.minorticks_on()
    ax.grid(True, which="minor", alpha=0.10)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="lower right",
        frameon=True,
        framealpha=0.9,
    )

    plt.show()


# }}}


def loadVideoRoi(videoPath, roiTomlPath):
    # {{{
    """
    Load the ROI for a video from a TOML file.

    Parameters
    ----------
    videoPath : str or Path
        Full or relative path to the video. Only the filename, including
        extension, is used for lookup.

    roiTomlPath : str or Path
        Path to the TOML file containing the [roi] table.

    Returns
    -------
    roi : tuple[int, int, int, int]
        ROI as (x, y, width, height).
    """
    videoName = Path(videoPath).name

    with open(roiTomlPath, "rb") as file:
        roiConfig = tomli.load(file)

    try:
        roi = roiConfig["roi"][videoName]
    except KeyError as err:
        availableVideos = "; ".join(roiConfig.get("roi", {}).keys())

        raise KeyError(
            f"No ROI found for video {videoName!r}. Available videos: {availableVideos}"
        ) from err

    if len(roi) != 4:
        raise ValueError(
            f"ROI for {videoName!r} must contain exactly 4 values. Got {roi}."
        )

    return tuple(map(int, roi))


# }}}


def findReleaseAndStartIndicesFromFile(videoPath, npzPath):
    # {{{
    assert isinstance(videoPath, Path)
    assert isinstance(npzPath, Path)
    videoName = videoPath.stem
    arq = np.load(npzPath / (videoName + ".npz"))
    cannyArr, timeArr = arq["cannyArr"], arq["timeArr"]
    releaseIndex, startIndex = findReleaseAndStartIndicesFromUnsmoothedArrays(
        cannyArr, timeArr
    )
    return releaseIndex, startIndex


# }}}
