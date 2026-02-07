from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
import gui

import numpy as np
import pandas as pd

try:
    import tomllib as tomli  # Python 3.11+
except ModuleNotFoundError:
    import tomli  # fallback for older Python

from pyCRT.simpleUI import DATETIME_FORMAT, PCRT, RoiTuple

PLAYBACK_FPS_SPEED = {
    "fast": np.inf,
    "normal": 0,
    "slow": 25,
}

NORM_LEVEL = 25
logging.addLevelName(NORM_LEVEL, "NORM")
VIDEO_FORMATS = (".mp4", ".wmv", ".avi", ".mov", ".mkv")


def getLogger(name: str):
    # {{{
    logger = logging.getLogger(name)
    if not logger.handlers:  # prevent duplicate handlers
        handler = logging.StreamHandler()
        fmt = logging.Formatter("[%(levelname)s] %(message)s")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


# }}}

LOGGER = getLogger("mauricio")



def loadConfigFile(
    tomlPath, defaultsPath: Path | str = "defaults.toml"
) -> dict:
    # {{{
    with open(tomlPath, "rb") as arq:
        configDict = tomli.load(arq)
    with open(defaultsPath, "rb") as arq:
        defaultsDict = tomli.load(arq)

    setDefaults(defaultsDict, configDict)
    return defaultsDict  # counter-intuitive, but it is correct


# }}}


def setDefaults(
    target: dict[str, Any],
    source: dict[str, Any],
) -> None:
    # {{{
    # {{{
    """
    Recursively update `target` in-place using `dict.update()` semantics.
    - Nested dicts are merged recursively.
    - Non-dict values overwrite defaults.
    """
    # }}}
    for key, value in source.items():
        if (
            key in target
            and isinstance(target[key], dict)
            and isinstance(value, dict)
        ):
            setDefaults(target[key], value)
        else:
            # dict.update semantics at this level
            target.update({key: value})


# }}}


def measureCRTVideoFromConfig(
    videoPath: Path | str, configDict: dict[str, Any]
) -> tuple[float, RoiTuple]:
    # {{{
    videoPath = Path(videoPath)

    # [Video] configuration
    rescaleFactor = configDict["Video"]["rescaleFactor"]
    try:
        playbackSpeed = configDict["Video"]["playbackSpeed"]
        playbackFPS = PLAYBACK_FPS_SPEED[playbackSpeed]
    except KeyError:
        raise ValueError(
            f"'{playbackSpeed}' is not a valid value. "
            "Valid values are 'fast', 'normal' or 'slow'."
        )
    displayVideo = configDict["Video"]["displayVideo"]
    livePlot = configDict["Video"]["livePlot"]

    # [Measurement] configuration
    sliceMethod = configDict["Measurement"]["sliceMethod"]
    fromTime = configDict["Measurement"]["fromTime"]
    toTime = configDict["Measurement"]["toTime"]
    exclusionCriteria = configDict["Measurement"]["exclusionCriteria"]
    exclusionMethod = configDict["Measurement"]["exclusionMethod"]
    channel = configDict["Measurement"]["channel"]
    initialGuesses = configDict["Measurement"]["initialGuesses"]
    initialGuessesDict = {
        "exponential": initialGuesses,
        "pCRT": initialGuesses,
    }
    roi = configDict["Measurement"]["roi"]
    if roi == -1:
        roi = None
    elif not isValidRoi:
        raise ValueError(
            f"{roi} is not a valid value for the ROI. "
            "Valid values are 4-element iterables."
        )
    else:
        roi = tuple(roi)

    pcrtObj = PCRT.fromVideoFile(
        str(videoPath),
        roi=roi,
        displayVideo=displayVideo,
        livePlot=livePlot,
        rescaleFactor=rescaleFactor,
        playbackFPS=playbackFPS,
        channel=channel,
        fromTime=fromTime,
        toTime=toTime,
        sliceMethod=sliceMethod,
        initialGuesses=initialGuessesDict,
        exclusionCriteria=exclusionCriteria,
        exclusionMethod=exclusionMethod,
    )
    return pcrtObj, roi


# }}}


def isValidRoi(obj: Any) -> bool:
    # {{{
    try:
        t = tuple(obj)
    except TypeError:
        return False
    return len(t) == 4


# }}}


def findUniquePath(path: str | Path) -> Path:
    # {{{
    path = Path(path)
    if not path.exists():
        return path

    stem = path.stem
    suffix = path.suffix
    parent = path.parent

    i = 1
    while True:
        candidate = parent / f"{stem} ({i}){suffix}"
        if not candidate.exists():
            return candidate
        i += 1


# }}}


def savePlots(
    pcrtObj: PCRT,
    videoName: str,
    plotPath: Path | str,
    overwrite: bool = False,
) -> None:
    # {{{
    plotPath = Path(plotPath)
    if not plotPath.exists() or not plotPath.is_dir():
        raise ValueError(f"{plotPath} does not exist or is not a directory.")
    avgIntensPath = plotPath / f"{videoName}_avgIntens.png"
    pcrtPath = plotPath / f"{videoName}_pCRTPlot.png"
    if not overwrite:
        avgIntensPath = findUniquePath(avgIntensPath)
        pcrtPath = findUniquePath(pcrtPath)
    pcrtObj.saveAvgIntensPlot(avgIntensPath)
    LOGGER.info(f"AvgIntens plot saved in {avgIntensPath}")
    pcrtObj.savePCRTPlot(pcrtPath)
    LOGGER.info(f"CRTPlot plot saved in {pcrtPath}")


# }}}


def saveCSV(
    pcrtObj: PCRT,
    videoName: str,
    csvPath: Path | str,
    overwrite: bool = False,
) -> None:
    # {{{
    colLabels = ["pCRT", "Unc", "RelUnc", "CT", "ExcCri", "ExcMet", "Time"]

    row = {
        "pCRT": round(float(pcrtObj.pCRT[0]), 3),
        "Unc": round(float(pcrtObj.pCRT[1]), 3),
        "RelUnc": round(float(pcrtObj.relativeUncertainty), 3),
        "CT": round(float(pcrtObj.criticalTime), 3),
        "ExcCri": pcrtObj.exclusionCriteria,
        "ExcMet": pcrtObj.exclusionMethod,
        "Time": pcrtObj.dateTime.strftime(DATETIME_FORMAT),
    }

    csvPath = Path(csvPath)

    if csvPath.exists():
        csvSheet = pd.read_csv(
            csvPath, sep=",", encoding="utf-8-sig", index_col=0
        )

        # validate expected schema
        if list(csvSheet.columns) != colLabels:
            raise RuntimeError(
                f"Could not load {csvPath}. CSV columns are invalid."
            )

        if videoName in csvSheet.index and not overwrite:
            oldVideoName = videoName
            videoName = findUniqueName(oldVideoName, list(csvSheet.index))
            LOGGER.info(f"{oldVideoName} found in CSV, saving as {videoName}")

        csvSheet.loc[videoName, colLabels] = [row[c] for c in colLabels]

    else:
        csvSheet = pd.DataFrame([row], index=[videoName], columns=colLabels)

    csvSheet.to_csv(csvPath, encoding="utf-8-sig")
    LOGGER.info(f"Saved CRT in {csvPath}")


# }}}


def saveNpz(
    pcrtObj: PCRT,
    videoName: str,
    npzPath: Path | str,
    overwrite: bool = False,
) -> None:
    # {{{
    npzPath = Path(npzPath)
    if not npzPath.exists() or not npzPath.is_dir():
        raise ValueError(f"{npzPath} does not exist or is not a directory.")
    npzFilePath = npzPath / f"{videoName}.npz"
    if not overwrite:
        npzFilePath = findUniquePath(npzFilePath)
    pcrtObj.name = videoName
    pcrtObj.save(npzFilePath)
    LOGGER.info(f"pCRT object saved in {npzFilePath}.")


# }}}


def findUniqueName(name: str, nameList: list[str]) -> str:
    # {{{
    i = 1
    while True:
        candidate = f"{name} ({i})"
        if candidate not in nameList:
            return candidate
        i += 1


# }}}


def roiToString(roi: RoiTuple) -> str:
    # {{{
    if not isValidRoi:
        raise ValueError(
            f"{roi} is not a valid value for the ROI. "
            "Valid values are 4-element iterables."
        )
    return str(roi).replace(",", "c")


# }}}


def stringToRoi(roiString: str) -> RoiTuple:
    # {{{
    if not isinstance(roiString, str):
        raise TypeError("The argument to stringToRoi must be a string")
    numbers = roiString.strip()[1:-2]
    roi = tuple(int(x) for x in numbers.split("c "))
    return roi


# }}}


def singleVideoPipeline(
    configPath: Path | str,
    defaultsPath: Path | str,
) -> PCRT:
    # {{{
    configDict = loadConfigFile(configPath, defaultsPath)

    LOGGER.setLevel(configDict["General"]["logLevel"])
    LOGGER.info(f"Config loaded from {configPath}.")
    crtVideoPath = gui.selectFile()
    videoName = crtVideoPath.stem
    pcrtObj, roi = measureCRTVideoFromConfig(crtVideoPath, configDict)

    LOGGER.log(NORM_LEVEL, f"{pcrtObj.plotTitle}: {pcrtObj}")

    plotPath = configDict["Files"]["plotPath"]
    csvPath = configDict["Files"]["csvPath"]
    npzPath = configDict["Files"]["npzPath"]
    overwrite = configDict["Files"]["overwrite"]

    showPlots = configDict["General"]["showPlots"]
    if showPlots:
        pcrtObj.showAvgIntensPlot()
        pcrtObj.showPCRTPlot()

    savePlots(pcrtObj, videoName, plotPath, overwrite)
    saveCSV(pcrtObj, videoName, csvPath, overwrite)
    saveNpz(pcrtObj, videoName, npzPath, overwrite)


# }}}


def multiVideoPipeline(
    configPath: Path | str,
    defaultsPath: Path | str,
) -> list[PCRT]:
    # {{{
    configDict = loadConfigFile(configPath, defaultsPath)

    LOGGER.setLevel(configDict["General"]["logLevel"])
    LOGGER.info(f"Config loaded from {configPath}.")

    askConfirmation = configDict["General"]["askConfirmation"]
    # Just in case the user decides to read the videos out of order
    processedPaths = []

    plotPath = configDict["Files"]["plotPath"]
    csvPath = configDict["Files"]["csvPath"]
    npzPath = configDict["Files"]["npzPath"]
    overwrite = configDict["Files"]["overwrite"]
    showPlots = configDict["General"]["showPlots"]

    dirPath = gui.selectDirectory()
    for candidatePath in dirPath.iterdir():
        if not candidatePath.is_file():
            LOGGER.debug(
                f"Skipping {candidatePath.stem}.{candidatePath.suffix} "
                "(not a file)..."
            )
            continue
        if candidatePath.suffix not in VIDEO_FORMATS:
            LOGGER.debug(
                f"Skipping {candidatePath.stem}.{candidatePath.suffix} "
                "(not a video)..."
            )
            continue
        if candidatePath in processedPaths:
            LOGGER.debug(
                f"Skipping {candidatePath.stem}.{candidatePath.suffix} "
                "(already processed)..."
            )
            continue
        if askConfirmation:
            answer = gui.askNextVideo(candidatePath)
            if answer == "skip":
                processedPaths.append(candidatePath)
                continue
            elif answer == "abort":
                break
            elif answer == "select":
                actualPath = gui.selectFile()
            else:
                actualPath = candidatePath
        else:
            actualPath = candidatePath

        LOGGER.info(f"Processing video {actualPath}.")
        
        videoName = actualPath.stem
        try:
            pcrtObj, _ = measureCRTVideoFromConfig(actualPath, configDict)
            processedPaths.append(actualPath)
        except (RuntimeError, TypeError, ValueError):
            LOGGER.error("CRT calculation failed on {actualPath}")
            processedPaths.append(actualPath)
            continue

        LOGGER.log(NORM_LEVEL, f"{pcrtObj.plotTitle}: {pcrtObj}")

        if showPlots:
            pcrtObj.showAvgIntensPlot()
            pcrtObj.showPCRTPlot()
        savePlots(pcrtObj, videoName, plotPath, overwrite)
        saveCSV(pcrtObj, videoName, csvPath, overwrite)
        saveNpz(pcrtObj, videoName, npzPath, overwrite)

    LOGGER.info(f"Finished processing videos in {dirPath}")
            


# }}}
