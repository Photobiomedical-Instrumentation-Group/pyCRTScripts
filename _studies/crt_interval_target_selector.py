import json
from pathlib import Path

import numpy as np
import pyqtgraph as pg
import tomllib
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

CACHE_DIR = Path("Npz/Cache")
TARGETS_PATH = Path("crt_interval_targets.toml")
TARGET_FRAMES_PATH = Path("target_frames.toml")

# Accepts a cache file, a directory of cache files, or a nested list of either.
RUN_ON = CACHE_DIR

SAVE_TARGETS = True
SKIP_EXISTING_TARGETS = False
DEFAULT_INTERVAL_SECONDS = 7.0
MAX_INTERVAL_SECONDS = 30.0

WINDOW_SIZE = (1600, 900)


pg.setConfigOptions(imageAxisOrder="row-major")


def qtEnum(containerName, enumName):
    container = getattr(QtCore.Qt, containerName, QtCore.Qt)
    return getattr(container, enumName)


HORIZONTAL = qtEnum("Orientation", "Horizontal")
KEY_Q = qtEnum("Key", "Key_Q")
KEY_A = qtEnum("Key", "Key_A")
KEY_ESCAPE = qtEnum("Key", "Key_Escape")

ABORTED = object()


def loadIntervalTargets(targetsPath=None):
    if targetsPath is None:
        targetsPath = TARGETS_PATH
    targetsPath = Path(targetsPath)
    if not targetsPath.exists():
        return {}

    with targetsPath.open("rb") as file:
        config = tomllib.load(file)

    targets = {}
    for videoName, value in config.get("targets", {}).items():
        if not isinstance(value, dict):
            continue
        targets[videoName] = {
            "crtIntervalStartIndex": int(value["crtIntervalStartIndex"]),
            "crtIntervalEndIndex": int(value["crtIntervalEndIndex"]),
        }
    return targets


def loadTargetFrames(targetFramesPath=None):
    if targetFramesPath is None:
        targetFramesPath = TARGET_FRAMES_PATH
    targetFramesPath = Path(targetFramesPath)
    if not targetFramesPath.exists():
        return {}

    with targetFramesPath.open("rb") as file:
        targetConfig = tomllib.load(file)
    return {
        videoName: int(frameIndex)
        for videoName, frameIndex in targetConfig.get("targets", {}).items()
    }


def saveIntervalTargets(targets, targetsPath=None):
    if targetsPath is None:
        targetsPath = TARGETS_PATH
    targetsPath = Path(targetsPath)
    targetsPath.parent.mkdir(parents=True, exist_ok=True)

    lines = ["[targets]"]
    for videoName, target in targets.items():
        lines.append(
            f"{json.dumps(videoName)} = "
            "{ "
            f"crtIntervalStartIndex = {int(target['crtIntervalStartIndex'])}, "
            f"crtIntervalEndIndex = {int(target['crtIntervalEndIndex'])}"
            " }"
        )

    temporaryPath = targetsPath.with_name(f"{targetsPath.name}.tmp")
    temporaryPath.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporaryPath.replace(targetsPath)


def updateIntervalTarget(targets, videoName, startIndex, endIndex, targetsPath=None):
    targets[videoName] = {
        "crtIntervalStartIndex": int(startIndex),
        "crtIntervalEndIndex": int(endIndex),
    }
    saveIntervalTargets(targets, targetsPath)


def iterCachePaths(runOn):
    if isinstance(runOn, (list, tuple, set)):
        for item in runOn:
            yield from iterCachePaths(item)
        return

    runOnPath = Path(runOn)
    if runOnPath.is_dir():
        for cachePath in sorted(
            runOnPath.glob("*.npz"), key=lambda path: path.name.lower()
        ):
            yield cachePath
        return

    yield runOnPath


def loadCacheData(cachePath):
    with np.load(cachePath) as data:
        return {key: data[key] for key in data.files}


def scalarValue(value):
    if value is None:
        return None
    if np.shape(value) == ():
        return value.item()
    return value


def videoNameFromCache(cachePath, cacheData):
    videoPath = scalarValue(cacheData.get("videoPath"))
    if videoPath is not None:
        return Path(str(videoPath)).name
    return cachePath.with_suffix("").name


def validateCache(cachePath, cacheData):
    requiredKeys = {"lFrames", "avgAArr", "avgGArr", "timesScdsArr"}
    missingKeys = requiredKeys - set(cacheData)
    if missingKeys:
        raise KeyError(
            f"{cachePath} is missing key(s): {', '.join(sorted(missingKeys))}"
        )

    frameCount = len(cacheData["timesScdsArr"])
    for key in ("lFrames", "avgAArr", "avgGArr"):
        if len(cacheData[key]) != frameCount:
            raise ValueError(
                f"{cachePath} has inconsistent {key} length: "
                f"{len(cacheData[key])} != {frameCount}"
            )

    if frameCount == 0:
        raise ValueError(f"{cachePath} has no cached frames")


def nearestIndexForTime(timesScdsArr, targetTime):
    timesScdsArr = np.asarray(timesScdsArr, dtype=float)
    return int(np.argmin(np.abs(timesScdsArr - float(targetTime))))


def minMaxNormalizeToUint8(frame):
    frame = np.asarray(frame, dtype=np.float32)
    frameMin = float(np.min(frame))
    frameMax = float(np.max(frame))
    if frameMax <= frameMin:
        return np.zeros(frame.shape, dtype=np.uint8)

    normalized = (frame - frameMin) / (frameMax - frameMin)
    return np.clip(np.round(255 * normalized), 0, 255).astype(np.uint8)


def releaseMarkerFromTargetFrames(videoName, cacheData, targetFrames):
    if videoName not in targetFrames:
        return None, None

    frameCount = len(cacheData["timesScdsArr"])
    releaseIndex = int(np.clip(targetFrames[videoName], 0, frameCount - 1))
    releaseTime = float(cacheData["timesScdsArr"][releaseIndex])
    return releaseIndex, releaseTime


def clampIntervalEndToMaxDuration(startIndex, endIndex, timesScdsArr):
    timesScdsArr = np.asarray(timesScdsArr, dtype=float)
    startIndex = int(startIndex)
    endIndex = int(endIndex)
    if endIndex <= startIndex:
        return startIndex, startIndex

    maxEndTime = float(timesScdsArr[startIndex]) + MAX_INTERVAL_SECONDS
    if float(timesScdsArr[endIndex]) <= maxEndTime:
        return startIndex, endIndex

    eligibleOffsets = np.flatnonzero(timesScdsArr[startIndex : endIndex + 1] <= maxEndTime)
    if len(eligibleOffsets) == 0:
        return startIndex, startIndex

    return startIndex, startIndex + int(eligibleOffsets[-1])


def finalizedIntervalResult(result, timesScdsArr):
    startIndex, endIndex = clampIntervalEndToMaxDuration(
        result["crtIntervalStartIndex"],
        result["crtIntervalEndIndex"],
        timesScdsArr,
    )
    return {
        **result,
        "crtIntervalStartIndex": startIndex,
        "crtIntervalEndIndex": endIndex,
    }


def initialInterval(cacheData, releaseIndex=None, releaseTime=None, existingTarget=None):
    frameCount = len(cacheData["timesScdsArr"])
    if existingTarget is not None:
        startIndex = existingTarget["crtIntervalStartIndex"]
        endIndex = existingTarget["crtIntervalEndIndex"]
        startIndex = int(np.clip(startIndex, 0, frameCount - 1))
        endIndex = int(np.clip(endIndex, 0, frameCount - 1))
        return min(startIndex, endIndex), max(startIndex, endIndex)

    if releaseIndex is None:
        return 0, frameCount - 1

    releaseIndex = int(np.clip(releaseIndex, 0, frameCount - 1))
    if releaseTime is None:
        return releaseIndex, frameCount - 1

    timesScdsArr = np.asarray(cacheData["timesScdsArr"], dtype=float)
    endTime = float(releaseTime) + DEFAULT_INTERVAL_SECONDS
    endIndex = nearestIndexForTime(timesScdsArr, endTime)
    endIndex = max(releaseIndex, endIndex)
    return releaseIndex, int(np.clip(endIndex, 0, frameCount - 1))


class CrtIntervalSelector(QtWidgets.QWidget):
    def __init__(
        self,
        cachePath,
        cacheData,
        videoName,
        initialStartIndex,
        initialEndIndex,
        releaseTime=None,
    ):
        super().__init__()
        self.cachePath = cachePath
        self.cacheData = cacheData
        self.videoName = videoName
        self.timesScdsArr = np.asarray(cacheData["timesScdsArr"], dtype=float)
        self.avgAArr = np.asarray(cacheData["avgAArr"], dtype=float)
        self.avgGArr = np.asarray(cacheData["avgGArr"], dtype=float)
        self.lFrames = np.asarray(cacheData["lFrames"])
        self.frameCount = len(self.timesScdsArr)
        self.startIndex = int(initialStartIndex)
        self.endIndex = int(initialEndIndex)
        self.releaseTime = releaseTime
        self.result = None
        self.eventLoop = None
        self._updatingSliders = False

        self.setWindowTitle(f"CRT interval selector: {self.videoName}")
        self.resize(*WINDOW_SIZE)
        self._buildUi()
        self._updateDisplays()

    def _buildUi(self):
        mainLayout = QtWidgets.QGridLayout(self)

        self.avgAPlot = pg.PlotWidget(title="LAB A average")
        self.avgGPlot = pg.PlotWidget(title="BGR G average")
        self.avgAPlot.plot(self.timesScdsArr, self.avgAArr, pen=pg.mkPen("b", width=1))
        self.avgGPlot.plot(self.timesScdsArr, self.avgGArr, pen=pg.mkPen("g", width=1))

        for plot in (self.avgAPlot, self.avgGPlot):
            plot.showGrid(x=True, y=True, alpha=0.35)
            plot.setLabel("bottom", "Time", units="s")

        if self.releaseTime is not None:
            for plot in (self.avgAPlot, self.avgGPlot):
                plot.addItem(
                    pg.InfiniteLine(
                        pos=self.releaseTime,
                        angle=90,
                        pen=pg.mkPen("r", width=2),
                        label=f"release {self.releaseTime:.3f}s",
                        labelOpts={"position": 0.95},
                    )
                )

        self.avgAStartLine, self.avgAEndLine = self._makeIntervalLines()
        self.avgGStartLine, self.avgGEndLine = self._makeIntervalLines()
        for plot, startLine, endLine in (
            (self.avgAPlot, self.avgAStartLine, self.avgAEndLine),
            (self.avgGPlot, self.avgGStartLine, self.avgGEndLine),
        ):
            plot.addItem(startLine)
            plot.addItem(endLine)

        self.startImageItem, startImageWidget = self._makeImageWidget()
        self.endImageItem, endImageWidget = self._makeImageWidget()

        self.startLabel = QtWidgets.QLabel()
        self.endLabel = QtWidgets.QLabel()

        imageLayout = QtWidgets.QVBoxLayout()
        imageLayout.addWidget(self.startLabel)
        imageLayout.addWidget(startImageWidget, stretch=1)
        imageLayout.addWidget(self.endLabel)
        imageLayout.addWidget(endImageWidget, stretch=1)

        self.startSlider = QtWidgets.QSlider(HORIZONTAL)
        self.endSlider = QtWidgets.QSlider(HORIZONTAL)
        for slider in (self.startSlider, self.endSlider):
            slider.setRange(0, self.frameCount - 1)
            slider.setTickInterval(max(1, self.frameCount // 20))

        self.startSlider.setValue(self.startIndex)
        self.endSlider.setValue(self.endIndex)
        self.startSlider.valueChanged.connect(self._startSliderChanged)
        self.endSlider.valueChanged.connect(self._endSliderChanged)

        self.instructionsLabel = QtWidgets.QLabel(
            'Press "q" to save this interval and continue. Press "a" to abort. Press Esc to skip this cache.'
        )
        self.acceptShortcut = QtGui.QShortcut(QtGui.QKeySequence("q"), self)
        self.acceptShortcut.activated.connect(self._acceptSelection)
        self.abortShortcut = QtGui.QShortcut(QtGui.QKeySequence("a"), self)
        self.abortShortcut.activated.connect(self._abortSelection)
        self.skipShortcut = QtGui.QShortcut(QtGui.QKeySequence("Esc"), self)
        self.skipShortcut.activated.connect(self._skipSelection)

        mainLayout.addWidget(self.avgAPlot, 0, 0)
        mainLayout.addWidget(self.avgGPlot, 1, 0)
        mainLayout.addLayout(imageLayout, 0, 1, 2, 1)
        mainLayout.addWidget(QtWidgets.QLabel("crtIntervalStartIndex"), 2, 0)
        mainLayout.addWidget(self.startSlider, 3, 0, 1, 2)
        mainLayout.addWidget(QtWidgets.QLabel("crtIntervalEndIndex"), 4, 0)
        mainLayout.addWidget(self.endSlider, 5, 0, 1, 2)
        mainLayout.addWidget(self.instructionsLabel, 6, 0, 1, 2)
        mainLayout.setColumnStretch(0, 3)
        mainLayout.setColumnStretch(1, 1)

    def _makeIntervalLines(self):
        startLine = pg.InfiniteLine(
            angle=90,
            pen=pg.mkPen("c", width=2),
            label="start",
            labelOpts={"position": 0.1},
        )
        endLine = pg.InfiniteLine(
            angle=90,
            pen=pg.mkPen("m", width=2),
            label="end",
            labelOpts={"position": 0.2},
        )
        return startLine, endLine

    def _makeImageWidget(self):
        widget = pg.GraphicsLayoutWidget()
        viewBox = widget.addViewBox(lockAspect=True)
        viewBox.invertY(True)
        imageItem = pg.ImageItem()
        viewBox.addItem(imageItem)
        return imageItem, widget

    def _startSliderChanged(self, value):
        if self._updatingSliders:
            return

        self.startIndex = int(value)
        if self.startIndex > self.endIndex:
            self.endIndex = self.startIndex
            self._setSliderValue(self.endSlider, self.endIndex)
        self._updateDisplays()

    def _endSliderChanged(self, value):
        if self._updatingSliders:
            return

        self.endIndex = int(value)
        if self.endIndex < self.startIndex:
            self.startIndex = self.endIndex
            self._setSliderValue(self.startSlider, self.startIndex)
        self._updateDisplays()

    def _setSliderValue(self, slider, value):
        self._updatingSliders = True
        try:
            slider.setValue(int(value))
        finally:
            self._updatingSliders = False

    def _updateDisplays(self):
        startTime = self.timesScdsArr[self.startIndex]
        endTime = self.timesScdsArr[self.endIndex]
        for line in (self.avgAStartLine, self.avgGStartLine):
            line.setPos(startTime)
        for line in (self.avgAEndLine, self.avgGEndLine):
            line.setPos(endTime)

        self.startImageItem.setImage(
            minMaxNormalizeToUint8(self.lFrames[self.startIndex]),
            autoLevels=False,
            levels=(0, 255),
        )
        self.endImageItem.setImage(
            minMaxNormalizeToUint8(self.lFrames[self.endIndex]),
            autoLevels=False,
            levels=(0, 255),
        )

        self.startLabel.setText(
            f"crtIntervalStartIndex = {self.startIndex} ({startTime:.3f} s)"
        )
        self.endLabel.setText(
            f"crtIntervalEndIndex = {self.endIndex} ({endTime:.3f} s)"
        )

    def keyPressEvent(self, event):
        if event.key() == KEY_Q:
            self._acceptSelection()
            return

        if event.key() == KEY_A:
            self._abortSelection()
            return

        if event.key() == KEY_ESCAPE:
            self._skipSelection()
            return

        super().keyPressEvent(event)

    def _acceptSelection(self):
        self.result = {
            "crtIntervalStartIndex": self.startIndex,
            "crtIntervalEndIndex": self.endIndex,
        }
        self.close()

    def _skipSelection(self):
        self.result = None
        self.close()

    def _abortSelection(self):
        self.result = {
            "crtIntervalStartIndex": self.startIndex,
            "crtIntervalEndIndex": self.endIndex,
            "aborted": True,
        }
        self.close()

    def closeEvent(self, event):
        if self.eventLoop is not None and self.eventLoop.isRunning():
            self.eventLoop.quit()
        super().closeEvent(event)


def selectInterval(
    cachePath, cacheData, videoName, initialStartIndex, initialEndIndex, releaseTime=None
):
    window = CrtIntervalSelector(
        cachePath,
        cacheData,
        videoName,
        initialStartIndex,
        initialEndIndex,
        releaseTime,
    )
    eventLoop = QtCore.QEventLoop()
    window.eventLoop = eventLoop
    window.show()
    eventLoop.exec()
    return window.result


def main():
    app = pg.mkQApp("CRT interval target selector")
    targets = loadIntervalTargets()
    targetFrames = loadTargetFrames()

    for cachePath in iterCachePaths(RUN_ON):
        if not cachePath.exists():
            print(f"Skipping {cachePath}: cache file not found", flush=True)
            continue

        try:
            cacheData = loadCacheData(cachePath)
            validateCache(cachePath, cacheData)
        except Exception as err:
            print(f"Skipping {cachePath.name}: {type(err).__name__}: {err}", flush=True)
            continue

        videoName = videoNameFromCache(cachePath, cacheData)
        if SKIP_EXISTING_TARGETS and videoName in targets:
            print(
                f"Skipping {videoName}: CRT interval target already exists", flush=True
            )
            continue

        existingTarget = targets.get(videoName)
        if existingTarget is not None:
            print(
                f"Verifying {videoName}: "
                f"start={existingTarget['crtIntervalStartIndex']}, "
                f"end={existingTarget['crtIntervalEndIndex']}",
                flush=True,
            )

        releaseIndex, releaseTime = releaseMarkerFromTargetFrames(
            videoName, cacheData, targetFrames
        )
        if releaseIndex is None:
            print(f"{videoName}: no release index found in {TARGET_FRAMES_PATH}", flush=True)

        startIndex, endIndex = initialInterval(
            cacheData, releaseIndex, releaseTime, existingTarget
        )
        result = selectInterval(
            cachePath, cacheData, videoName, startIndex, endIndex, releaseTime
        )
        if isinstance(result, dict) and result.get("aborted"):
            result = finalizedIntervalResult(result, cacheData["timesScdsArr"])
            if SAVE_TARGETS:
                updateIntervalTarget(
                    targets,
                    videoName,
                    result["crtIntervalStartIndex"],
                    result["crtIntervalEndIndex"],
                )
                print(f"Saved {videoName} to {TARGETS_PATH}", flush=True)
            print(f"aborted on {videoName}", flush=True)
            break

        if result is ABORTED:
            if SAVE_TARGETS:
                saveIntervalTargets(targets)
            print(f"aborted on {videoName}", flush=True)
            break

        if result is None:
            print(f"Skipped {videoName}", flush=True)
            continue

        result = finalizedIntervalResult(result, cacheData["timesScdsArr"])
        print(
            f'"{videoName}" = '
            "{ "
            f"crtIntervalStartIndex = {result['crtIntervalStartIndex']}, "
            f"crtIntervalEndIndex = {result['crtIntervalEndIndex']}"
            " }",
            flush=True,
        )
        if SAVE_TARGETS:
            updateIntervalTarget(
                targets,
                videoName,
                result["crtIntervalStartIndex"],
                result["crtIntervalEndIndex"],
            )
            print(f"Saved {videoName} to {TARGETS_PATH}", flush=True)

    app.quit()


if __name__ == "__main__":
    main()
