import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-pycrt")

import cv2 as cv
import numpy as np
from funcs import loadVideoRoi
from pyCRT.videoReading import frameReader, videoCapture
from release_frame_processing import frameToCachedMetrics

VIDEOS_DIR_LIST = [
    Path("/home/eduardo/Data/miscVideos"),
    Path("/home/eduardo/Data/raquelMasters"),
    Path("/home/eduardo/Data/yutaoNewEquipment"),
    Path("/home/eduardo/Data/yutaoPostNew"),
    Path("/home/eduardo/Data/WorstVideos"),
    # Path("TrainingVideos"),
]

VIDEO_EXTENSIONS = [".MOV", ".wmv", ".mp4"]

ROIS_PATH = Path("rois_full.toml")
CACHE_DIR = Path("Npz/Cache")
SKIP_EXISTING = True
RESCALE_FACTOR = 0.5
MEDIAN_KERNEL_RADIUS = 1


def cacheVideo(videoPath, cachePath, rescaleFactor, roi):
    assert isinstance(videoPath, Path)
    assert isinstance(cachePath, Path)

    lFrames = []
    avgAList = []
    avgGList = []
    timeScdsList = []

    with videoCapture(str(videoPath)) as cap:
        for frame in frameReader(cap):
            lFrame, avgA, avgG = frameToCachedMetrics(
                frame,
                roi,
                medianKernelRadius=MEDIAN_KERNEL_RADIUS,
                rescaleFactor=rescaleFactor,
            )

            lFrames.append(lFrame)
            avgAList.append(avgA)
            avgGList.append(avgG)
            timeScdsList.append(cap.get(cv.CAP_PROP_POS_MSEC) / 1000.0)

    lFramesArr = np.array(lFrames)
    avgAArr = np.array(avgAList)
    avgAArr = avgAArr.max() - avgAArr
    avgGArr = np.array(avgGList)
    # avgGArr = avgGArr.max() - avgGArr
    timesScdsArr = np.array(timeScdsList)

    cachePath.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cachePath,
        lFrames=lFramesArr,
        avgAArr=avgAArr,
        avgGArr=avgGArr,
        timesScdsArr=timesScdsArr,
        roi=np.array(roi),
        rescaleFactor=rescaleFactor,
        medianKernelRadius=MEDIAN_KERNEL_RADIUS,
        videoPath=str(videoPath),
    )


def iterVideoPaths():
    for videoDir in VIDEOS_DIR_LIST:
        for filePath in sorted(videoDir.iterdir(), key=lambda path: path.name.lower()):
            if filePath.suffix in VIDEO_EXTENSIONS:
                yield filePath


if __name__ == "__main__":
    skippedVideos = []

    for videoPath in iterVideoPaths():
        cachePath = CACHE_DIR / f"{videoPath.stem}.npz"
        if SKIP_EXISTING and cachePath.exists():
            reason = "cache already exists"
            print(f"skipping {videoPath}: {reason}")
            skippedVideos.append((videoPath, reason))
            continue

        try:
            roi = loadVideoRoi(videoPath, ROIS_PATH)
        except (KeyError, ValueError) as err:
            reason = f"no ROI found ({err})"
            print(f"skipping {videoPath}: {reason}")
            skippedVideos.append((videoPath, reason))
            continue

        print(f"caching {videoPath}...")
        cacheVideo(videoPath, cachePath, RESCALE_FACTOR, roi)

    if skippedVideos:
        print("\nSkipped videos:")
        for videoPath, reason in skippedVideos:
            print(f"  {videoPath}: {reason}")
    else:
        print("\nSkipped videos: none")
