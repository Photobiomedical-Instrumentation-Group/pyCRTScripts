import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
# from pathlib import Path

testPath = "/home/eduardo/Code/Python/pyCRTScripts/_studies/Npz/Cache/0528_test1.1.npz"
arq = np.load(testPath)
lFrames = arq["lFrames"]
times = arq["timesScdsArr"]
roi = arq["roi"]
avgAArr = arq["avgAArr"]
avgGArr = arq["avgGArr"]
print(lFrames.shape)

numFrames = lFrames.shape[0]
for frameIndex in range(numFrames):
    # print(frame.shape)
    cv.imshow("lFrames", lFrames[frameIndex, :, :])

    key = cv.waitKey(1)
    if key == ord("q"):
        break
cv.destroyAllWindows()

plt.plot(times, avgAArr)
plt.plot(times, avgGArr)
plt.show()
