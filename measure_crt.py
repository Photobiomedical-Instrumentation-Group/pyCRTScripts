from functions import multiVideoPipeline, singleVideoPipeline
from gui import (askSingleVideoOrDirectory, showDirSuccessMessage,
                 showVideoErrorMessage, showVideoSuccessMessage)

answer = askSingleVideoOrDirectory()

if answer == "directory":
    failedMeasurements = multiVideoPipeline(
        "configuration.toml", "defaults.toml"
    )
    showDirSuccessMessage(failedMeasurements)
else:
    if singleVideoPipeline("configuration.toml", "defaults.toml"):
        showVideoSuccessMessage()
    else:
        showVideoErrorMessage()
