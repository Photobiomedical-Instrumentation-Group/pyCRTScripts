# pyCRTScripts

User-friendly scripts designed to make pyCRT accessible to non-programmers.

# Usage

## Through the EXE file (Windows only)

Assuming the settings in the configuration file haven't been altered, the
following steps describe the entire usage of the program:

1. Navigate to the directory `dist/measure_crt/`
2. Execute the file `measure_crt.exe`

Windows may block the execution of the program due to "unknown source". This
is expected. Simply click "More info" -> "Run anyway".

3. Select whether you would like to measure CRT in a single video or in all
   the videos inside a directory.
4. If a Region of Interest (ROI) was not specified in the configuration file
   (see below), press the spacebar key when pressure is being applied, click
   and drag the mouse to draw a square around the desired ROI.
5. Check the terminal. If a ROI was selected manually, it should appear there
   as a tuple of 4 numbers.
6. Wait for video playback to finish.
7. If CRT measurement was successful, you should see plots of the average
   intensities in each channel and the CRT fit result.
8. The aforementioned graphs are saved in the `dists/measure_crt/Plots`
   directory.
9. The file `results_sheet.csv` is created automatically. It contains a
   spreadsheet of the parameters and results of every CRT measurement
   performed using the program. The file `results_sheet.xlsx` is the same
   spreadsheet, but in a format more suitable for spreadsheet software such as
   MS Excel.
10. A log of all the messages printed on the terminal is kept in the file
    log.txt, organized by the date and time of execution of this program.

## Through the command line

The script can also be executed in your Python environment. Simply run either
`measure_crt.py` in the command line:

```
python measure_crt.py
```

Upon running the script, you may proceed according to the instructions for the
EXE file.

In case you specifically want to measure CRT from a single video file or from
an entire directory of videos, you may run the scripts `measure_crt_video.py`
and `measure_crt_dir.py` respectively.


## Configuration file

The CRT measurement parameters, as well as the program's settings can be
configured with the `configuration.toml`, which is read at the start of each
execution. More details to be added later...

# To-do

This is a list of features I would like to implement (I don't promise I will
actually implement them all):

* Add option to view the plots and CRT before saving;
* Try to implement interactive fromTime, toTime and algorithm selection;
* Implement PCA results as a channel;
* Develop algorithm to find appropriate fromTime and toTime;
* Add fromTime and toTime to avgIntensPlot;
* Add monochrome as a possible channel;
* Add tests for pyCRTScripts;
* Add 9010 as an algorithm;
* Add timestamp to csv;
* Add input "Test another video?";
* Add tests;
* Output error messages to the log;
* Create specific exception types for not selecting the ROI and for failed CRT
  fit;
* Add option to ignore duplicates in multiVideoPipeline;
