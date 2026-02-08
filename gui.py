import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk, messagebox

def askSingleVideoOrDirectory() -> str:
    """
    Opens a modal dialog asking:
    "Measure CRT on a single video or a directory?"

    Returns
    -------
    str
        Either "single video" or "directory".
    """
    result: str | None = None

    root = tk.Tk()
    root.withdraw()  # hide main window

    dialog = tk.Toplevel(root)
    dialog.title("Select input type")
    dialog.resizable(False, False)
    dialog.attributes("-topmost", True)
    dialog.grab_set()  # modal

    ttk.Label(
        dialog,
        text="Measure CRT on a single video or a directory?",
        padding=(20, 15),
    ).pack()

    btn_frame = ttk.Frame(dialog, padding=(10, 10))
    btn_frame.pack()

    def choose(value: str) -> None:
        nonlocal result
        result = value
        dialog.destroy()

    ttk.Button(
        btn_frame,
        text="Single video",
        command=lambda: choose("single video"),
        width=18,
    ).grid(row=0, column=0, padx=5)

    ttk.Button(
        btn_frame,
        text="Directory",
        command=lambda: choose("directory"),
        width=18,
    ).grid(row=0, column=1, padx=5)

    dialog.protocol("WM_DELETE_WINDOW", lambda: None)  # disable close
    root.wait_window(dialog)
    root.destroy()

    if result is None:
        raise RuntimeError("Dialog closed without a selection.")

    return result

def askNextVideo(videoName: str) -> str:
    # {{{
    """
    Open a dialog asking whether to continue to the next video. By chatGPT

    Parameters
    ----------
    videoName : str
        Name of the next video.

    Returns
    -------
    str
        One of: "yes", "skip", "select", "abort"
    """
    result: str | None = None

    def set_result(value: str) -> None:
        nonlocal result
        result = value
        root.destroy()

    root = tk.Tk()
    root.title("Continue?")
    root.resizable(False, False)
    root.attributes("-topmost", True)

    frame = ttk.Frame(root, padding=12)
    frame.pack(fill="both", expand=True)

    label = ttk.Label(
        frame,
        text=f"Continue to next video:\n{videoName}?",
        justify="center",
    )
    label.pack(pady=(0, 12))

    btn_frame = ttk.Frame(frame)
    btn_frame.pack()

    ttk.Button(btn_frame, text="Yes", command=lambda: set_result("yes")).grid(
        row=0, column=0, padx=5
    )
    ttk.Button(
        btn_frame, text="Skip", command=lambda: set_result("skip")
    ).grid(row=0, column=1, padx=5)
    ttk.Button(
        btn_frame, text="Select Video", command=lambda: set_result("select")
    ).grid(row=0, column=2, padx=5)
    ttk.Button(
        btn_frame, text="Abort", command=lambda: set_result("abort")
    ).grid(row=0, column=3, padx=5)

    root.protocol("WM_DELETE_WINDOW", lambda: set_result("abort"))
    root.mainloop()

    assert result is not None
    return result


# }}}


def selectFile() -> Path:
    # {{{
    root = tk.Tk()
    root.withdraw()  # hide main window
    root.attributes("-topmost", True)
    selection = filedialog.askopenfilename()
    if not selection:
        raise RuntimeError("No video selected. Aborting...")
    filePath = Path(selection)
    root.destroy()
    return filePath

# }}}


def selectDirectory() -> Path:
    # {{{
    root = tk.Tk()
    root.withdraw()  # hide main window
    root.attributes("-topmost", True)
    selection = filedialog.askdirectory()
    if not selection:
        raise RuntimeError("No directory selected. Aborting...")
    dirPath = Path(selection)
    root.destroy()
    return dirPath


# }}}

def showVideoSuccessMessage() -> None:
    # {{{
    root = tk.Tk()
    root.withdraw()  # hide main window
    root.attributes("-topmost", True)

    messagebox.showinfo(
        title="Success",
        message=(
            "CRT measured successfully.\n\n"
            "Check the console and/or the plots directory and the spreadsheet "
            "for the results."
        )
    )

    root.destroy()
    
# }}}


def showDirSuccessMessage(failedMeasurements) -> None:
    # {{{
    root = tk.Tk()
    root.withdraw()  # hide main window
    root.attributes("-topmost", True)

    if failedMeasurements > 0:
        messagebox.showinfo(
            title="Success",
            message=(
                "Finished processing directory.\n\n"
                f"Measurement failed on {failedMeasurements} videos.\n"
                "Check the console and/or logs, plots directory, "
                "and spreadsheets for results"
            ),
        )
    else:
        messagebox.showinfo(
            title="Success",
            message=(
                "Finished processing directory.\n\n"
                "Measurement successfull on all videos.\n"
                "Check the console and/or logs, plots directory, "
                "and spreadsheets for results"
            ),
        )

    root.destroy()
    
 # }}}
 
def showVideoErrorMessage() -> None:
    # {{{
    root = tk.Tk()
    root.withdraw()  # hide main window
    root.attributes("-topmost", True)

    messagebox.showinfo(
        title="Error",
        message=(
            "CRT measured failed.\n\n"
            "Check the console for details.\n"
            "Try changing the ROi, exclusionMethod, and/or exclusionCriteria "
            "and try again."
        )
    )

    root.destroy()
    
# }}}
