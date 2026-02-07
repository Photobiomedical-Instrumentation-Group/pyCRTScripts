import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk


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
