import os
import subprocess
from pathlib import Path


MOTIVATION_URL = "https://www.youtube.com/shorts/_dtgxpEBNwA"
CLAP_SONG_URL = "https://www.youtube.com/watch?v=pAgnJDJN4VA&t=5s"


def find_chrome():
    configured_path = os.environ.get("CHROME_PATH")
    candidates = []
    if configured_path:
        candidates.append(Path(configured_path))

    candidates.extend(
        (
            Path(os.environ.get("PROGRAMFILES", "C:/Program Files"))
            / "Google/Chrome/Application/chrome.exe",
            Path(os.environ.get("PROGRAMFILES(X86)", "C:/Program Files (x86)"))
            / "Google/Chrome/Application/chrome.exe",
            Path(os.environ.get("LOCALAPPDATA", ""))
            / "Google/Chrome/Application/chrome.exe",
        )
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        "No encuentro Chrome. Instálalo o define CHROME_PATH con la ruta de chrome.exe."
    )


def open_chrome_window(url):
    chrome_path = find_chrome()
    return subprocess.Popen(
        [str(chrome_path), "--new-window", url],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def open_motivation_video():
    return open_chrome_window(MOTIVATION_URL)


def open_clap_song():
    return open_chrome_window(CLAP_SONG_URL)
