import unittest
from pathlib import Path
from unittest.mock import patch

import browser_actions


class BrowserActionTests(unittest.TestCase):
    @patch("browser_actions.subprocess.Popen")
    @patch("browser_actions.find_chrome", return_value=Path("chrome.exe"))
    def test_rock_opens_motivation_video_in_new_window(self, find_chrome, popen):
        browser_actions.open_motivation_video()
        find_chrome.assert_called_once_with()
        popen.assert_called_once_with(
            ["chrome.exe", "--new-window", browser_actions.MOTIVATION_URL],
            stdout=browser_actions.subprocess.DEVNULL,
            stderr=browser_actions.subprocess.DEVNULL,
        )

    @patch("browser_actions.subprocess.Popen")
    @patch("browser_actions.find_chrome", return_value=Path("chrome.exe"))
    def test_double_clap_opens_one_song_window_at_five_seconds(
        self,
        find_chrome,
        popen,
    ):
        browser_actions.open_clap_song()
        find_chrome.assert_called_once_with()
        self.assertIn("&t=5s", browser_actions.CLAP_SONG_URL)
        popen.assert_called_once_with(
            ["chrome.exe", "--new-window", browser_actions.CLAP_SONG_URL],
            stdout=browser_actions.subprocess.DEVNULL,
            stderr=browser_actions.subprocess.DEVNULL,
        )


if __name__ == "__main__":
    unittest.main()
