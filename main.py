# MIT License

# Copyright (c) 2024 Danyal Zia

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import dxcam

import threading
import time
import screen_brightness_control as sbc
import numpy as np

from monitorcontrol import get_monitors

CPU_MODE_FORCED = True
PERFORMANCE_MODE = False

EXPERIMENTAL_CONTRAST_ADAPTATION = False  # Work in Progress

ADJUSTMENT_INTERVAL = 0.1
BLOCKING = False
LUMA_DIFFERENCE_THRESHOLD = 0

MIN_BRIGHTNESS = 0
MAX_BRIGHTNESS = 100

MAKE_ADJUSTMENTS_INSTANT = False

# Luminance calculating algorithms: https://stackoverflow.com/questions/596216/formula-to-determine-perceived-brightness-of-rgb-color
# Performance mode is less accurate (more sensitive to luminiance changes)
if PERFORMANCE_MODE:
    luma_lut = [255.299, 255.299, 177.833]
else:
    luma_lut = [2550.299, 2550.587, 1770.833]

try:
    if CPU_MODE_FORCED:
        raise ValueError("[!] CPU Mode is forced, will use CPU (slower) version.")

    import torch

    if not torch.cuda.is_available():
        raise ValueError("[!] CUDA is not available, will use CPU (slower) version.")

    print("Using PyTorch/CUDA... \n")

    def get_average_luminance(arr):
        arr = torch.tensor(arr).cuda()
        d = torch.tensor(luma_lut).cuda()

        total_num_sum = np.prod(arr.shape[:-1])
        luminance_total = (arr / d).sum().cuda().item()
        return luminance_total / total_num_sum

except (ModuleNotFoundError, ValueError) as err:
    print(err)

    print("Using CPU... \n")

    def get_average_luminance(arr):
        total_num_sum = np.prod(arr.shape[:-1])
        luminance_total = (arr / luma_lut).sum()
        return luminance_total / total_num_sum


def _fade_contrast(
    monitor,
    finish: int,
    start: int | None = None,
    interval: float = 0.01,
    increment: int = 1,
):
    current_thread = threading.current_thread()

    increment = abs(increment)
    if start > finish:
        increment = -increment

    next_change_start_time = time.time()

    for value in sbc.helpers.logarithmic_range(start, finish, increment):
        if threading.current_thread() != current_thread:
            break

        with monitor:
            monitor.set_contrast(value)

        next_change_start_time += interval
        sleep_time = next_change_start_time - time.time()

        if sleep_time > 0:
            time.sleep(sleep_time)
    else:
        with monitor:
            monitor.set_contrast(finish)


def fade_contrast(monitor, value: int, current: int, interval: float):

    thread = threading.Thread(
        target=_fade_contrast, args=(monitor, value, current, interval)
    )
    thread.start()
    threads = [thread]

    for t in threads:
        t.join()


if __name__ == "__main__":
    monitor = get_monitors()[0]

    # width, height = 1920, 1200
    # size = 100
    # # Take the center of the screen
    # left, top = (width - size) // 2, (height - size) // 2
    # right, bottom = left + size, top + size
    # region = (left, top, right, bottom)

    camera = dxcam.create(output_idx=0, output_color="GRAY")
    camera.start()

    import time

    while True:
        frame = camera.get_latest_frame()

        try:
            luma = get_average_luminance(frame) * 255

            luma = round(luma / 10) if PERFORMANCE_MODE else round(luma)
        except ValueError:
            luma = 0

        brightness = sbc.get_brightness()[0]

        # Skip if the luma is same as current monitor's brightness
        if luma == brightness:
            continue

        if (
            LUMA_DIFFERENCE_THRESHOLD != 0
            and luma > brightness
            and (luma - brightness) < LUMA_DIFFERENCE_THRESHOLD
        ):
            continue

        if (
            LUMA_DIFFERENCE_THRESHOLD != 0
            and luma < brightness
            and (brightness - luma) < LUMA_DIFFERENCE_THRESHOLD
        ):
            continue

        if luma < MIN_BRIGHTNESS:
            (
                sbc.set_brightness(MIN_BRIGHTNESS)
                if MAKE_ADJUSTMENTS_INSTANT
                else sbc.fade_brightness(
                    MIN_BRIGHTNESS, interval=ADJUSTMENT_INTERVAL, blocking=BLOCKING
                )
            )
            print(f"Luma: {luma}. Clamping to min brightness: {MIN_BRIGHTNESS}")

        if luma > MAX_BRIGHTNESS:
            (
                sbc.set_brightness(MAX_BRIGHTNESS)
                if MAKE_ADJUSTMENTS_INSTANT
                else sbc.fade_brightness(
                    MAX_BRIGHTNESS, interval=ADJUSTMENT_INTERVAL, blocking=BLOCKING
                )
            )
            print(f"Luma: {luma}. Clamping to max brightness: {MAX_BRIGHTNESS}")

        if luma >= MIN_BRIGHTNESS and luma <= MAX_BRIGHTNESS:
            (
                sbc.set_brightness(luma)
                if MAKE_ADJUSTMENTS_INSTANT
                else sbc.fade_brightness(
                    luma, interval=ADJUSTMENT_INTERVAL, blocking=BLOCKING
                )
            )
            print(f"Brightness: {luma}")

        if EXPERIMENTAL_CONTRAST_ADAPTATION:
            while True:
                try:
                    with monitor:
                        contrast = monitor.get_contrast()
                        break
                except Exception:
                    continue

            # Work in progress
            # Naive implementation but helps in gradual adjustments of display's luminance
            # It also leads to less washed out (relaxing) colors in high luminance content
            # Blacks are not improved though in very low lumninance content
            average_contrast = 100 - luma

            if average_contrast != contrast:
                fade_contrast(
                    monitor, average_contrast, contrast, interval=ADJUSTMENT_INTERVAL
                )
                print(f"Contrast: {average_contrast} (from {contrast})")
    else:
        camera.stop()
        del camera
