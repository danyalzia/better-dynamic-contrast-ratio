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
import os
import dxcam
import ctypes
import math

import threading
import time
import screen_brightness_control as sbc
import numpy as np

from monitorcontrol import get_monitors, VCPError

# If two monitors are connected then by default the program will run on primary monitor
# Set it to 1 to run on external/second monitor
MONITOR_INDEX = 0
# VRR is not currently supported
# Program must be running at the current refresh rate of monitor otherwise it behaves badly
TARGET_FPS = 60
# Forcing CPU for now because CPU (numpy based) luminance calculating algorithms are actually faster due to PyTorch/CUDA overhead
CPU_MODE_FORCED = True

BRIGHTNESS_ADAPTATION = True # It's main functionality; you don't want to disable it :D
EXPERIMENTAL_BRIGHTNESS_ADAPTIVE_INCREMENTS = False  # Work in Progress
EXPERIMENTAL_CONTRAST_ADAPTATION = (
    False  # Work in Progress (Don't use it in combination of GAMMA RAMP)
)
EXPERIMENTAL_GAMMA_RAMP_ADJUSTMENTS = False  # Work in Progress

BRIGHTNESS_ADJUSTMENT_INTERVAL = 0.1
BRIGHTNESS_INSTANT_ADJUSTMENTS = False

CONTRAST_ADJUSTMENT_INTERVAL = 0.1

BLOCKING = True # Keep it enabled as disabling it causes a bunch of threads related bugs; needs a lot of testing

# Tolerance levels
LUMA_DIFFERENCE_THRESHOLD = 0
CONTRAST_DIFFERENCE_THRESHOLD = 0
GAMMA_DIFFERENCE_THRESHOLD = 0.0

MIN_BRIGHTNESS = 0
MAX_BRIGHTNESS = 100

MIN_CONTRAST = 0
MAX_CONTRAST = 100

MIN_GAMMA = 0.85
MAX_GAMMA = 1.05

# Luminance calculating algorithms: https://stackoverflow.com/questions/596216/formula-to-determine-perceived-brightness-of-rgb-color
try:
    if CPU_MODE_FORCED:
        raise ValueError("[!] CPU mode is forced. Using CPU version... ")

    try:
        import torch
    except ModuleNotFoundError:
        raise ModuleNotFoundError("[!] PyTorch not found. Using CPU version... ")

    if not torch.cuda.is_available():
        raise ValueError("[!] CUDA is not available. Using CPU version... ")

    print("[!] Using PyTorch (CUDA)... \n")

    def get_average_luminance1(arr: np.ndarray):
        arr = torch.tensor(arr)
        d = torch.tensor([2550.299, 2550.587, 1770.833])

        total_num_sum = np.prod(arr.shape[:-1])
        luminance_total = (arr / d).sum()
        luminance_total = ((luminance_total / total_num_sum) * 255).cuda().item()
        return luminance_total

    # ITU BT.709
    def get_average_luminance2(arr: np.ndarray):
        arr = torch.tensor(arr)
        d = torch.tensor([0.2126, 0.7152, 0.0722])

        mean_rgb = arr.reshape(-1, 3).mean(axis=0, dtype=float)

        luminance = (mean_rgb * d).sum().cuda().item()
        return (luminance / 255) * 100

    # ITU BT.601
    def get_average_luminance3(arr: np.ndarray):
        arr = torch.tensor(arr)
        d = torch.tensor([0.299, 0.587, 0.114])

        mean_rgb = arr.reshape(-1, 3).mean(axis=0, dtype=float)

        luminance = (mean_rgb * d).sum().cuda().item()
        return (luminance / 255) * 100

except (ModuleNotFoundError, ValueError) as err:
    print(err)

    def get_average_luminance1(arr: np.ndarray):
        total_num_sum = np.prod(arr.shape[:-1])
        luminance_total = (arr / [2550.299, 2550.587, 1770.833]).sum()
        return (luminance_total / total_num_sum) * 255

    # ITU BT.709
    def get_average_luminance2(arr: np.ndarray):

        mean_rgb = arr.reshape(-1, 3).mean(axis=0)
        luminance = (mean_rgb * [0.2126, 0.7152, 0.0722]).sum()
        return (luminance / 255) * 100

    # ITU BT.601
    def get_average_luminance3(arr: np.ndarray):

        mean_rgb = arr.reshape(-1, 3).mean(axis=0)
        luminance = (mean_rgb * [0.299, 0.587, 0.114]).sum()
        return (luminance / 255) * 100


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


def fade_contrast(
    monitor, value: int, current: int, interval: float, increment: int = 1
):

    thread = threading.Thread(
        target=_fade_contrast, args=(monitor, value, current, interval, increment)
    )
    thread.start()
    threads = [thread]

    for t in threads:
        t.join()


def set_contrast(monitor, value: int):
    with monitor:
        monitor.set_contrast(value)


def clamp(d, minValue, maxValue):
    t = max(d, minValue)
    return min(t, maxValue)


def change_gamma_values(
    rGamma, gGamma, bGamma, rContrast, gContrast, bContrast, rBright, gBright, bBright
):

    MaxGamma = 4.4
    MinGamma = 0.3
    rGamma = clamp(rGamma, MinGamma, MaxGamma)
    gGamma = clamp(gGamma, MinGamma, MaxGamma)
    bGamma = clamp(bGamma, MinGamma, MaxGamma)

    print(rGamma, gGamma, bGamma)

    MaxContrast = 100.0
    MinContrast = 0.1
    rContrast = clamp(rContrast, MinContrast, MaxContrast)
    gContrast = clamp(gContrast, MinContrast, MaxContrast)
    bContrast = clamp(bContrast, MinContrast, MaxContrast)

    print(rContrast, gContrast, bContrast)

    MaxBright = 1.0
    MinBright = -1.0
    rBright = clamp(rBright, MinBright, MaxBright)
    gBright = clamp(gBright, MinBright, MaxBright)
    bBright = clamp(bBright, MinBright, MaxBright)

    print(rBright, gBright, bBright)

    rInvgamma = 1 / rGamma
    gInvgamma = 1 / gGamma
    bInvgamma = 1 / bGamma
    rNorm = math.pow(255.0, rInvgamma - 1)
    gNorm = math.pow(255.0, gInvgamma - 1)
    bNorm = math.pow(255.0, bInvgamma - 1)

    print(rInvgamma, gInvgamma, bInvgamma)
    print(rNorm, gNorm, bNorm)

    newGamma = np.empty((3, 256), dtype=np.uint16)  # init R, G, and B ramps

    for i in range(256):
        rVal = i * rContrast - (rContrast - 1) * 127
        gVal = i * gContrast - (gContrast - 1) * 127
        bVal = i * bContrast - (bContrast - 1) * 127

        if rGamma != 1:
            rVal = math.pow(rVal, rInvgamma) / rNorm
        if gGamma != 1:
            gVal = math.pow(gVal, gInvgamma) / gNorm
        if bGamma != 1:
            bVal = math.pow(bVal, bInvgamma) / bNorm

        rVal += rBright * 128
        gVal += gBright * 128
        bVal += bBright * 128

        newGamma[0][i] = clamp((int)(rVal * 256), 0, 65535)
        # r
        newGamma[1][i] = clamp((int)(gVal * 256), 0, 65535)
        # g
        newGamma[2][i] = clamp((int)(bVal * 256), 0, 65535)
        # b

    return newGamma


def get_default_gamma_ramp(GetDeviceGammaRamp, hdc):
    default_gamma_ramp = np.empty((3, 256), dtype=np.uint16)
    if not GetDeviceGammaRamp(hdc, default_gamma_ramp.ctypes):
        raise RuntimeError("Can't get default gamma ramp")

    return default_gamma_ramp


def save_gamma_ramp(ramp, filename):
    np.save(filename, ramp)


def load_gamma_ramp(filename):
    return np.load(filename)


# value range: 0.7 - 1.1
def set_gamma(SetDeviceGammaRamp, hdc, gamma_ramp, value):
    Scale = np.array([[value], [value], [value]], float)
    NewRamps = np.uint16(np.round(np.multiply(Scale, gamma_ramp)))

    if not SetDeviceGammaRamp(hdc, NewRamps.ctypes):
        raise ValueError(f"Unable to set Gamma to {value}")


def _fade_gamma(
    SetDeviceGammaRamp,
    hdc,
    gamma_ramp,
    finish: float,
    start: float | None = None,
    interval: float = 0.01,
    increment: float = 0.01,
):
    current_thread = threading.current_thread()

    finishInt = round(finish * 100)
    startInt = round(start * 100)
    incrementInt = round(abs(increment * 100))

    if startInt > finishInt:
        incrementInt = -incrementInt

    next_change_start_time = time.time()

    for valueInt in sbc.helpers.logarithmic_range(startInt, finishInt, incrementInt):

        if threading.current_thread() != current_thread:
            break

        value = valueInt / 100
        # print(f"Setting to {value} from {start / 100} uptil {finish / 100}")

        set_gamma(SetDeviceGammaRamp, hdc, gamma_ramp, value)

        next_change_start_time += interval
        sleep_time = next_change_start_time - time.time()

        if sleep_time > 0:
            time.sleep(sleep_time)
    else:
        set_gamma(SetDeviceGammaRamp, hdc, gamma_ramp, finish)


def fade_gamma(
    SetDeviceGammaRamp,
    hdc,
    gamma_ramp,
    finish: float,
    start: float | None = None,
    interval: float = 0.01,
    increment: float = 0.01,
):

    thread = threading.Thread(
        target=_fade_gamma,
        args=(
            SetDeviceGammaRamp,
            hdc,
            gamma_ramp,
            finish,
            start,
            interval,
            increment,
        ),
    )
    thread.start()
    threads = [thread]

    for t in threads:
        t.join()


if __name__ == "__main__":
    if EXPERIMENTAL_GAMMA_RAMP_ADJUSTMENTS:
        GetDC = ctypes.windll.user32.GetDC
        ReleaseDC = ctypes.windll.user32.ReleaseDC
        SetDeviceGammaRamp = ctypes.windll.gdi32.SetDeviceGammaRamp
        GetDeviceGammaRamp = ctypes.windll.gdi32.GetDeviceGammaRamp

        hdc = ctypes.wintypes.HDC(GetDC(None))
        if not hdc:
            raise RuntimeError("No HDC")

        if os.path.exists("defaultgamma.npy"):
            default_gamma_ramp = load_gamma_ramp("defaultgamma.npy")
        else:
            default_gamma_ramp = get_default_gamma_ramp(GetDeviceGammaRamp, hdc)
            save_gamma_ramp(default_gamma_ramp, "defaultgamma")

        # Start with a bit darkened image
        gamma = 0.90
        set_gamma(SetDeviceGammaRamp, hdc, default_gamma_ramp, gamma)

    monitor = get_monitors()[0]

    # width, height = 1920, 1200
    # size = 100
    # # Take the center of the screen
    # left, top = (width - size) // 2, (height - size) // 2
    # right, bottom = left + size, top + size
    # region = (left, top, right, bottom)

    try:
        camera = dxcam.create(output_idx=MONITOR_INDEX, output_color="GRAY")
    except IndexError as e:
        raise RuntimeError(
            f"Monitor at index {MONITOR_INDEX} is not available, please try a different value."
        ) from e

    camera.start(target_fps=TARGET_FPS)

    try:
        while True:
            # This is almost 2-3 times faster than sbc.get_brightness()[0]
            with monitor:
                while True:
                    try:
                        brightness = monitor.get_luminance()
                        break
                    except VCPError:
                        continue

            frame = camera.get_latest_frame()

            try:
                # current_time = time.time()
                # luma = get_average_luminance1(frame)
                # print(
                #     f"Luma1: {luma} ------------- took {time.time() - current_time} seconds"
                # )

                # current_time = time.time()
                # luma = get_average_luminance2(frame)
                # print(
                #     f"Luma2: {luma} ------------- took {time.time() - current_time} seconds"
                # )

                # current_time = time.time()
                # luma = get_average_luminance3(frame)
                # print(
                #     f"Luma3: {luma} ------------- took {time.time() - current_time} seconds"
                # )
                luma = get_average_luminance3(frame)

                luma = round(luma)
            except ValueError:
                luma = 0

            if EXPERIMENTAL_GAMMA_RAMP_ADJUSTMENTS:
                # May need more than just average luma to adjust gamma appropriately
                # For now, I am certain approximations according to anecdotal experience
                adjusted_gamma = (100 - luma) / 82
                adjusted_gamma = clamp(adjusted_gamma, MIN_GAMMA, MAX_GAMMA)

                # Skip if adjusted gamma is same as previous gamma
                if (
                    adjusted_gamma != gamma
                    and not (
                        GAMMA_DIFFERENCE_THRESHOLD != 0.0
                        and adjusted_gamma > gamma
                        and (adjusted_gamma - gamma) < GAMMA_DIFFERENCE_THRESHOLD
                    )
                    and not (
                        GAMMA_DIFFERENCE_THRESHOLD != 0.0
                        and adjusted_gamma < gamma
                        and (gamma - adjusted_gamma) < GAMMA_DIFFERENCE_THRESHOLD
                    )
                ):
                    fade_gamma(
                        SetDeviceGammaRamp,
                        hdc,
                        default_gamma_ramp,
                        adjusted_gamma,
                        gamma,
                        interval=0.01,
                        increment=0.01,
                    )

                    print(f"Gamma: {adjusted_gamma} (from {gamma})")

                    gamma = adjusted_gamma

            if BRIGHTNESS_ADAPTATION:
                # Skip if the luma is same as current monitor's brightness
                if luma == brightness:
                    print(" ...Skipping brightness adjustment... ")
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
                        if BRIGHTNESS_INSTANT_ADJUSTMENTS
                        else sbc.fade_brightness(
                            MIN_BRIGHTNESS,
                            interval=BRIGHTNESS_ADJUSTMENT_INTERVAL,
                            blocking=BLOCKING,
                        )
                    )
                    # brightness = luma
                    print(f"Luma: {luma}. Clamping to min brightness: {MIN_BRIGHTNESS}")

                elif luma > MAX_BRIGHTNESS:
                    (
                        sbc.set_brightness(MAX_BRIGHTNESS)
                        if BRIGHTNESS_INSTANT_ADJUSTMENTS
                        else sbc.fade_brightness(
                            MAX_BRIGHTNESS,
                            interval=BRIGHTNESS_ADJUSTMENT_INTERVAL,
                            blocking=BLOCKING,
                        )
                    )
                    # brightness = luma
                    print(f"Luma: {luma}. Clamping to max brightness: {MAX_BRIGHTNESS}")

                else:
                    if not EXPERIMENTAL_BRIGHTNESS_ADAPTIVE_INCREMENTS:
                        (
                            sbc.set_brightness(luma)
                            if BRIGHTNESS_INSTANT_ADJUSTMENTS
                            else sbc.fade_brightness(
                                luma,
                                interval=BRIGHTNESS_ADJUSTMENT_INTERVAL,
                                blocking=BLOCKING,
                            )
                        )
                        print(f"Brightness: {luma} (from {brightness})")
                        # brightness = luma
                    else:
                        # Adaptive intervals
                        diff_for_instant = 50
                        diff_for_2x_interval = 10
                        diff_for_3x_interval = 20
                        diff_for_4x_interval = 30
                        diff_for_5x_interval = 40
                        if (
                            luma > brightness
                            and (change_in_luma := (luma - brightness))
                            > diff_for_instant
                        ):
                            print(
                                f"Too much sudden increase in brightness: {luma} from {brightness}"
                            )
                            sbc.set_brightness(luma)
                            brightness = luma
                        elif (
                            luma > brightness
                            and (change_in_luma := (luma - brightness))
                            > diff_for_2x_interval
                        ):
                            print(
                                f"Sudden increase in brightness: {luma} from {brightness}"
                            )
                            sbc.fade_brightness(
                                luma,
                                interval=BRIGHTNESS_ADJUSTMENT_INTERVAL,
                                blocking=BLOCKING,
                                increment=2,
                            )
                            brightness = luma
                        elif (
                            luma > brightness
                            and (change_in_luma := (luma - brightness))
                            > diff_for_3x_interval
                        ):
                            print(
                                f"Sudden increase in brightness: {luma} from {brightness}"
                            )
                            sbc.fade_brightness(
                                luma,
                                interval=BRIGHTNESS_ADJUSTMENT_INTERVAL,
                                blocking=BLOCKING,
                                increment=3,
                            )
                            brightness = luma
                        elif (
                            luma > brightness
                            and (change_in_luma := (luma - brightness))
                            > diff_for_4x_interval
                        ):
                            print(
                                f"Sudden increase in brightness: {luma} from {brightness}"
                            )
                            sbc.fade_brightness(
                                luma,
                                interval=BRIGHTNESS_ADJUSTMENT_INTERVAL,
                                blocking=BLOCKING,
                                increment=4,
                            )
                            brightness = luma
                        elif (
                            luma > brightness
                            and (change_in_luma := (luma - brightness))
                            > diff_for_5x_interval
                        ):
                            print(
                                f"Sudden increase in brightness: {luma} from {brightness}"
                            )
                            sbc.fade_brightness(
                                luma,
                                interval=BRIGHTNESS_ADJUSTMENT_INTERVAL,
                                blocking=BLOCKING,
                                increment=5,
                            )
                            brightness = luma
                        elif (
                            luma < brightness
                            and (change_in_luma := (brightness - luma))
                            > diff_for_instant
                        ):
                            print(
                                f"Too much sudden decrease in brightness: {luma} from {brightness}"
                            )
                            sbc.set_brightness(luma)
                            brightness = luma
                        elif (
                            luma < brightness
                            and (change_in_luma := (brightness - luma))
                            > diff_for_2x_interval
                        ):
                            print(
                                f"Sudden decrease in brightness: {luma} from {brightness}"
                            )
                            sbc.fade_brightness(
                                luma,
                                interval=BRIGHTNESS_ADJUSTMENT_INTERVAL,
                                blocking=BLOCKING,
                                increment=2,
                            )
                            brightness = luma
                        elif (
                            luma < brightness
                            and (change_in_luma := (brightness - luma))
                            > diff_for_3x_interval
                        ):
                            print(
                                f"Sudden decrease in brightness: {luma} from {brightness}"
                            )
                            sbc.fade_brightness(
                                luma,
                                interval=BRIGHTNESS_ADJUSTMENT_INTERVAL,
                                blocking=BLOCKING,
                                increment=3,
                            )
                            brightness = luma
                        elif (
                            luma < brightness
                            and (change_in_luma := (brightness - luma))
                            > diff_for_4x_interval
                        ):
                            print(
                                f"Sudden decrease in brightness: {luma} from {brightness}"
                            )
                            sbc.fade_brightness(
                                luma,
                                interval=BRIGHTNESS_ADJUSTMENT_INTERVAL,
                                blocking=BLOCKING,
                                increment=4,
                            )
                            brightness = luma
                        elif (
                            luma < brightness
                            and (change_in_luma := (brightness - luma))
                            > diff_for_5x_interval
                        ):
                            print(
                                f"Sudden decrease in brightness: {luma} from {brightness}"
                            )
                            sbc.fade_brightness(
                                luma,
                                interval=BRIGHTNESS_ADJUSTMENT_INTERVAL,
                                blocking=BLOCKING,
                                increment=5,
                            )
                            brightness = luma
                        else:
                            if luma != brightness:
                                (
                                    sbc.set_brightness(luma)
                                    if BRIGHTNESS_INSTANT_ADJUSTMENTS
                                    else sbc.fade_brightness(
                                        luma,
                                        interval=BRIGHTNESS_ADJUSTMENT_INTERVAL,
                                        blocking=BLOCKING,
                                    )
                                )
                                print(f"Brightness: {luma} (from {brightness})")
                                brightness = luma

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

                if average_contrast == contrast:
                    continue

                if (
                    CONTRAST_DIFFERENCE_THRESHOLD != 0
                    and average_contrast > contrast
                    and (average_contrast - contrast) < CONTRAST_DIFFERENCE_THRESHOLD
                ):
                    continue

                if (
                    CONTRAST_DIFFERENCE_THRESHOLD != 0
                    and average_contrast < contrast
                    and (contrast - average_contrast) < CONTRAST_DIFFERENCE_THRESHOLD
                ):
                    continue

                if average_contrast < MIN_CONTRAST:
                    fade_contrast(
                        monitor,
                        MIN_CONTRAST,
                        contrast,
                        interval=CONTRAST_ADJUSTMENT_INTERVAL,
                    )
                    print(
                        f"Contrast: {average_contrast}. Clamping to min contrast: {MIN_CONTRAST}"
                    )
                    contrast = average_contrast

                elif average_contrast > MAX_CONTRAST:
                    fade_contrast(
                        monitor,
                        MAX_CONTRAST,
                        contrast,
                        interval=CONTRAST_ADJUSTMENT_INTERVAL,
                    )
                    print(
                        f"Contrast: {average_contrast}. Clamping to max contrast: {MAX_CONTRAST}"
                    )
                    contrast = average_contrast

                elif luma <= MAX_CONTRAST:
                    fade_contrast(
                        monitor,
                        average_contrast,
                        contrast,
                        interval=CONTRAST_ADJUSTMENT_INTERVAL,
                    )
                    print(f"Contrast: {average_contrast} (from {contrast})")
                    contrast = average_contrast
    except KeyboardInterrupt:
        print("[!] Programm interrupted. Closing now... ")
        camera.stop()
        del camera

        if EXPERIMENTAL_GAMMA_RAMP_ADJUSTMENTS and not ReleaseDC(hdc):
            print("[!] Could not release the HDC")

        time.sleep(1)
