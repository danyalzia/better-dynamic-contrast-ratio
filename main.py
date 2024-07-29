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
import numpy as np

from monitorcontrol import get_monitors, VCPError
import config

# Luminance calculating algorithms: https://stackoverflow.com/questions/596216/formula-to-determine-perceived-brightness-of-rgb-color
try:
    if config.CPU_MODE_FORCED:
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

    # Fastest method when the source is grayscale
    # Will not work in RGB mode
    def get_average_luminance4(arr: np.ndarray):
        arr = torch.tensor(arr)
        l = arr.shape[0] * arr.shape[1]
        luminance = arr.sum().cuda().item() / l
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

    # Fastest method when the source is grayscale
    # Will not work in RGB mode
    def get_average_luminance4(arr: np.ndarray):
        l = arr.shape[0] * arr.shape[1]
        luminance = arr.sum() / l
        return (luminance / 255) * 100


def fade_brightness(
    monitor, finish: int, start: int, interval: float, increment: int = 1
):
    increment = abs(increment)
    if start > finish:
        increment = -increment

    next_change_start_time = time.time()
    for value in range(start, finish, increment):
        monitor.set_luminance(value)

        next_change_start_time += interval
        sleep_time = next_change_start_time - time.time()

        if sleep_time > 0:
            time.sleep(sleep_time)

    return finish


def fade_contrast(
    monitor, finish: int, start: int, interval: float, increment: int = 1
):
    increment = abs(increment)
    if start > finish:
        increment = -increment

    next_change_start_time = time.time()

    for value in range(start, finish, increment):
        monitor.set_contrast(value)

        next_change_start_time += interval
        sleep_time = next_change_start_time - time.time()

        if sleep_time > 0:
            time.sleep(sleep_time)

    return finish


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

    for valueInt in range(startInt, finishInt, incrementInt):

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
    threads: list[threading.Thread] = []
    if config.EXPERIMENTAL_GAMMA_RAMP_ADJUSTMENTS:
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

    # monitor_w = screeninfo.get_monitors()[config.MONITOR_INDEX].width
    # monitor_h = screeninfo.get_monitors()[config.MONITOR_INDEX].height
    # region_size = 100
    # # Take the center of the screen
    # left, top = (monitor_w - region_size) // 2, (monitor_h - region_size) // 2
    # right, bottom = left + region_size, top + region_size
    # region = (left, top, right, bottom)

    try:
        camera = dxcam.create(output_idx=config.MONITOR_INDEX, output_color="GRAY")
    except IndexError as e:
        raise RuntimeError(
            f"Monitor at index {config.MONITOR_INDEX} is not available, please try a different value."
        ) from e

    camera.start(target_fps=config.TARGET_FPS)

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
                luma = get_average_luminance4(frame)

                luma = round(luma)
            except ValueError as e:
                raise RuntimeError(
                    "[!] Cannot calculate average luminance of the content."
                ) from e

            # Gamma ramp adjustments need to be done before Brightness adjustments to make the perceived lumniance changes less aggressive
            if config.EXPERIMENTAL_GAMMA_RAMP_ADJUSTMENTS:
                # May need more than just average luma to adjust gamma appropriately
                # For now, I am certain approximations according to anecdotal experience
                adjusted_gamma = (100 - luma) / 82
                adjusted_gamma = clamp(
                    adjusted_gamma, config.MIN_GAMMA, config.MAX_GAMMA
                )

                # Skip if adjusted gamma is same as previous gamma
                if (
                    adjusted_gamma != gamma
                    and not (
                        config.GAMMA_DIFFERENCE_THRESHOLD != 0.0
                        and adjusted_gamma > gamma
                        and (adjusted_gamma - gamma) < config.GAMMA_DIFFERENCE_THRESHOLD
                    )
                    and not (
                        config.GAMMA_DIFFERENCE_THRESHOLD != 0.0
                        and adjusted_gamma < gamma
                        and (gamma - adjusted_gamma) < config.GAMMA_DIFFERENCE_THRESHOLD
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

            if config.BRIGHTNESS_ADAPTATION:
                # Clamp to min/max values
                luma = clamp(luma, config.MIN_BRIGHTNESS, config.MAX_BRIGHTNESS)

                change_in_luma = abs(luma - brightness)

                # Skip if the luma is same as current monitor's brightness
                if (
                    luma == brightness
                    or change_in_luma <= config.LUMA_DIFFERENCE_THRESHOLD
                ):
                    print(" ...Skipping brightness adjustment... ")
                    continue

                if config.EXPERIMENTAL_BRIGHTNESS_ADAPTIVE_INCREMENTS:
                    # Adaptive increments
                    diff_for_instant = 50

                    # all divisible by 8
                    diff_for_2x_interval = 16
                    diff_for_3x_interval = 24
                    diff_for_4x_interval = 32
                    diff_for_5x_interval = 40
                    diff_for_6x_interval = 48

                    if change_in_luma == 1:
                        with monitor:
                            monitor.set_luminance(luma)
                    elif change_in_luma >= diff_for_instant:
                        print(f"Too much change in luminance: {change_in_luma}")
                        with monitor:
                            monitor.set_luminance(luma)
                    elif change_in_luma > diff_for_2x_interval:
                        print(f"Sudden change in luminance: {change_in_luma}")
                        with monitor:
                            fade_brightness(
                                monitor=monitor,
                                finish=luma,
                                start=brightness,
                                interval=config.BRIGHTNESS_ADJUSTMENT_INTERVAL,
                                increment=2,
                            )
                    elif change_in_luma > diff_for_3x_interval:
                        print(f"Sudden change in luminance: {change_in_luma}")
                        with monitor:
                            fade_brightness(
                                monitor=monitor,
                                finish=luma,
                                start=brightness,
                                interval=config.BRIGHTNESS_ADJUSTMENT_INTERVAL,
                                increment=3,
                            )
                    elif change_in_luma > diff_for_4x_interval:
                        print(f"Sudden change in luminance: {change_in_luma}")
                        with monitor:
                            fade_brightness(
                                monitor=monitor,
                                finish=luma,
                                start=brightness,
                                interval=config.BRIGHTNESS_ADJUSTMENT_INTERVAL,
                                increment=4,
                            )
                    elif change_in_luma > diff_for_5x_interval:
                        print(f"Sudden change in luminance: {change_in_luma}")
                        with monitor:
                            fade_brightness(
                                monitor=monitor,
                                finish=luma,
                                start=brightness,
                                interval=config.BRIGHTNESS_ADJUSTMENT_INTERVAL,
                                increment=5,
                            )
                    elif change_in_luma > diff_for_6x_interval:
                        print(f"Sudden change in luminance: {change_in_luma}")
                        with monitor:
                            fade_brightness(
                                monitor=monitor,
                                finish=luma,
                                start=brightness,
                                interval=config.BRIGHTNESS_ADJUSTMENT_INTERVAL,
                                increment=6,
                            )
                    else:
                        # Normal (non-adaptive increments) brightness adjustments
                        if config.BRIGHTNESS_INSTANT_ADJUSTMENTS:
                            with monitor:
                                monitor.set_luminance(luma)
                        else:
                            diff = (
                                (luma - brightness)
                                if luma > brightness
                                else (brightness - luma)
                            )
                            increment = clamp(diff, 1, 2)

                            with monitor:
                                fade_brightness(
                                    monitor=monitor,
                                    finish=luma,
                                    start=brightness,
                                    interval=config.BRIGHTNESS_ADJUSTMENT_INTERVAL,
                                    increment=increment,
                                )

                        print(f"Brightness: {luma} (from {brightness})")
                else:
                    # Normal (non-adaptive increments) brightness adjustments
                    if config.BRIGHTNESS_INSTANT_ADJUSTMENTS:
                        with monitor:
                            monitor.set_luminance(luma)
                    else:
                        diff = (
                            (luma - brightness)
                            if luma > brightness
                            else (brightness - luma)
                        )
                        increment = clamp(diff, 1, 2)

                        with monitor:
                            fade_brightness(
                                monitor=monitor,
                                finish=luma,
                                start=brightness,
                                interval=config.BRIGHTNESS_ADJUSTMENT_INTERVAL,
                                increment=increment,
                            )

                    print(f"Brightness: {luma} (from {brightness})")

            brightness = luma

            if config.EXPERIMENTAL_CONTRAST_ADAPTATION:
                with monitor:
                    while True:
                        try:
                            contrast = monitor.get_contrast()
                            break
                        except VCPError:
                            continue

                # Work in progress
                # Naive implementation but helps in gradual adjustments of display's luminance
                # It also leads to less washed out (relaxing) colors in high luminance content
                # Blacks are not improved though in very low lumninance content
                average_contrast = 100 - luma

                # Clamp to min/max values
                average_contrast = clamp(luma, config.MIN_CONTRAST, config.MAX_CONTRAST)

                change_in_contrast = abs(average_contrast - contrast)

                # Skip if the average contrast is same as current monitor's contrast
                if (
                    average_contrast == contrast
                    or change_in_contrast <= config.CONTRAST_DIFFERENCE_THRESHOLD
                ):
                    print(" ...Skipping contrast adjustment... ")
                    continue

                with monitor:
                    fade_contrast(
                        monitor,
                        average_contrast,
                        contrast,
                        interval=config.CONTRAST_ADJUSTMENT_INTERVAL,
                    )

                print(f"Contrast: {average_contrast} (from {contrast})")

                contrast = average_contrast

    except KeyboardInterrupt:
        print("[!] Programm interrupted. Closing now... ")
        del camera

        if config.EXPERIMENTAL_GAMMA_RAMP_ADJUSTMENTS and not ReleaseDC(hdc):
            print("[!] Could not release the HDC")

        time.sleep(1)
