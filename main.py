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

from functools import cache
import os
import dxcam
import math

import time
import numpy as np
from ctypes import Structure, windll, byref
from ctypes.wintypes import DWORD, HANDLE, BYTE, WCHAR, HDC

import config

# Luminance calculating algorithms: https://stackoverflow.com/questions/596216/formula-to-determine-perceived-brightness-of-rgb-color
@cache
def lum_to_0_100(luminance: float):
    return (luminance / 255) * 100

@cache
def sum_to_0_100(summed: float, x: int, y: int):
    luminance = summed / (x * y)
    return (luminance / 255) * 100

def get_average_luminance1(arr: np.ndarray):
    total_num_sum = np.prod(arr.shape[:-1])
    luminance_total = (arr / [2550.299, 2550.587, 1770.833]).sum()
    return (luminance_total / total_num_sum) * 255

# ITU BT.709
def get_average_luminance2(arr: np.ndarray):
    mean_rgb = arr.reshape(-1, 3).mean(axis=0)
    luminance = (mean_rgb * [0.2126, 0.7152, 0.0722]).sum()
    return lum_to_0_100(luminance)

# ITU BT.601
def get_average_luminance3(arr: np.ndarray):
    mean_rgb = arr.reshape(-1, 3).mean(axis=0)
    luminance = (mean_rgb * [0.299, 0.587, 0.114]).sum()
    return lum_to_0_100(luminance)

# Fastest method when the source is grayscale
# Will not work in RGB mode
def get_average_luminance4(arr: np.ndarray):
    return sum_to_0_100(arr.sum(), arr.shape[0], arr.shape[1])


class PhysicalMonitor(Structure):
    _fields_ = [("handle", HANDLE), ("description", WCHAR * 128)]


def get_primary_monitor_handle():
    monitor_HMONITOR = windll.user32.MonitorFromPoint(0, 0, 1)
    physical_monitors = (PhysicalMonitor * 1)()

    windll.dxva2.GetPhysicalMonitorsFromHMONITOR(monitor_HMONITOR, 1, physical_monitors)

    return physical_monitors[0].handle


def vcp_set_luminance(handle, value):
    windll.dxva2.SetVCPFeature(HANDLE(handle), BYTE(0x10), DWORD(value))


def vcp_get_luminance(handle):
    feature_current = DWORD()
    feature_max = DWORD()

    windll.dxva2.GetVCPFeatureAndVCPFeatureReply(
        HANDLE(handle),
        BYTE(0x10),
        None,
        byref(feature_current),
        byref(feature_max),
    )

    return feature_current.value


def fade_luminance(
    handle, finish: int, start: int, interval: float, increment: int = 1
):
    increment = abs(increment)
    if start > finish:
        increment = -increment

    next_change_start_time = time.time()
    for value in range(start, finish, increment):
        vcp_set_luminance(handle, value)

        next_change_start_time += interval
        sleep_time = next_change_start_time - time.time()

        if sleep_time > 0:
            time.sleep(sleep_time)

    return finish


def clamp(d, minValue, maxValue):
    t = max(d, minValue)
    return min(t, maxValue)


def get_default_gamma_ramp(GetDeviceGammaRamp, hdc):
    default_gamma_ramp = np.empty((3, 256), dtype=np.uint16)
    if not GetDeviceGammaRamp(hdc, default_gamma_ramp.ctypes):
        raise RuntimeError("Can't get default gamma ramp")

    return default_gamma_ramp


def save_gamma_ramp(ramp, filename):
    np.save(filename, ramp)


def load_gamma_ramp(filename):
    return np.load(filename)


def get_gamma_allowed_values(SetDeviceGammaRamp, hdc, gamma_ramp):
    supported_values = []

    # Check 0.5 - 1.5 range
    for value in range(50, 150 + 1):
        value = value / 100

        Scale = np.array([[value], [value], [value]], float)
        NewRamps = np.uint16(np.round(np.multiply(Scale, gamma_ramp)))

        if not SetDeviceGammaRamp(hdc, NewRamps.ctypes):  # 0-1 = False/True
            value += 0.01
        else:
            supported_values.append(value)

    return supported_values


def set_gamma(SetDeviceGammaRamp, hdc, gamma_ramp, value):
    Scale = np.array([[value], [value], [value]], float)
    NewRamps = np.uint16(np.round(np.multiply(Scale, gamma_ramp)))

    if not SetDeviceGammaRamp(hdc, NewRamps.ctypes):  # 0-1 = False/True
        ValueError(f"Unable to set Gamma to {value}")

    return value


def fade_gamma(
    SetDeviceGammaRamp,
    hdc,
    gamma_ramp,
    finish: float,
    start: float | None = None,
    interval: float = 0.01,
    increment: float = 0.01,
):
    finishInt = round(finish * 100)
    startInt = round(start * 100)
    incrementInt = round(abs(increment * 100))

    if startInt > finishInt:
        incrementInt = -incrementInt

    next_change_start_time = time.time()

    for valueInt in range(startInt, finishInt, incrementInt):

        value = valueInt / 100

        set_gamma(SetDeviceGammaRamp, hdc, gamma_ramp, value)

        next_change_start_time += interval
        sleep_time = next_change_start_time - time.time()

        if sleep_time > 0:
            time.sleep(sleep_time)

    return finish


def scale_number(unscaled, to_min, to_max, from_min, from_max):
    return (to_max - to_min) * (unscaled - from_min) / (from_max - from_min) + to_min


def scale_list(l, to_min, to_max):
    return [scale_number(i, to_min, to_max, min(l), max(l)) for i in l]


def fade_gamma_luminance_combined(
    SetDeviceGammaRamp,
    hdc,
    gamma_ramp,
    handle,
    mid_point: float,
    gammaFinish: float,
    gammaStart: float,
    luminanceFinish: float,
    luminanceStart: float,
    interval: float = 0.01,
    increment: float = 0.01,
):
    finishInt = round(gammaFinish * 100)
    startInt = round(gammaStart * 100)
    incrementInt = round(abs(increment * 100))

    if startInt > finishInt:
        incrementInt = -incrementInt

    next_change_start_time = time.time()

    scaled = scale_list(
        list(range(startInt, finishInt, incrementInt)),
        luminanceFinish,
        luminanceStart,
    )

    if gammaFinish > (mid_point * 10):
        scaled = list(reversed(scaled))

    for n, valueInt in enumerate(range(startInt, finishInt, incrementInt)):

        value = valueInt / 100

        set_gamma(SetDeviceGammaRamp, hdc, gamma_ramp, value)

        luminanceInt = int(scaled[n])
        # print(f"L: {luminanceInt}, {luminanceFinish}, {luminanceStart}")
        vcp_set_luminance(handle, luminanceInt)

        next_change_start_time += interval
        sleep_time = next_change_start_time - time.time()

        if sleep_time > 0:
            time.sleep(sleep_time)

    return gammaFinish, luminanceFinish


@cache
def get_adjusted_gamma(log_mid_point: float, mean_luma: float):
    return log_mid_point / math.log(mean_luma)


if __name__ == "__main__":
    if config.GAMMA_RAMP_ADJUSTMENTS:
        GetDC = windll.user32.GetDC
        SetDeviceGammaRamp = windll.gdi32.SetDeviceGammaRamp
        GetDeviceGammaRamp = windll.gdi32.GetDeviceGammaRamp

        hdc = HDC(GetDC(None))
        if not hdc:
            raise RuntimeError("No HDC")

        if os.path.exists("defaultgamma.npy"):
            default_gamma_ramp = load_gamma_ramp("defaultgamma.npy")
        else:
            default_gamma_ramp = get_default_gamma_ramp(GetDeviceGammaRamp, hdc)
            save_gamma_ramp(default_gamma_ramp, "defaultgamma")

        supported_values = get_gamma_allowed_values(
            SetDeviceGammaRamp, hdc, default_gamma_ramp
        )
        min_gamma_allowed, max_gamma_allowed = min(supported_values), max(
            supported_values
        )

        mid_point = (
            ((min_gamma_allowed + max_gamma_allowed) / 2) / 10
        ) + config.MID_POINT_BIAS

        log_mid_point = math.log(mid_point * 255)

        luminance_map = {
            y: int(x)
            for x, y in zip(
                scale_list(
                    list(range(101)),
                    config.MIN_DESIRED_MONITOR_LUMINANCE,
                    config.MAX_DESIRED_MONITOR_LUMINANCE,
                ),
                list(range(101)),
            )
        }

        gamma_map = {
            y: round(x, 2)
            for x, y in zip(
                scale_list(
                    [
                        (x / 100)
                        for x in range(
                            round(config.MIN_DESIRED_GAMMA * 100),
                            round(config.MAX_DESIRED_GAMMA * 100),
                        )
                    ],
                    min_gamma_allowed,
                    max_gamma_allowed,
                ),
                supported_values,
            )
        }

        print(f"Min Desired Gamma: {config.MIN_DESIRED_GAMMA}")
        print(f"Max Desired Gamma: {config.MAX_DESIRED_GAMMA}")
        print(f"Gamma Values: {','.join(str(v) for v in gamma_map.values())}")
        print(f"Mid Point: {mid_point}")
        print(f"Log Mid Point: {log_mid_point}")
        print(
            f"Min Monitor's Desired Luminance: {config.MIN_DESIRED_MONITOR_LUMINANCE}"
        )
        print(
            f"Max Monitor's Desired Luminance: {config.MAX_DESIRED_MONITOR_LUMINANCE}"
        )
        print(
            f"Monitor's Luminance Values: {','.join(str(v) for v in luminance_map.values())}"
        )
        print()

        # Start with a darkened image
        gamma = min_gamma_allowed
        set_gamma(SetDeviceGammaRamp, hdc, default_gamma_ramp, gamma)

        # Ignore annoying divide by zero and overflow warnings
        np.seterr(divide="ignore", over="ignore")

    handle = get_primary_monitor_handle()

    default_monitor_luminance = vcp_get_luminance(handle)
    print(f"Default Monitor's Luminance: {default_monitor_luminance}")
    print()

    try:
        camera = dxcam.create(output_idx=config.MONITOR_INDEX, output_color="GRAY")
    except IndexError as e:
        raise RuntimeError(
            f"Monitor at index {config.MONITOR_INDEX} is not available, please try a different value."
        ) from e

    camera.start(target_fps=config.TARGET_FPS)

    try:
        while True:
            monitor_luminance = vcp_get_luminance(handle)

            frame = camera.get_latest_frame()

            try:
                mean_luma = get_average_luminance4(frame)
            except ValueError as e:
                raise RuntimeError(
                    "[!] Cannot calculate average luminance of the content."
                ) from e

            meta_info = ""

            # Gamma ramp adjustments work together with monitor's luminance adjustments to make the perceived lumniance changes less aggressive
            if config.GAMMA_RAMP_ADJUSTMENTS:
                # This method requires frame to be in grayscale (which we already have)

                # All pixels are black (i.e., value of 0)
                # e.g., Black fullscreen wallpaper
                if mean_luma == 0:
                    adjusted_gamma = max_gamma_allowed
                # All pixels are black (i.e., value of 255)
                # e.g., White fullscreen wallpaper
                elif mean_luma == 255:
                    adjusted_gamma = min_gamma_allowed
                else:
                    adjusted_gamma = get_adjusted_gamma(log_mid_point, mean_luma)

                # Gamma correction is applied to current frame now
                frame = np.power(frame, adjusted_gamma).clip(0, 255).astype(np.uint8)

                if config.MONITOR_LUMINANCE_ADJUSTMENTS:
                    mean_luma = get_average_luminance4(frame)

                    # Using gamma correction information to decide the levels of monitor's luminance
                    # Increase monitor's luminance when gamma is lower than half of gamma to compensate the lower perceived luminance for the darker colors
                    if adjusted_gamma < (mid_point * 10):
                        mean_luma = 100 - mean_luma
                        # Small exposure bias when gamma (i.e., content luminance) has been adjusted to increase the perceived dynamic range
                        mean_luma = mean_luma * 1.15
                    # Do the opposite for brighter/washed out colors
                    else:
                        mean_luma = mean_luma * 0.85

                    mean_luma = round(mean_luma)
                    adjusted_monitor_luminance = luminance_map[mean_luma]

                adjusted_gamma = clamp(
                    round(adjusted_gamma, 2), min_gamma_allowed, max_gamma_allowed
                )
                adjusted_gamma = gamma_map[adjusted_gamma]

                change_in_gamma = round(abs(adjusted_gamma - gamma), 2)
                if (
                    adjusted_gamma != gamma
                    and change_in_gamma > config.GAMMA_DIFFERENCE_THRESHOLD
                ):
                    if change_in_gamma == 0.01:
                        meta_info = f"Minor change in gamma: {change_in_gamma}"
                        if config.MONITOR_LUMINANCE_ADJUSTMENTS:
                            set_gamma(
                                SetDeviceGammaRamp,
                                hdc,
                                default_gamma_ramp,
                                adjusted_gamma,
                            )
                            vcp_set_luminance(handle, adjusted_monitor_luminance)
                        else:
                            set_gamma(
                                SetDeviceGammaRamp,
                                hdc,
                                default_gamma_ramp,
                                adjusted_gamma,
                            )

                    elif change_in_gamma >= 0.10 and change_in_gamma < 0.20:
                        meta_info = f"Small change in gamma: {change_in_gamma}"
                        if config.MONITOR_LUMINANCE_ADJUSTMENTS:
                            fade_gamma_luminance_combined(
                                SetDeviceGammaRamp,
                                hdc,
                                default_gamma_ramp,
                                handle,
                                mid_point,
                                gammaFinish=adjusted_gamma,
                                gammaStart=gamma,
                                luminanceFinish=adjusted_monitor_luminance,
                                luminanceStart=monitor_luminance,
                                interval=0.01,
                                increment=0.02,
                            )
                        else:
                            fade_gamma(
                                SetDeviceGammaRamp,
                                hdc,
                                default_gamma_ramp,
                                finish=adjusted_gamma,
                                start=gamma,
                                interval=0.01,
                                increment=0.02,
                            )

                    elif change_in_gamma >= 0.20 and change_in_gamma < 0.30:
                        meta_info = f"Moderate change in gamma: {change_in_gamma}"
                        if config.MONITOR_LUMINANCE_ADJUSTMENTS:
                            fade_gamma_luminance_combined(
                                SetDeviceGammaRamp,
                                hdc,
                                default_gamma_ramp,
                                handle,
                                mid_point,
                                gammaFinish=adjusted_gamma,
                                gammaStart=gamma,
                                luminanceFinish=adjusted_monitor_luminance,
                                luminanceStart=monitor_luminance,
                                interval=0.01,
                                increment=0.03,
                            )
                        else:
                            fade_gamma(
                                SetDeviceGammaRamp,
                                hdc,
                                default_gamma_ramp,
                                finish=adjusted_gamma,
                                start=gamma,
                                interval=0.01,
                                increment=0.03,
                            )
                    elif change_in_gamma >= 0.30 and change_in_gamma < 0.40:
                        meta_info = f"Large change in gamma: {change_in_gamma}"
                        if config.MONITOR_LUMINANCE_ADJUSTMENTS:
                            fade_gamma_luminance_combined(
                                SetDeviceGammaRamp,
                                hdc,
                                default_gamma_ramp,
                                handle,
                                mid_point,
                                gammaFinish=adjusted_gamma,
                                gammaStart=gamma,
                                luminanceFinish=adjusted_monitor_luminance,
                                luminanceStart=monitor_luminance,
                                interval=0.01,
                                increment=0.04,
                            )
                        else:
                            fade_gamma(
                                SetDeviceGammaRamp,
                                hdc,
                                default_gamma_ramp,
                                finish=adjusted_gamma,
                                start=gamma,
                                interval=0.01,
                                increment=0.04,
                            )
                    elif change_in_gamma >= 0.40 and change_in_gamma < 0.50:
                        meta_info = f"Huge change in gamma: {change_in_gamma} (Note: It can be avoided)"
                        if config.MONITOR_LUMINANCE_ADJUSTMENTS:
                            fade_gamma_luminance_combined(
                                SetDeviceGammaRamp,
                                hdc,
                                default_gamma_ramp,
                                handle,
                                mid_point,
                                gammaFinish=adjusted_gamma,
                                gammaStart=gamma,
                                luminanceFinish=adjusted_monitor_luminance,
                                luminanceStart=monitor_luminance,
                                interval=0.01,
                                increment=0.05,
                            )
                        else:
                            fade_gamma(
                                SetDeviceGammaRamp,
                                hdc,
                                default_gamma_ramp,
                                finish=adjusted_gamma,
                                start=gamma,
                                interval=0.01,
                                increment=0.05,
                            )
                    elif change_in_gamma >= 0.50 and change_in_gamma < 0.60:
                        meta_info = f"Sudden change in gamma: {change_in_gamma} (Note: It must be avoided)"
                        if config.MONITOR_LUMINANCE_ADJUSTMENTS:
                            fade_gamma_luminance_combined(
                                SetDeviceGammaRamp,
                                hdc,
                                default_gamma_ramp,
                                handle,
                                mid_point,
                                gammaFinish=adjusted_gamma,
                                gammaStart=gamma,
                                luminanceFinish=adjusted_monitor_luminance,
                                luminanceStart=monitor_luminance,
                                interval=0.01,
                                increment=0.06,
                            )
                        else:
                            fade_gamma(
                                SetDeviceGammaRamp,
                                hdc,
                                default_gamma_ramp,
                                finish=adjusted_gamma,
                                start=gamma,
                                interval=0.01,
                                increment=0.06,
                            )
                    else:
                        if config.MONITOR_LUMINANCE_ADJUSTMENTS:
                            fade_gamma_luminance_combined(
                                SetDeviceGammaRamp,
                                hdc,
                                default_gamma_ramp,
                                handle,
                                mid_point,
                                gammaFinish=adjusted_gamma,
                                gammaStart=gamma,
                                luminanceFinish=adjusted_monitor_luminance,
                                luminanceStart=monitor_luminance,
                                interval=0.01,
                                increment=0.01,
                            )
                        else:
                            fade_gamma(
                                SetDeviceGammaRamp,
                                hdc,
                                default_gamma_ramp,
                                finish=adjusted_gamma,
                                start=gamma,
                                interval=0.01,
                                increment=0.01,
                            )

                    print(
                        f"Gamma: {adjusted_gamma} (from {gamma}) {f'[-- {meta_info} --]' if meta_info else ''}"
                    )
                    gamma = adjusted_gamma

                    if config.MONITOR_LUMINANCE_ADJUSTMENTS:
                        print(
                            f"Luminance: {adjusted_monitor_luminance} (from {monitor_luminance})"
                        )
                        monitor_luminance = vcp_get_luminance(handle)

            meta_info = ""

            if (
                config.MONITOR_LUMINANCE_ADJUSTMENTS
                and not config.GAMMA_RAMP_ADJUSTMENTS
            ):
                mean_luma = round(mean_luma)
                adjusted_monitor_luminance = luminance_map[mean_luma]

                change_in_luma = abs(adjusted_monitor_luminance - monitor_luminance)

                # Skip if the adjusted luminance is same as current monitor's luminance
                if (
                    adjusted_monitor_luminance == monitor_luminance
                    or change_in_luma <= config.LUMA_DIFFERENCE_THRESHOLD
                ):
                    continue

                if config.MONITOR_LUMINANCE_INSTANT_ADJUSTMENTS:
                    vcp_set_luminance(handle, adjusted_monitor_luminance)
                else:
                    # Adaptive increments
                    diff_for_instant = 50

                    # all divisible by 8
                    diff_for_2x_interval = 16
                    diff_for_3x_interval = 24
                    diff_for_4x_interval = 32
                    diff_for_5x_interval = 40
                    diff_for_6x_interval = 48

                    if change_in_luma == 1:
                        vcp_set_luminance(handle, adjusted_monitor_luminance)
                    elif change_in_luma >= diff_for_instant:
                        meta_info = f"Sudden change in luminance: {change_in_luma} (Note: It must be avoided)"
                        vcp_set_luminance(handle, adjusted_monitor_luminance)
                    elif change_in_luma > diff_for_2x_interval:
                        meta_info = f"Minor change in luminance: {change_in_luma}"
                        fade_luminance(
                            handle=handle,
                            finish=adjusted_monitor_luminance,
                            start=monitor_luminance,
                            interval=0.1,
                            increment=2,
                        )
                    elif change_in_luma > diff_for_3x_interval:
                        meta_info = f"Moderate change in luminance: {change_in_luma}"
                        fade_luminance(
                            handle=handle,
                            finish=adjusted_monitor_luminance,
                            start=monitor_luminance,
                            interval=0.1,
                            increment=3,
                        )
                    elif change_in_luma > diff_for_4x_interval:
                        meta_info = f"Large change in luminance: {change_in_luma}"
                        fade_luminance(
                            handle=handle,
                            finish=adjusted_monitor_luminance,
                            start=monitor_luminance,
                            interval=0.1,
                            increment=4,
                        )
                    elif change_in_luma > diff_for_5x_interval:
                        meta_info = f"Very large change in luminance: {change_in_luma} (Note: It can be avoided)"
                        fade_luminance(
                            handle=handle,
                            finish=adjusted_monitor_luminance,
                            start=monitor_luminance,
                            interval=0.1,
                            increment=5,
                        )
                    elif change_in_luma > diff_for_6x_interval:
                        meta_info = f"Huge change in luminance: {change_in_luma} (Note: It must be avoided)"
                        fade_luminance(
                            handle=handle,
                            finish=adjusted_monitor_luminance,
                            start=monitor_luminance,
                            interval=0.0,
                            increment=6,
                        )
                    else:
                        diff = (
                            (adjusted_monitor_luminance - monitor_luminance)
                            if adjusted_monitor_luminance > monitor_luminance
                            else (monitor_luminance - adjusted_monitor_luminance)
                        )
                        increment = clamp(diff, 1, 2)

                        fade_luminance(
                            handle=handle,
                            finish=adjusted_monitor_luminance,
                            start=monitor_luminance,
                            interval=0.1,
                            increment=increment,
                        )

                print(
                    f"Luminance: {adjusted_monitor_luminance} (from {monitor_luminance}) {f'[-- {meta_info} --]' if meta_info else ''}"
                )
                monitor_luminance = vcp_get_luminance(handle)

    except KeyboardInterrupt:
        print("\n[!] Program is interrupted.\n")
        del camera

        if config.GAMMA_RAMP_ADJUSTMENTS:
            ReleaseDC = windll.user32.ReleaseDC
            ReleaseDC(hdc)

        print("[!] Setting to default values... \n")

        # Gamma
        if config.GAMMA_RAMP_ADJUSTMENTS:
            default_gamma_ramp = load_gamma_ramp("defaultgamma.npy")
            set_gamma(SetDeviceGammaRamp, hdc, default_gamma_ramp, 1.0)

        # Luminance
        handle = get_primary_monitor_handle()
        vcp_set_luminance(handle, default_monitor_luminance)

        print("[!] Closing... \n")

        time.sleep(1)
