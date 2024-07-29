import time
import screen_brightness_control as sbc
from monitorcontrol import get_monitors, VCPError
import ctypes
from ctypes.wintypes import (
    DWORD,
    RECT,
    BOOL,
    HMONITOR,
    HDC,
    LPARAM,
    HANDLE,
    BYTE,
    WCHAR,
)


class PhysicalMonitor(ctypes.Structure):
    _fields_ = [("handle", HANDLE), ("description", WCHAR * 128)]


def get_monitor_handle(index=0):
    num_physical = DWORD()
    hmonitors = []

    def _callback(hmonitor, hdc, lprect, lparam):
        hmonitors.append(HMONITOR(hmonitor))
        del hmonitor, hdc, lprect, lparam
        return True  # continue enumeration

    MONITORENUMPROC = ctypes.WINFUNCTYPE(
        BOOL, HMONITOR, HDC, ctypes.POINTER(RECT), LPARAM
    )
    callback = MONITORENUMPROC(_callback)
    ctypes.windll.user32.EnumDisplayMonitors(0, 0, callback, 0)
    ctypes.windll.dxva2.GetNumberOfPhysicalMonitorsFromHMONITOR(
        hmonitors[index], ctypes.byref(num_physical)
    )

    if num_physical.value == 0:
        raise VCPError("no physical monitor found")

    physical_monitors = (PhysicalMonitor * num_physical.value)()

    if not ctypes.windll.dxva2.GetPhysicalMonitorsFromHMONITOR(
        hmonitors[index], num_physical.value, physical_monitors
    ):
        raise VCPError(
            f"Call to GetPhysicalMonitorsFromHMONITOR failed: {ctypes.FormatError()}"
        )

    handle = physical_monitors[index].handle

    return handle


def get_primary_monitor_handle():
    monitor_HMONITOR = ctypes.windll.user32.MonitorFromPoint(0, 0, 1)
    physical_monitors = (PhysicalMonitor * 1)()

    ctypes.windll.dxva2.GetPhysicalMonitorsFromHMONITOR(
        monitor_HMONITOR, 1, physical_monitors
    )

    return physical_monitors[0].handle


def raw_set_brightness(handle, value):
    ctypes.windll.dxva2.SetVCPFeature(HANDLE(handle), BYTE(0x10), DWORD(value))


def raw_get_brightness(handle):
    feature_current = DWORD()
    feature_max = DWORD()

    ctypes.windll.dxva2.GetVCPFeatureAndVCPFeatureReply(
        HANDLE(handle),
        BYTE(0x10),
        None,
        ctypes.byref(feature_current),
        ctypes.byref(feature_max),
    )

    return feature_current.value


def average_luminance():
    import dxcam
    from main import (
        get_average_luminance1,
        get_average_luminance2,
        get_average_luminance3,
        get_average_luminance4,
    )

    camera = dxcam.create(output_idx=0, output_color="GRAY")
    camera.start(target_fps=60)
    frame = camera.get_latest_frame()

    current_time = time.time()
    luma = get_average_luminance1(frame)
    print(f"Luma1: {luma} ------------- took {time.time() - current_time} seconds")

    current_time = time.time()
    luma = get_average_luminance2(frame)
    print(f"Luma2: {luma} ------------- took {time.time() - current_time} seconds")

    current_time = time.time()
    luma = get_average_luminance3(frame)
    print(f"Luma3: {luma} ------------- took {time.time() - current_time} seconds")

    current_time = time.time()
    luma = get_average_luminance4(frame)
    print(f"Luma4: {luma} ------------- took {time.time() - current_time} seconds")

    del camera


def set_brightness():
    monitor = get_monitors()[0]

    sbc.set_brightness(0)

    current_time = time.time()
    for i in range(0, 100 + 10, 10):
        sbc.set_brightness(i)

    set_brigthness_time = time.time() - current_time
    print(f"set_brightness ------------- took {set_brigthness_time} seconds")

    sbc.set_brightness(0)

    current_time = time.time()
    with monitor:
        while True:
            try:
                for i in range(0, 100 + 10, 10):
                    monitor.set_luminance(i)
                break
            except VCPError:
                continue

    set_luminance_time = time.time() - current_time
    print(f"set_luminance ------------- took {set_luminance_time} seconds")

    sbc.set_brightness(0)

    current_time = time.time()
    handle = get_monitor_handle()

    for i in range(0, 100 + 10, 10):
        raw_set_brightness(handle, i)

    set_raw_brightness_time = time.time() - current_time
    print(f"raw_set_brightness ------------- took {set_raw_brightness_time} seconds")

    sbc.set_brightness(0)

    current_time = time.time()
    handle = get_primary_monitor_handle()

    for i in range(0, 100 + 10, 10):
        raw_set_brightness(handle, i)

    set_raw_brightness_time2 = time.time() - current_time
    print(
        f"raw_set_brightness (get_primary_monitor_handle) ------------- took {set_raw_brightness_time2} seconds"
    )

    print(
        f"set_luminance is {100 - int((set_luminance_time / set_brigthness_time) * 100)} % faster than set_brigthness_time"
    )
    print(
        f"raw_set_brightness is {100 - int((set_raw_brightness_time / set_brigthness_time) * 100)} % faster than set_brigthness_time"
    )
    print(
        f"raw_set_brightness is {100 - int((set_raw_brightness_time / set_luminance_time) * 100)} % faster than set_luminance"
    )
    print(
        f"raw_set_brightness (get_primary_monitor_handle) is {100 - int((set_raw_brightness_time2 / set_raw_brightness_time) * 100)} % faster than set_raw_brightness_time"
    )

    sbc.set_brightness(20)


if __name__ == "__main__":
    average_luminance()
    set_brightness()
