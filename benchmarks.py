import numpy as np
import time
import cv2
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
from pixel_forge import Capture, enumerate_monitors
import win32gui
import win32ui
import win32con
from PIL import ImageGrab
import dxcam
from main import (
    get_average_luminance1,
    get_average_luminance2,
    get_average_luminance3,
    get_average_luminance4,
)
import dxcam
from main import get_average_luminance4
import mss
import screen_brightness_control as sbc


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


def average_luminance():

    camera = dxcam.create(output_idx=0, output_color="GRAY")
    camera.start(target_fps=60)
    frame = camera.get_latest_frame()

    samples = 60

    current_time = time.time()
    for _ in range(samples):
        luma = get_average_luminance1(frame)
    print(f"Luma1: {luma} ------------- took {time.time() - current_time} seconds")

    current_time = time.time()
    for _ in range(samples):
        luma = get_average_luminance2(frame)
    print(f"Luma2: {luma} ------------- took {time.time() - current_time} seconds")

    current_time = time.time()
    for _ in range(samples):
        luma = get_average_luminance3(frame)
    print(f"Luma3: {luma} ------------- took {time.time() - current_time} seconds")

    current_time = time.time()
    for _ in range(samples):
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


def win32_frame(w, h):
    hwnd = win32gui.GetActiveWindow()

    wDC = win32gui.GetWindowDC(hwnd)
    dcObj = win32ui.CreateDCFromHandle(wDC)
    cDC = dcObj.CreateCompatibleDC()
    dataBitMap = win32ui.CreateBitmap()
    dataBitMap.CreateCompatibleBitmap(dcObj, w, h)
    cDC.SelectObject(dataBitMap)
    cDC.BitBlt((0, 0), (w, h), dcObj, (0, 0), win32con.SRCCOPY)

    signedIntsArray = dataBitMap.GetBitmapBits(True)
    frame = np.frombuffer(signedIntsArray, dtype="uint8")
    frame.shape = (h, w, 4)
    return hwnd, wDC, dcObj, cDC, dataBitMap, frame


def clean_win32(hwnd, wDC, dcObj, cDC, dataBitMap):
    dcObj.DeleteDC()
    cDC.DeleteDC()
    win32gui.ReleaseDC(hwnd, wDC)
    win32gui.DeleteObject(dataBitMap.GetHandle())


def frame_capture():

    samples = 60

    # camera = dxcam.create(output_idx=0, output_color="GRAY")
    # camera.start(target_fps=60)

    # current_time = time.perf_counter()
    # for _ in range(samples):
    #     frame = camera.get_latest_frame()
    #     luma = get_average_luminance4(frame)
    # print(
    #     f"Luma: {luma} ------------- took {time.perf_counter() - current_time} seconds"
    # )

    current_time = time.perf_counter()
    for _ in range(samples):
        img = ImageGrab.grab()
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        luma = get_average_luminance4(img)
    print(
        f"Luma: {luma} ------------- took {time.perf_counter() - current_time} seconds"
    )

    sct = mss.mss()
    monitor = sct.monitors[1]

    current_time = time.perf_counter()
    for _ in range(samples):
        img = sct.grab(monitor)
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        luma = get_average_luminance4(img)
    print(
        f"Luma: {luma} ------------- took {time.perf_counter() - current_time} seconds"
    )
    sct.close()

    current_time = time.perf_counter()
    for _ in range(samples):
        hwnd, wDC, dcObj, cDC, dataBitMap, frame = win32_frame(1920, 1200)
        img = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2GRAY)
        luma = get_average_luminance4(img)
        clean_win32(hwnd, wDC, dcObj, cDC, dataBitMap)
    print(
        f"Luma: {luma} ------------- took {time.perf_counter() - current_time} seconds"
    )

    camera = Capture()
    m = enumerate_monitors()[0]
    camera.start(m)
    
    current_time = time.perf_counter()
    for _ in range(samples):
        frame = camera.frame()
        img = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2GRAY)
        luma = get_average_luminance4(img)
    print(
        f"Luma: {luma} ------------- took {time.perf_counter() - current_time} seconds"
    )
    
    from PIL import Image
    
    current_time = time.perf_counter()
    for _ in range(samples):
        frame = camera.frame()
        img = np.array(Image.fromarray(frame).convert('L')).astype(np.uint)
        luma = get_average_luminance4(img)
    print(
        f"Luma: {luma} ------------- took {time.perf_counter() - current_time} seconds"
    )
    
    current_time = time.perf_counter()
    for _ in range(samples):
        frame = camera.frame()
        luma = get_average_luminance4(img)
    print(
        f"Luma: {luma} ------------- took {time.perf_counter() - current_time} seconds"
    )

if __name__ == "__main__":
    average_luminance()
    set_brightness()
    frame_capture()
