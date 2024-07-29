import time
import screen_brightness_control as sbc
from monitorcontrol import get_monitors, VCPError


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

    print(
        f"set_luminance is {int((set_luminance_time / set_brigthness_time) * 100)} % faster"
    )

    sbc.set_brightness(20)


if __name__ == "__main__":
    average_luminance()
    set_brightness()
