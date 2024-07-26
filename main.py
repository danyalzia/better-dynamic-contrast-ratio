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

import screen_brightness_control as sbc
from PIL import ImageGrab, Image
import win32gui
import numpy as np

CPU_MODE_FORCED = True

MIN_BRIGHTNESS = 10
MAX_BRIGHTNESS = 50

# Luminance algorithms: https://stackoverflow.com/questions/596216/formula-to-determine-perceived-brightness-of-rgb-color
try:
    if CPU_MODE_FORCED:
        raise ValueError("[!] CPU Mode is forced, will use CPU (slower) version.")

    import torch
    from torchvision.transforms import v2

    if not torch.cuda.is_available():
        raise ValueError("[!] CUDA is not available, will use CPU (slower) version.")

    print("Using PyTorch/CUDA... \n")

    def get_average_luminance(img):
        transforms = v2.Compose([v2.PILToTensor(), v2.Resize(8)])

        arr = transforms(img).permute(1, 2, 0).cuda()

        d = torch.tensor([2550.299, 2550.587, 1770.833]).cuda()

        total_num_sum = np.prod(arr.shape[:-1])
        luminance_total = (arr / d).sum().item()
        return luminance_total / total_num_sum

except (ModuleNotFoundError, ValueError) as err:
    print(err)

    print("Using CPU... \n")

    def get_average_luminance(img):
        base_width = 8

        img_sizex, img_sizey = float(img.size[0]), float(img.size[1])

        if img_sizex == 0.0:
            img_sizex = 1.0
            
        if img_sizey == 0.0:
            img_sizey = 1.0
            
        wpercent = base_width / img_sizex

        hsize = int((img_sizey * float(wpercent)))

        if hsize == 0:
            hsize = 1

        img = img.resize((base_width, hsize), Image.Resampling.LANCZOS)

        arr = np.array(img)
        total_num_sum = np.prod(arr.shape[:-1])
        luminance_total = (arr / [2550.299, 2550.587, 1770.833]).sum()
        return luminance_total / total_num_sum


if __name__ == "__main__":
    while True:
        while True:
            try:
                # Sometimes it can't retreive the current/foreground window when doing a lot of alt-tab operations
                bbox = win32gui.GetWindowRect(win32gui.GetForegroundWindow())
                break
            except:
                continue

        img = ImageGrab.grab(bbox)

        average_luminance = get_average_luminance(img) * 255

        try:
            luma = int(str(average_luminance)[:2])
        except ValueError:
            luma = 0

        brightness = sbc.get_brightness()[0]

        # Skip if the luma is same as current monitor's brightness
        if luma == brightness:
            continue
        
        if luma < MIN_BRIGHTNESS:
            sbc.fade_brightness(MIN_BRIGHTNESS, interval=0)
            print(f"Luma: {luma}. Clamping to max brightness: {MIN_BRIGHTNESS}")

        if luma > MAX_BRIGHTNESS:
            sbc.fade_brightness(MAX_BRIGHTNESS, interval=0)
            print(f"Luma: {luma}. Clamping to min brightness: {MAX_BRIGHTNESS}")

        if luma >= MIN_BRIGHTNESS and luma <= MAX_BRIGHTNESS:
            sbc.fade_brightness(luma, interval=0)
            print("Brightness: " + str(luma))
