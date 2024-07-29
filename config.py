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

# -----------
# -- Monitor
# -----------

# If two monitors are connected then by default the program will run on primary monitor
# Set it to 1 to run on external/second monitor
MONITOR_INDEX = 0
# VRR is not currently supported
# Program must be running at the current refresh rate of monitor otherwise it behaves badly
TARGET_FPS = 60

# -------------
# -- Algorithm
# -------------

# Forcing CPU for now because CPU (numpy based) luminance calculating algorithms are actually faster due to PyTorch/CUDA overhead
CPU_MODE_FORCED = True

# --------------
# -- Brightness
# --------------

BRIGHTNESS_ADAPTATION = True  # It's main functionality; you don't want to disable it :D
BRIGHTNESS_ADAPTIVE_INCREMENTS = True  # Recommended
BRIGHTNESS_INSTANT_ADJUSTMENTS = (
    False  # Only works when BRIGHTNESS_ADAPTIVE_INCREMENTS is False
)
BRIGHTNESS_ADJUSTMENT_INTERVAL = 0.1
LUMA_DIFFERENCE_THRESHOLD = 0.0  # Tolerance level
MIN_BRIGHTNESS = 0
MAX_BRIGHTNESS = 100

# ------------
# -- Contrast
# ------------

EXPERIMENTAL_CONTRAST_ADAPTATION = (
    False  # Work in Progress (Don't use it in combination of Gamma adjustments below)
)
CONTRAST_ADJUSTMENT_INTERVAL = 0.1
CONTRAST_DIFFERENCE_THRESHOLD = 0  # Tolerance level
MIN_CONTRAST = 0
MAX_CONTRAST = 100

# ---------
# -- Gamma
# ---------

EXPERIMENTAL_GAMMA_RAMP_ADJUSTMENTS = False  # Work in Progress (Don't use it in combination of Contrast adjustments above)
GAMMA_DIFFERENCE_THRESHOLD = 0.0  # Tolerance level
MIN_GAMMA = 0.85
MAX_GAMMA = 1.05

# --------
# -- Misc
# --------

BLOCKING = True  # Keep it enabled as disabling it causes a bunch of threads related bugs; needs a lot of testing
