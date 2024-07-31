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

# ------------------------------------
# -- Monitor's Luminance (Brightness)
# ------------------------------------

MONITOR_LUMINANCE_ADJUSTMENTS = True
MONITOR_LUMINANCE_INSTANT_ADJUSTMENTS = False

# Monitor's Brightness range (0 - 100) will be scaled to this range
# Shorter ranges provide better luminance stability but dramatic shifts
MIN_DESIRED_MONITOR_LUMINANCE = 10
MAX_DESIRED_MONITOR_LUMINANCE = 50

# ---------
# -- Gamma
# ---------

GAMMA_RAMP_ADJUSTMENTS = True
# Gamma range (0.60 - 1.20) will be scaled to this range
MIN_DESIRED_GAMMA = 0.60
MAX_DESIRED_GAMMA = 1.20

# --------
# -- Misc
# --------
LUMA_DIFFERENCE_THRESHOLD = 0.0  # Tolerance level
GAMMA_DIFFERENCE_THRESHOLD = 0.00  # Tolerance level

# Bias towards blacks/whites
# Negative value means bias towards blacks/shadows, so on average darker screen but less blown out highlights
# Positive value means bias towards blacks/shadows, so on average lighter screen but also less crushed blacks
# Increaes or decrease it slowly by 0.01 till the right balance is found
MID_POINT_BIAS = 0.0
