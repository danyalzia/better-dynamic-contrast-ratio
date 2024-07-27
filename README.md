# Better Dynamic Contrast Ratio (DCR)

Software implementation of the **Dynamic Contrast Ratio (DCR)** technology (sometimes also known as **Advanced Contrast Ratio (ACR)**, **Smart Contrast Ratio (SCR)** and other similar names) that is usually found in SDR monitors but lack any kind of manual adjustments to minimum and maximum allowed backlight brightness/luminance and can go between zero brightness to full brightness instantly which can look distracting and blinding thus limiting its usability when watching content or playing video games.

This tool makes DCR technology more viable to use on any kind of SDR monitor with or without native DCR support as long as monitor's brightness (backlight) can be controlled from the OS side (at the moment only Windows is supported) using DDC/CI that is almost always available in HDMI supported monitors.

It can also be used to fake the "HDR" effects, but instead of Local Dimming (Full Array or Edge-Lit) adjustments (as in the case of real HDR display), it will only be Global Dimming (Direct-Lit) adjustments which is vastly inferior to real HDR but may still be convincing to your eyes.

It is hardware accelerated meaning your GPU will be used for calculating average luminance of the content on your screen. Currently only Nvidia's GPUs are supported that can run CUDA 12.4 otherwise it will run on your CPU (which will be a bit slower).

### Installation

Install Python 3.10.11: https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe

Create Python virtual environment:
* python -m venv .venv

Install dependencies:
* pip install -r requirements.txt
