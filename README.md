# Better Dynamic Contrast Ratio (DCR)

Software implementation of the Dynamic Contrast Ratio (DCR) technology that is usually found in SDR monitors but lack any kind of manual adjustments to min and max brightness/luminance and can go between zero brightness to max brightness instantly which can look distracting and blinding thus limiting its usability. This tool makes that technology more viable to use on any kind of SDR monitor with our without native DCR support as long as monitor's brightness can be controlled from OS side (only Windows is supported at the moment). It is hardware accelerated meaning your GPU will be used for calculating average luminance of content on your screen. Currently only Nvidia's GPUs are supported that can run CUDA 12.4 otherwise it will run on your CPU (which will be much slower).

### Installation

Install Python 3.10.11: https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe

Create Python virtual environment:
* python -m venv .venv

Install dependencies:
* pip install -r requirements.txt
