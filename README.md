# Better Dynamic Contrast Ratio (DCR) - [WORK IN PROGRESS]

Software implementation of the **Dynamic Contrast Ratio (DCR)** technology (sometimes also known as **Advanced Contrast Ratio (ACR)**, **Smart Contrast Ratio (SCR)** and other similar names) that is found in many SDR monitors but often lack support for any kind of manual adjustments (such as the minimum and maximum allowed luminance, etc.) and would instantly transition to new brightness which can look distracting and nauseating thus limiting its usability when watching content or playing video games.

This tool makes DCR technology more viable to use on any kind of external SDR monitor with or without native DCR support as long as monitor's brightness (backlight) can be controlled from the OS side using DDC/CI protocol that is almost always available in HDMI supported monitors.

It can also be used to fake the "HDR" effects, but instead of the display making **Local Dimming** (Full Array or Edge-Lit) adjustments (as in the case of real HDR display), it will only be making **Global Dimming** (Direct-Lit) adjustments (along with content gamma adjustments to compensate for poor blacks visibility under low backlight and blinding highlights under high backlight) which is vastly inferior to real HDR but may still be convincing to your eyes.

<br>
<img src="docs/Dimming_Techniques.gif" alt="Backlight Dimming Technologies - Courtesy of Wikipedia" width="500">

## Requirements

Your monitor must support DDC/CI for backlight adjustments to happen. If it's supported, then make sure it is enabled in your monitor's OSD. Without software backlight adjustments support, you may still benefit from content adaptive gamma adjustments.

For content gamma adjustments to happen, Windows must have a default system-level gamma ramp generated either by selecting a color profile (in "Display settings") or making callibration using Windows callibration tool.

Only Windows is supported at the moment.

## Installation

Install Python 3.12.5: <https://www.python.org/ftp/python/3.12.5/python-3.12.5-amd64.exe>

1. Clone the repository: `git clone https://github.com/danyalzia/better-dynamic-contrast-ratio.git`
2. Navigate to the project directory: `cd better-dynamic-contrast-ratio`
3. Create Python virtual environment: `python -m venv .venv`
4. Install dependencies: `pip install -r requirements.txt`
5. Run the program `python main.py` (or use `run.bat` script)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
