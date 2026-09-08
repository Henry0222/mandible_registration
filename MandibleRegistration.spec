# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules


open3d_datas, open3d_binaries, open3d_hiddenimports = collect_all("open3d")
application_datas = collect_data_files(
    "mandible_registration", includes=["assets/*.svg"]
)
application_datas.append(
    (
        str(Path("src/mandible_registration/assets/app_icon.ico").resolve()),
        "mandible_registration/assets",
    )
)
application_hiddenimports = collect_submodules("mandible_registration")
registration_hiddenimports = collect_submodules("auto_alignment")

analysis = Analysis(
    ["scripts/gui_entry.py"],
    pathex=["src"],
    binaries=open3d_binaries,
    datas=open3d_datas + application_datas,
    hiddenimports=(
        open3d_hiddenimports
        + application_hiddenimports
        + registration_hiddenimports
    ),
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "pytest",
        "IPython",
        "ipywidgets",
        "jupyter",
        "matplotlib",
        "pandas",
        "sklearn",
        "torch",
        "tkinter",
        "vtkmodules.all",
    ],
    noarchive=False,
    optimize=1,
)

# Do not bundle incompatible ICU DLLs accidentally discovered on PATH. Windows
# supplies the unversioned ICU shim expected by Qt 6.
incompatible_icu_names = {"icuuc.dll", "icudt78.dll"}
analysis.binaries = [
    entry
    for entry in analysis.binaries
    if Path(entry[0]).name.lower() not in incompatible_icu_names
]

pyz = PYZ(analysis.pure)

exe = EXE(
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="MandibleRegistration-v1.0.0",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon="src/mandible_registration/assets/app_icon.ico",
    version="scripts/version_info.txt",
)

collection = COLLECT(
    exe,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=False,
    name="MandibleRegistration-v1.0.0",
)
