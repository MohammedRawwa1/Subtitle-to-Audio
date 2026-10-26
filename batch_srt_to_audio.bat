@echo off
setlocal EnableExtensions EnableDelayedExpansion

pushd "%~dp0"
set "SRT_DIR=%~dp0srt"
set "OUT_DIR=%~dp0results"
if not exist "%SRT_DIR%\" (
    echo SRT folder not found: "%SRT_DIR%"
    popd
    exit /b 1
)

if not exist "%OUT_DIR%\" mkdir "%OUT_DIR%"
if errorlevel 1 (
    echo Could not create output folder: "%OUT_DIR%"
    popd
    exit /b 1
)

set "FOUND="
for %%F in ("%SRT_DIR%\*.srt") do (
    if exist "%%~fF" (
        set "FOUND=1"
        echo.
        echo Processing: %%~nxF
        python "%~dp0subtitle_to_audio.py" "%%~fF" --engine edge_tts --auto-voice --gender male --format m4a --max-rate 15 --max-speed 1.10 --overflow bleed --output-dir "%OUT_DIR%"
        if errorlevel 1 (
            echo Failed: %%~nxF
            popd
            exit /b 1
        )
    )
)

if not defined FOUND (
    echo No .srt files found in "%SRT_DIR%"
    popd
    exit /b 1
)

echo.
echo Finished processing all SRT files.
echo Output folder: "%OUT_DIR%"
popd
