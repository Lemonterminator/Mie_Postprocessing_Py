@echo off
setlocal

cd /d "%~dp0.."

where pdflatex >nul 2>nul
if errorlevel 1 (
    echo ERROR: pdflatex was not found. Install MiKTeX and make sure its bin folder is on PATH.
    exit /b 1
)

where biber >nul 2>nul
if errorlevel 1 (
    echo ERROR: biber was not found. Install or update MiKTeX so biblatex can build references.
    exit /b 1
)

if not exist pdf mkdir pdf

call :build thesis
if errorlevel 1 exit /b 1

call :build thesis_zh
if errorlevel 1 exit /b 1

echo.
echo Built thesis.pdf and thesis_zh.pdf with the local Aalto thesis template to the pdf folder.
exit /b 0

:build
echo.
echo === Building %1.pdf ===
pdflatex -interaction=nonstopmode -file-line-error -output-directory=pdf latex\%1.tex
if errorlevel 1 exit /b 1
biber --output-directory=pdf pdf\%1
if errorlevel 1 exit /b 1
pdflatex -interaction=nonstopmode -file-line-error -output-directory=pdf latex\%1.tex
if errorlevel 1 exit /b 1
pdflatex -interaction=nonstopmode -file-line-error -output-directory=pdf latex\%1.tex
if errorlevel 1 exit /b 1
exit /b 0
