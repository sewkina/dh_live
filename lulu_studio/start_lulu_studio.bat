@echo off
chcp 65001 >nul
rem 噜噜工坊：第一次运行会建虚拟环境并装依赖，之后直接启动
cd /d %~dp0
if exist .venv\Scripts\python.exe goto :run
set PY=
where python >nul 2>nul && set PY=python
if "%PY%"=="" where py >nul 2>nul && set PY=py -3
if "%PY%"=="" if exist "%USERPROFILE%\anaconda3\python.exe" set PY="%USERPROFILE%\anaconda3\python.exe"
if "%PY%"=="" if exist "%USERPROFILE%\miniconda3\python.exe" set PY="%USERPROFILE%\miniconda3\python.exe"
if "%PY%"=="" (
  echo 没找到 Python，请先安装 Python 3.10 以上版本。
  goto :err
)
echo 第一次运行，正在建虚拟环境并安装依赖（约 30 MB）……
%PY% -m venv .venv || goto :err
.venv\Scripts\python.exe -m pip install -r requirements.txt || goto :err
:run
if "%LULU_STUDIO_PORT%"=="" set LULU_STUDIO_PORT=8020
echo.
echo 噜噜工坊已启动：本机打开 http://localhost:%LULU_STUDIO_PORT%
echo 同一网络的手机打开 http://电脑IP:%LULU_STUDIO_PORT%   关掉这个窗口就停止
echo.
.venv\Scripts\python.exe app.py
goto :eof
:err
echo 启动失败，请把上面的报错发给 Claude。
pause
