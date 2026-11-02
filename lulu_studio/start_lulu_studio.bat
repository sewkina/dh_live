@echo off
chcp 65001 >nul
rem 噜噜工坊：第一次运行会建虚拟环境并装依赖，之后直接启动
cd /d %~dp0
if not exist .venv\Scripts\python.exe (
  echo 第一次运行，正在建虚拟环境并安装依赖……
  python -m venv .venv || goto :err
  .venv\Scripts\python.exe -m pip install -r requirements.txt || goto :err
)
if "%LULU_STUDIO_PORT%"=="" set LULU_STUDIO_PORT=8020
echo 噜噜工坊已启动：本机打开 http://localhost:%LULU_STUDIO_PORT%  ，同一网络的手机打开 http://电脑IP:%LULU_STUDIO_PORT%
.venv\Scripts\python.exe app.py
goto :eof
:err
echo 启动失败，请把上面的报错发给 Claude。
pause
