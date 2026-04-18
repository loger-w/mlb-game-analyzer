@echo off
set PYTHON=C:\Users\Loger\AppData\Local\Programs\Python\Python312\python.exe
set SCRIPT=C:\Users\Loger\.claude\skills\mlb-game-analyzer\scripts\fetch_odds.py
set TASKNAME=MLB_FetchOdds_Every4Hours

schtasks /Delete /TN "%TASKNAME%" /F 2>nul

schtasks /Create /TN "%TASKNAME%" /TR "\"%PYTHON%\" \"%SCRIPT%\"" /SC HOURLY /MO 4 /ST 00:00 /F

if %errorlevel%==0 (
    echo Task created successfully.
    schtasks /Query /TN "%TASKNAME%" /FO LIST
) else (
    echo Failed to create task.
)
