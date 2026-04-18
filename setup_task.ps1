# 建立 MLB Odds 每 4 小時自動抓取的 Task Scheduler 工作
# 執行方式（系統管理員 PowerShell）：
#   powershell -ExecutionPolicy Bypass -File setup_task.ps1

$pythonPath = (Get-Command python).Source
$scriptPath = "C:\Users\Loger\.claude\skills\mlb-game-analyzer\scripts\fetch_odds.py"
$taskName   = "MLB_FetchOdds_Every4Hours"

# 移除舊的同名工作
Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue

# Action：執行 python fetch_odds.py
$action = New-ScheduledTaskAction `
    -Execute  $pythonPath `
    -Argument "`"$scriptPath`""

# Trigger：每天 00:00 起，每 4 小時重複，持續一整天
$trigger = New-ScheduledTaskTrigger -Daily -At "00:00"
$trigger.RepetitionInterval = "PT4H"
$trigger.RepetitionDuration = "P1D"

# 設定
$settings = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit      (New-TimeSpan -Hours 1) `
    -RunOnlyIfNetworkAvailable `
    -StartWhenAvailable `
    -MultipleInstances       IgnoreNew

# 建立工作（以目前登入的使用者身份執行）
Register-ScheduledTask `
    -TaskName    $taskName `
    -Action      $action `
    -Trigger     $trigger `
    -Settings    $settings `
    -RunLevel    Highest `
    -Force | Out-Null

$info = Get-ScheduledTaskInfo -TaskName $taskName
Write-Host "OK  工作已建立：$taskName"
Write-Host "    下次執行時間：$($info.NextRunTime)"
Write-Host ""
Write-Host "手動立即測試執行："
Write-Host ("    Start-ScheduledTask -TaskName " + $taskName)
