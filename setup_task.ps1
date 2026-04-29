# 建立 MLB Odds 每日 4 次自動抓取 + 聰明錢分析的 Task Scheduler 工作
# 排程：TW 12:00 / 15:00 / 18:00 / 21:00（local time = TW）
#       對應 EDT（球季 4-10 月）= ET 00:00 / 03:00 / 06:00 / 09:00
#       注意：冬季 EST（11-3 月）TW 與 ET 差 13h，本排程僅球季有效
#
# 執行方式（系統管理員 PowerShell）：
#   powershell -ExecutionPolicy Bypass -File setup_task.ps1

$pythonPath     = (Get-Command python).Source
$fetchScript    = "$PSScriptRoot\scripts\fetch_odds.py"
$analyzeScript  = "$PSScriptRoot\odds\analyze_smart_money.py"
$taskName       = "MLB_OddsAnalysis_DailyTW"
$oldTaskName    = "MLB_FetchOdds_Every4Hours"

# 移除舊任務（若存在）
Unregister-ScheduledTask -TaskName $oldTaskName -Confirm:$false -ErrorAction SilentlyContinue
Unregister-ScheduledTask -TaskName $taskName    -Confirm:$false -ErrorAction SilentlyContinue

# 兩段 action：先 fetch 再 analyze（Task Scheduler 順序執行）
$action_fetch   = New-ScheduledTaskAction `
    -Execute  $pythonPath `
    -Argument "`"$fetchScript`""

$action_analyze = New-ScheduledTaskAction `
    -Execute  $pythonPath `
    -Argument "`"$analyzeScript`""

# 4 個 daily triggers（local time = TW）
$t1 = New-ScheduledTaskTrigger -Daily -At "12:00"
$t2 = New-ScheduledTaskTrigger -Daily -At "15:00"
$t3 = New-ScheduledTaskTrigger -Daily -At "18:00"
$t4 = New-ScheduledTaskTrigger -Daily -At "21:00"
$triggers = @($t1, $t2, $t3, $t4)

# 設定
$settings = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit      (New-TimeSpan -Hours 1) `
    -RunOnlyIfNetworkAvailable `
    -StartWhenAvailable `
    -MultipleInstances       IgnoreNew

# 建立工作（以目前登入的使用者身份執行；RunLevel Limited 可不需 admin 註冊）
Register-ScheduledTask `
    -TaskName    $taskName `
    -Action      @($action_fetch, $action_analyze) `
    -Trigger     $triggers `
    -Settings    $settings `
    -Force | Out-Null

$info = Get-ScheduledTaskInfo -TaskName $taskName
Write-Host "OK  工作已建立：$taskName"
Write-Host "    觸發時間（TW）：12:00 / 15:00 / 18:00 / 21:00"
Write-Host "    對應 ET（EDT）：00:00 / 03:00 / 06:00 / 09:00"
Write-Host "    動作：1) fetch_odds.py  ->  2) analyze_smart_money.py"
Write-Host "    下次執行時間：$($info.NextRunTime)"
Write-Host ""
Write-Host "手動立即測試執行："
Write-Host ("    Start-ScheduledTask -TaskName " + $taskName)
