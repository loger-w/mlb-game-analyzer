# 建立 MLB Odds 每日 5 次自動抓取 + 聰明錢分析的 Task Scheduler 工作
#
# Cron 設計用 ET 思考（對齊 MLB 賽程與 Pinnacle 市場節奏）：
#   ET 22 (prev day) — opener / 早期市場 anchor
#   ET 12             — 早場 / 日場前 1-2h
#   ET 15             — 早晚場前 1-3h
#   ET 18             — 主流晚場前 1h（Pinnacle 限額峰值）
#   ET 21             — 西岸前 1h
#
# 對應 Windows Task Scheduler local time（TW = ET + 12h，球季 EDT）：
#   TW 00:00 / 03:00 / 06:00 / 09:00 / 10:00
#
# 注意：冬季 EST（11-3 月）TW 與 ET 差 13h，本排程僅球季有效
# Credit：5 次/天 × 3 = 15/天，月均 450（500 額度內，剩 50 buffer）
#
# 執行方式（系統管理員 PowerShell）：
#   powershell -ExecutionPolicy Bypass -File setup_task.ps1

$pythonPath     = (Get-Command python).Source
$fetchScript    = "$PSScriptRoot\odds\fetch_odds.py"
$analyzeScript  = "$PSScriptRoot\odds\analyze_smart_money.py"
$taskName       = "MLB_OddsAnalysis_Daily"

# 清理舊命名的 task（沿革：TW-localized → ET-native）
$oldTaskNames = @(
    "MLB_FetchOdds_Every4Hours",   # 原始版本
    "MLB_OddsAnalysis_DailyTW"     # TW-localized 期間命名
)
foreach ($n in $oldTaskNames) {
    Unregister-ScheduledTask -TaskName $n -Confirm:$false -ErrorAction SilentlyContinue
}
Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue

# 兩段 action：先 fetch 再 analyze（Task Scheduler 順序執行）
$action_fetch   = New-ScheduledTaskAction `
    -Execute  $pythonPath `
    -Argument "`"$fetchScript`""

$action_analyze = New-ScheduledTaskAction `
    -Execute  $pythonPath `
    -Argument "`"$analyzeScript`""

# 5 個 daily triggers（local time = TW；對應 ET 12 / 15 / 18 / 21 / 22 prev day）
$t1 = New-ScheduledTaskTrigger -Daily -At "00:00"   # ET 12 — 早場 / 日場前
$t2 = New-ScheduledTaskTrigger -Daily -At "03:00"   # ET 15 — 早晚場前
$t3 = New-ScheduledTaskTrigger -Daily -At "06:00"   # ET 18 — 主流晚場前 1h
$t4 = New-ScheduledTaskTrigger -Daily -At "09:00"   # ET 21 — 西岸前 1h
$t5 = New-ScheduledTaskTrigger -Daily -At "10:00"   # ET 22 prev day — opener / 早期市場 anchor
$triggers = @($t1, $t2, $t3, $t4, $t5)

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
Write-Host "    觸發時間（TW local）：00:00 / 03:00 / 06:00 / 09:00 / 10:00"
Write-Host "    對應 ET（EDT）        ：12:00 / 15:00 / 18:00 / 21:00 / 22:00 (prev day)"
Write-Host "    動作：1) fetch_odds.py  ->  2) analyze_smart_money.py"
Write-Host "    下次執行時間：$($info.NextRunTime)"
Write-Host ""
Write-Host "手動立即測試執行："
Write-Host ("    Start-ScheduledTask -TaskName " + $taskName)
