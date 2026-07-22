# 毎日の実行 + GitHubへの自動publish用スクリプト。
# タスクスケジューラはこのスクリプトを呼び出す(main.py単体ではなくこちらに変更する)。

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

& "$PSScriptRoot\.venv\Scripts\python.exe" "$PSScriptRoot\main.py"

git add reports

$changes = git status --porcelain reports
if ($changes) {
    git commit -m "Daily report $(Get-Date -Format 'yyyy-MM-dd')"
    git push
    Write-Output "GitHubへpushしました。"
} else {
    Write-Output "レポートに変更がないためpushをスキップしました。"
}
