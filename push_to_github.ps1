# PowerShell script to push to GitHub
# Run this after creating the repository on GitHub

param(
    [Parameter(Mandatory=$true)]
    [string]$GitHubUsername,
    
    [Parameter(Mandatory=$false)]
    [string]$RepoName = "glucose-mlp-interactive"
)

Write-Host "🚀 Pushing to GitHub..." -ForegroundColor Cyan

# Check if remote already exists
$existingRemote = git remote get-url origin 2>$null
if ($existingRemote) {
    Write-Host "⚠️  Remote already exists: $existingRemote" -ForegroundColor Yellow
    $response = Read-Host "Do you want to update it? (y/n)"
    if ($response -ne "y") {
        Write-Host "Cancelled." -ForegroundColor Red
        exit
    }
    git remote remove origin
}

# Add remote
$remoteUrl = "https://github.com/$GitHubUsername/$RepoName.git"
Write-Host "Adding remote: $remoteUrl" -ForegroundColor Yellow
git remote add origin $remoteUrl

# Rename branch to main
Write-Host "Renaming branch to main..." -ForegroundColor Yellow
git branch -M main

# Push to GitHub
Write-Host "Pushing to GitHub..." -ForegroundColor Yellow
git push -u origin main

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n✅ Successfully pushed to GitHub!" -ForegroundColor Green
    Write-Host "`n📋 Repository URL:" -ForegroundColor Cyan
    Write-Host "   https://github.com/$GitHubUsername/$RepoName" -ForegroundColor White
    Write-Host "`n🔗 Share this URL with your colleague!" -ForegroundColor Yellow
    Write-Host "`n💡 They can clone and run with:" -ForegroundColor Cyan
    Write-Host "   git clone https://github.com/$GitHubUsername/$RepoName.git" -ForegroundColor White
    Write-Host "   cd $RepoName" -ForegroundColor White
    Write-Host "   .\setup.ps1" -ForegroundColor White
    Write-Host "   streamlit run app.py" -ForegroundColor White
} else {
    Write-Host "`n❌ Error pushing to GitHub!" -ForegroundColor Red
    Write-Host "`nPossible issues:" -ForegroundColor Yellow
    Write-Host "  1. Repository doesn't exist - create it at https://github.com/new"
    Write-Host "  2. Authentication required - use GitHub CLI or Personal Access Token"
    Write-Host "  3. Network issues - check internet connection"
    Write-Host "`nFor detailed instructions, see PUSH_TO_GITHUB.md" -ForegroundColor Cyan
}
