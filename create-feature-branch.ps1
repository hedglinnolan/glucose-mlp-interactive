# PowerShell script to create a new feature branch
# Usage: .\create-feature-branch.ps1 -FeatureName "your-feature-name"

param(
    [Parameter(Mandatory=$true)]
    [string]$FeatureName
)

# Validate feature name (no spaces, lowercase, hyphens only)
if ($FeatureName -match '[^a-z0-9-]') {
    Write-Host "❌ Error: Feature name should be lowercase with hyphens only (e.g., 'add-new-model')" -ForegroundColor Red
    exit 1
}

Write-Host "🚀 Creating feature branch: feature/$FeatureName" -ForegroundColor Cyan

# Make sure we're on main
$currentBranch = git rev-parse --abbrev-ref HEAD
if ($currentBranch -ne "main") {
    Write-Host "⚠️  Currently on branch: $currentBranch" -ForegroundColor Yellow
    $switch = Read-Host "Switch to main? (y/n)"
    if ($switch -eq "y") {
        git checkout main
    } else {
        Write-Host "❌ Cancelled" -ForegroundColor Red
        exit 1
    }
}

# Pull latest changes
Write-Host "📥 Pulling latest changes from main..." -ForegroundColor Cyan
git pull origin main

# Create and switch to feature branch
Write-Host "🌿 Creating feature branch..." -ForegroundColor Cyan
git checkout -b "feature/$FeatureName"

Write-Host "✅ Successfully created and switched to: feature/$FeatureName" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Make your changes"
Write-Host "  2. git add ."
Write-Host "  3. git commit -m 'Your commit message'"
Write-Host "  4. git push origin feature/$FeatureName"
Write-Host "  5. Create a Pull Request on GitHub"
Write-Host ""
