# Development Workflow Guide

This guide helps you safely add new features and test changes without breaking what's currently working.

## 🎯 Branching Strategy

We use a simple branching strategy:
- **`main`**: Production-ready, working code (always stable)
- **`develop`**: Integration branch for features (optional)
- **`feature/*`**: Individual feature branches

## 🚀 Quick Start: Adding a New Feature

### Step 1: Create a Feature Branch

```powershell
# Make sure you're on main and it's up to date
git checkout main
git pull origin main

# Create and switch to a new feature branch
git checkout -b feature/your-feature-name

# Example:
git checkout -b feature/add-new-model
```

### Step 2: Make Your Changes

Work on your feature, make commits as you go:

```powershell
# Make changes to files
# ... edit files ...

# Stage and commit
git add .
git commit -m "Add new feature: description of what you did"
```

### Step 3: Test Your Changes

Before merging, test thoroughly:

```powershell
# Run your app locally
streamlit run app.py

# Test all functionality
# Make sure existing features still work
```

### Step 4: Push and Create Pull Request

```powershell
# Push your feature branch
git push origin feature/your-feature-name
```

Then go to GitHub and create a Pull Request:
1. Visit: https://github.com/hedglinnolan/glucose-mlp-interactive
2. Click "New Pull Request"
3. Select your feature branch
4. Review changes
5. Merge when ready

### Step 5: Clean Up

After merging:

```powershell
# Switch back to main
git checkout main

# Pull the latest changes
git pull origin main

# Delete local feature branch (optional)
git branch -d feature/your-feature-name
```

## 📋 Best Practices

### Commit Often, Commit Small

Make frequent, small commits with clear messages:

```powershell
git commit -m "Add data validation for CSV upload"
git commit -m "Fix neural network training progress display"
```

### Keep Main Stable

- **Never** commit directly to `main` for new features
- Always use feature branches
- Test thoroughly before merging

### Before Starting New Work

Always start from an up-to-date main:

```powershell
git checkout main
git pull origin main
```

### If Something Breaks

Don't panic! You can always:

```powershell
# See what changed
git status
git diff

# Discard changes to a file
git checkout -- filename.py

# Discard all uncommitted changes (careful!)
git reset --hard HEAD

# Go back to a previous commit (if needed)
git log --oneline
git checkout <commit-hash>
```

## 🔄 Common Workflows

### Working on Multiple Features

```powershell
# Feature 1
git checkout -b feature/feature-1
# ... work ...
git commit -m "..."

# Switch to feature 2
git checkout main
git checkout -b feature/feature-2
# ... work ...
git commit -m "..."

# Switch back to feature 1
git checkout feature/feature-1
```

### Updating Your Feature Branch

If main has new changes while you're working:

```powershell
# On your feature branch
git checkout feature/your-feature-name

# Get latest main
git fetch origin
git merge origin/main

# Resolve any conflicts if needed
# Then continue working
```

### Stashing Changes (Temporary Save)

If you need to switch branches but aren't ready to commit:

```powershell
# Save changes temporarily
git stash

# Switch branches, do other work
git checkout main

# Come back and restore
git checkout feature/your-feature-name
git stash pop
```

## 🛡️ Safety Checklist

Before merging a feature branch:

- [ ] Code runs without errors
- [ ] Existing features still work
- [ ] No console errors or warnings
- [ ] Tested with sample data
- [ ] Code is clean and readable
- [ ] Commits have clear messages

## 📝 Branch Naming Conventions

Use descriptive names:

- ✅ `feature/add-random-forest-tuning`
- ✅ `feature/improve-visualizations`
- ✅ `bugfix/fix-csv-upload-error`
- ❌ `feature/test`
- ❌ `feature/new-stuff`

## 🆘 Emergency: Reverting Changes

If you accidentally break something on main:

```powershell
# See recent commits
git log --oneline -10

# Revert a specific commit
git revert <commit-hash>

# Or go back to a working commit
git reset --hard <commit-hash>
git push origin main --force  # ⚠️ Use carefully!
```

## 📚 Additional Resources

- [Git Basics](https://git-scm.com/book/en/v2/Getting-Started-Git-Basics)
- [GitHub Flow](https://guides.github.com/introduction/flow/)
- [Git Cheat Sheet](https://education.github.com/git-cheat-sheet-education.pdf)
