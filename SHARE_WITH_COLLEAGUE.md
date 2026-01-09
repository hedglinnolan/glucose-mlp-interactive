# How to Share This App with Your Colleague

## Quick Steps to Make It Public on GitHub

### Step 1: Create Repository on GitHub

1. **Go to**: https://github.com/new
2. **Repository name**: `glucose-mlp-interactive`
3. **Description**: "Interactive web app for training and comparing regression models"
4. **Visibility**: ⚠️ **Choose PUBLIC** (so your colleague can access it)
5. **DO NOT** check any boxes (no README, .gitignore, or license)
6. **Click "Create repository"**

### Step 2: Push Your Code

**Option A: Using the automated script (Recommended)**

```powershell
# Run this script (it will prompt for your GitHub username)
.\push_to_github.ps1 -GitHubUsername YOUR_GITHUB_USERNAME
```

**Option B: Manual push**

```bash
# Add remote (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/glucose-mlp-interactive.git

# Rename branch
git branch -M main

# Push to GitHub
git push -u origin main
```

**Note**: If prompted for authentication:
- Use a **Personal Access Token** (not password)
- Or use **GitHub CLI**: `gh auth login` then `gh repo create`

### Step 3: Share the URL

Once pushed, share this URL with your colleague:
```
https://github.com/YOUR_USERNAME/glucose-mlp-interactive
```

### Step 4: Your Colleague Can Now Use It

Your colleague can clone and run:

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/glucose-mlp-interactive.git
cd glucose-mlp-interactive

# Setup (Windows)
.\setup.ps1

# Setup (macOS/Linux)
chmod +x setup.sh
./setup.sh
source .venv/bin/activate

# Run the app
streamlit run app.py
```

## Making It Private Later

If you want to make it private later:
1. Go to your repository on GitHub
2. Click "Settings"
3. Scroll down to "Danger Zone"
4. Click "Change visibility"
5. Choose "Make private"

## Adding Collaborators (Alternative)

If you want to keep it private but share with specific people:
1. Go to repository Settings
2. Click "Collaborators"
3. Click "Add people"
4. Enter their GitHub username or email
5. They'll get an invitation to access the private repo

## Troubleshooting

### "Repository not found"
- Make sure you created the repository on GitHub first
- Check that the repository name matches exactly
- Verify your GitHub username is correct

### Authentication Issues

**Option 1: Use Personal Access Token**
1. Go to: GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Generate new token with `repo` scope
3. Use token as password when pushing

**Option 2: Use GitHub CLI**
```bash
gh auth login
gh repo create glucose-mlp-interactive --public
git push -u origin main
```

**Option 3: Use SSH**
```bash
# Generate SSH key if needed
ssh-keygen -t ed25519 -C "your_email@example.com"

# Add to GitHub: Settings → SSH and GPG keys

# Use SSH URL
git remote set-url origin git@github.com:YOUR_USERNAME/glucose-mlp-interactive.git
git push -u origin main
```

## Current Status

✅ All code is committed and ready to push
✅ Documentation is complete
✅ Setup scripts are included
✅ Example data is included

**Next step**: Create the repository on GitHub and push!
