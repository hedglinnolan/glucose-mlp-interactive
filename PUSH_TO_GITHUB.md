# Push to GitHub - Step by Step

## Step 1: Create GitHub Repository

1. **Go to GitHub**: https://github.com/new
2. **Repository name**: `glucose-mlp-interactive`
3. **Description**: "Interactive web app for training and comparing regression models (Neural Network, Random Forest, GLM)"
4. **Visibility**: Choose Public or Private
5. **DO NOT** check any boxes (no README, .gitignore, or license)
6. **Click "Create repository"**

## Step 2: Push Your Code

After creating the repository, GitHub will show you commands. Use these:

```bash
# Navigate to project directory
cd glucose-mlp-interactive

# Add remote (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/glucose-mlp-interactive.git

# Rename branch to main (if needed)
git branch -M main

# Push to GitHub
git push -u origin main
```

## Step 3: Verify

After pushing, visit:
```
https://github.com/YOUR_USERNAME/glucose-mlp-interactive
```

You should see all your files!

## Step 4: Share with Others

Others can now clone and run:

```bash
git clone https://github.com/YOUR_USERNAME/glucose-mlp-interactive.git
cd glucose-mlp-interactive
.\setup.ps1  # Windows
# or
./setup.sh   # macOS/Linux
streamlit run app.py
```

## Troubleshooting

### "Repository not found"

- Make sure you created the repository on GitHub first
- Check that the repository name matches
- Verify your GitHub username is correct

### "Permission denied"

- Make sure you're authenticated with GitHub
- Use SSH instead: `git@github.com:YOUR_USERNAME/glucose-mlp-interactive.git`
- Or use GitHub CLI: `gh auth login`

### "Remote already exists"

```bash
# Remove existing remote
git remote remove origin

# Add new remote
git remote add origin https://github.com/YOUR_USERNAME/glucose-mlp-interactive.git
```

### Authentication Issues

**Option 1: Use Personal Access Token**
1. Go to GitHub Settings → Developer settings → Personal access tokens
2. Generate new token (classic)
3. Use token as password when pushing

**Option 2: Use GitHub CLI**
```bash
gh auth login
git push -u origin main
```

**Option 3: Use SSH**
```bash
# Generate SSH key (if needed)
ssh-keygen -t ed25519 -C "your_email@example.com"

# Add to GitHub: Settings → SSH and GPG keys

# Use SSH URL
git remote set-url origin git@github.com:YOUR_USERNAME/glucose-mlp-interactive.git
git push -u origin main
```
