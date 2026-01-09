# GitHub Repository Setup

## Creating a New Repository

1. **Go to GitHub**: https://github.com/new
2. **Repository name**: `glucose-mlp-interactive` (or your preferred name)
3. **Description**: "Interactive web app for training and comparing regression models"
4. **Visibility**: Choose Public or Private
5. **DO NOT** initialize with README, .gitignore, or license (we already have these)
6. **Click "Create repository"**

## Pushing to GitHub

### If Repository Already Exists on GitHub

```bash
# Add remote (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/glucose-mlp-interactive.git

# Push to GitHub
git branch -M main
git push -u origin main
```

### If You Need to Create Repository First

1. **Create the repository on GitHub** (see above)
2. **Then run:**
```bash
git remote add origin https://github.com/YOUR_USERNAME/glucose-mlp-interactive.git
git branch -M main
git push -u origin main
```

## After Pushing

Your repository will be available at:
```
https://github.com/YOUR_USERNAME/glucose-mlp-interactive
```

## Cloning the Repository (For Others)

Once pushed, others can clone and run:

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/glucose-mlp-interactive.git
cd glucose-mlp-interactive

# Run setup
# Windows:
.\setup.ps1

# macOS/Linux:
chmod +x setup.sh
./setup.sh

# Run the app
streamlit run app.py
```

## Updating the Repository

After making changes:

```bash
git add .
git commit -m "Description of changes"
git push origin main
```
