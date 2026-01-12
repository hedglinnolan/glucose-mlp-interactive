#!/bin/bash
# Script to create a new feature branch
# Usage: ./create-feature-branch.sh your-feature-name

if [ -z "$1" ]; then
    echo "❌ Error: Please provide a feature name"
    echo "Usage: ./create-feature-branch.sh your-feature-name"
    exit 1
fi

FEATURE_NAME="$1"

# Validate feature name (lowercase, hyphens only)
if [[ ! "$FEATURE_NAME" =~ ^[a-z0-9-]+$ ]]; then
    echo "❌ Error: Feature name should be lowercase with hyphens only (e.g., 'add-new-model')"
    exit 1
fi

echo "🚀 Creating feature branch: feature/$FEATURE_NAME"

# Make sure we're on main
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [ "$CURRENT_BRANCH" != "main" ]; then
    echo "⚠️  Currently on branch: $CURRENT_BRANCH"
    read -p "Switch to main? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git checkout main
    else
        echo "❌ Cancelled"
        exit 1
    fi
fi

# Pull latest changes
echo "📥 Pulling latest changes from main..."
git pull origin main

# Create and switch to feature branch
echo "🌿 Creating feature branch..."
git checkout -b "feature/$FEATURE_NAME"

echo "✅ Successfully created and switched to: feature/$FEATURE_NAME"
echo ""
echo "Next steps:"
echo "  1. Make your changes"
echo "  2. git add ."
echo "  3. git commit -m 'Your commit message'"
echo "  4. git push origin feature/$FEATURE_NAME"
echo "  5. Create a Pull Request on GitHub"
echo ""
