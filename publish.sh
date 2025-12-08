#!/bin/bash

# Publish script for kotogram
# This script bumps the version number, updates all version files, creates a git tag, and pushes it.
# The pushed tag triggers the python_publish.yml and typescript_publish.yml workflows.

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${GREEN}$1${NC}"
}

print_warning() {
    echo -e "${YELLOW}$1${NC}"
}

print_error() {
    echo -e "${RED}$1${NC}"
}

# Check if we're in the right directory
if [ ! -f "version.txt" ]; then
    print_error "Error: version.txt not found. Are you in the project root?"
    exit 1
fi

# Check for uncommitted changes
if ! git diff-index --quiet HEAD --; then
    print_error "Error: You have uncommitted changes. Please commit or stash them first."
    git status --short
    exit 1
fi

# Read current version
CURRENT_VERSION=$(cat version.txt)
print_info "Current version: $CURRENT_VERSION"

# Parse version components
IFS='.' read -r -a VERSION_PARTS <<< "$CURRENT_VERSION"
MAJOR="${VERSION_PARTS[0]}"
MINOR="${VERSION_PARTS[1]}"
PATCH="${VERSION_PARTS[2]}"

# Default to patch bump
BUMP_TYPE="${1:-patch}"

# Calculate new version
case "$BUMP_TYPE" in
    major)
        NEW_MAJOR=$((MAJOR + 1))
        NEW_MINOR=0
        NEW_PATCH=0
        ;;
    minor)
        NEW_MAJOR=$MAJOR
        NEW_MINOR=$((MINOR + 1))
        NEW_PATCH=0
        ;;
    patch)
        NEW_MAJOR=$MAJOR
        NEW_MINOR=$MINOR
        NEW_PATCH=$((PATCH + 1))
        ;;
    *)
        print_error "Error: Invalid bump type '$BUMP_TYPE'. Use 'major', 'minor', or 'patch'."
        exit 1
        ;;
esac

NEW_VERSION="$NEW_MAJOR.$NEW_MINOR.$NEW_PATCH"
print_info "New version: $NEW_VERSION"

# Ask for confirmation
read -p "Bump version from $CURRENT_VERSION to $NEW_VERSION? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    print_warning "Aborted."
    exit 1
fi

# Update version.txt
echo "$NEW_VERSION" > version.txt
print_info "✓ Updated version.txt"

# Update Python package version
sed -i "s/__version__ = \".*\"/__version__ = \"$NEW_VERSION\"/" kotogram/__init__.py
print_info "✓ Updated kotogram/__init__.py"

# Update pyproject.toml
sed -i "s/^version = \".*\"/version = \"$NEW_VERSION\"/" pyproject.toml
print_info "✓ Updated pyproject.toml"

# Update package.json
if command -v node &> /dev/null; then
    node -e "
    const fs = require('fs');
    const pkg = JSON.parse(fs.readFileSync('package.json', 'utf8'));
    pkg.version = '$NEW_VERSION';
    fs.writeFileSync('package.json', JSON.stringify(pkg, null, 2) + '\n');
    "
    print_info "✓ Updated package.json"
else
    print_warning "Warning: Node.js not found. Please manually update package.json to version $NEW_VERSION"
fi

# Commit the version changes
git add version.txt kotogram/__init__.py pyproject.toml package.json
git commit -m "Bump version to $NEW_VERSION"
print_info "✓ Committed version changes"

# Create git tag
TAG_NAME="v$NEW_VERSION"
git tag -a "$TAG_NAME" -m "Release version $NEW_VERSION"
print_info "✓ Created git tag $TAG_NAME"

# Push to remote
print_warning "Pushing to remote repository..."
git push origin main
git push origin "$TAG_NAME"

print_info "✅ Successfully published version $NEW_VERSION!"
print_info ""
print_info "The following GitHub Actions workflows should now be triggered:"
print_info "  - python_publish.yml (publishes to PyPI)"
print_info "  - typescript_publish.yml (publishes to npm)"
print_info ""
print_info "Monitor the workflows at: https://github.com/$(git remote get-url origin | sed 's/.*://;s/.git$//')/actions"
