#!/bin/bash

# ExactTime Xcode Project Setup Helper
# Run this after installing Xcode to prepare the project

echo "⏱️  ExactTime Xcode Project Setup"
echo "=================================="
echo ""

# Check if Xcode is installed
if ! command -v xcodebuild &> /dev/null; then
    echo "❌ Xcode is not installed."
    echo ""
    echo "Please install Xcode from the Mac App Store first:"
    echo "1. Open Mac App Store"
    echo "2. Search for 'Xcode'"
    echo "3. Click 'Get' / 'Install'"
    echo "4. Wait for download to complete (~12GB)"
    echo "5. Run this script again"
    exit 1
fi

XCODE_VERSION=$(xcodebuild -version | head -n1)
echo "✅ Found: $XCODE_VERSION"
echo ""

# Accept Xcode license if needed
echo "Checking Xcode license..."
sudo xcodebuild -license accept 2>/dev/null || true

# Install iOS platform if needed
echo "Checking for iOS platform..."
xcodebuild -downloadPlatform iOS 2>/dev/null || true

echo ""
echo "=================================="
echo "📋 Setup Steps"
echo "=================================="
echo ""
echo "1. Open the project in Xcode:"
echo "   cd $(pwd)/ExactTime"
echo "   open ExactTime.xcodeproj"
echo ""
echo "2. Configure Code Signing:"
echo "   • Click 'ExactTime' project in sidebar"
echo "   • Select 'ExactTime' target"
echo "   • Go to 'Signing & Capabilities'"
echo "   • Check 'Automatically manage signing'"
echo "   • Select your Team (Personal or Developer)"
echo ""
echo "3. Connect your iPhone and press ⌘R to build!"
echo ""
echo "   No external packages required - the app is self-contained."
echo ""
echo "=================================="
echo ""

# Offer to open Xcode
read -p "Would you like to open the project in Xcode now? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    cd "$(dirname "$0")/ExactTime"
    open ExactTime.xcodeproj
    echo "✅ Xcode opened!"
fi

echo ""
echo "Time to set those watches! ⌚"
