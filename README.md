# ExactTime

A native iOS app that displays precise time synced with NIST NTP servers, perfect for setting mechanical and automatic wristwatches.

Built with SwiftUI and featuring a retro terminal aesthetic matching the SkyWatch app.

## Features

- Accurate time synced with NIST time servers via NTP
- Compensates for network latency to provide true time
- Continuously updating clock display (HH:MM:SS)
- Manual re-sync button
- Shows sync status and last sync time
- Terminal-style UI with green-on-black aesthetic
- No external dependencies - pure Swift implementation

## Requirements

- iPhone running iOS 16.0 or later
- Mac with Xcode 15+ installed
- Free Apple ID (for personal device installation)
- Internet connection for NTP sync

---

## Installation Guide

Follow these steps to download and install ExactTime on your iPhone.

### Step 1: Download the Code

**Option A: Using Git (recommended)**
```bash
git clone https://github.com/adchamberlain/exact-time.git
```

**Option B: Download ZIP**
1. Go to https://github.com/adchamberlain/exact-time
2. Click the green **"Code"** button
3. Click **"Download ZIP"**
4. Unzip the downloaded file

### Step 2: Open in Xcode

1. Open **Finder** and navigate to the downloaded folder
2. Go to: `exact-time/ExactTime/`
3. Double-click **`ExactTime.xcodeproj`** to open it in Xcode

Or from Terminal:
```bash
cd exact-time/ExactTime
open ExactTime.xcodeproj
```

### Step 3: Configure Code Signing

This allows the app to run on your personal iPhone.

1. In Xcode, click on **"ExactTime"** in the left sidebar (the blue project icon at the top)
2. Under **TARGETS**, select **"ExactTime"**
3. Click the **"Signing & Capabilities"** tab
4. Check ✅ **"Automatically manage signing"**
5. Click the **Team** dropdown:
   - If you see your name, select it
   - If not, click **"Add an Account..."** and sign in with your Apple ID
6. Select **"Your Name (Personal Team)"**

> ⚠️ **Note**: With a free Apple ID, apps expire after 7 days and must be reinstalled. A paid Apple Developer account ($99/year) removes this limitation.

### Step 4: Connect Your iPhone

1. Connect your iPhone to your Mac with a USB cable
2. **On your iPhone**: When prompted "Trust This Computer?", tap **Trust** and enter your passcode
3. Wait a few seconds for Xcode to recognize your device

### Step 5: Select Your iPhone as the Target

1. At the top of Xcode, find the device selector (it may say "iPhone 15 Pro" or similar)
2. Click the dropdown
3. Under **"iOS Devices"**, select your connected iPhone

### Step 6: Build and Run

1. Press **⌘R** (Command + R) or click the **▶ Play** button
2. Xcode will build the app and install it on your iPhone
3. **First time only**: The build may take 1-2 minutes

### Step 7: Trust the Developer Certificate (First Time Only)

The first time you install, your iPhone won't run the app until you trust it:

1. On your iPhone, go to **Settings**
2. Tap **General**
3. Tap **VPN & Device Management**
4. Under "Developer App", tap your **Apple ID email**
5. Tap **"Trust [your email]"**
6. Tap **Trust** to confirm

### Step 8: Launch the App

1. Return to your iPhone home screen
2. Find the **ExactTime** app icon
3. Tap to open!

The app will automatically sync with NIST time servers on launch.

---

## Troubleshooting

### "Untrusted Developer" error
See Step 7 above to trust the developer certificate.

### Play button is greyed out
- Make sure your iPhone is connected and selected as the target device
- Check that code signing is configured (Step 3)

### "Could not launch" error
- Unlock your iPhone screen
- Try unplugging and reconnecting your iPhone

### App expires after 7 days
This is normal with a free Apple ID. Just reconnect your iPhone and press ⌘R in Xcode to reinstall.

### Time sync fails
- Ensure you have an internet connection
- Try tapping the [Re-Sync] button
- Check if your network blocks NTP (UDP port 123)

---

## How It Works

ExactTime uses the **Network Time Protocol (NTP)** to sync with NIST (National Institute of Standards and Technology) time servers using Apple's Network framework:

1. Sends NTP requests via UDP to NIST servers
2. Measures round-trip time to calculate network latency
3. Computes the offset between your device clock and true time
4. Displays corrected time: `device_time + offset`

This provides millisecond-level accuracy, making it perfect for setting precision timepieces.

**NIST Servers Used:**
- `time.nist.gov` (primary)
- `time.apple.com` (backup)
- `time-a-wwv.nist.gov`
- `time-b-wwv.nist.gov`

---

## Credits

**Author**: Andrew Chamberlain, Ph.D. ([andrewchamberlain.com](https://andrewchamberlain.com))

**Time Data**: [NIST Internet Time Service](https://www.nist.gov/pml/time-and-frequency-division/time-distribution/internet-time-service-its)

**License**: MIT
