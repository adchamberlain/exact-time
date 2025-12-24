import SwiftUI

// MARK: - Terminal Color Extensions
extension Color {
    static let terminalGreen = Color(red: 0.2, green: 1.0, blue: 0.2)
    static let terminalDim = Color(red: 0.15, green: 0.6, blue: 0.15)
    static let terminalBright = Color(red: 0.4, green: 1.0, blue: 0.4)
}

// MARK: - Terminal Font Extensions
extension Font {
    static func terminal(_ size: CGFloat) -> Font {
        .system(size: size, weight: .regular, design: .monospaced)
    }
    static let terminalBody: Font = .system(size: 15, weight: .regular, design: .monospaced)
    static let terminalCaption: Font = .system(size: 15, weight: .regular, design: .monospaced)
    static let terminalSmall: Font = .system(size: 13, weight: .regular, design: .monospaced)
    static let terminalTitle: Font = .system(size: 18, weight: .medium, design: .monospaced)
    static let terminalLarge: Font = .system(size: 28, weight: .regular, design: .monospaced)
}

// MARK: - Main Content View
struct ContentView: View {
    @EnvironmentObject var ntpService: NTPService
    @State private var showingAbout = false
    
    var body: some View {
        // TimelineView updates the view on a schedule - perfect for a clock
        // Update 10x per second for minimal display lag
        TimelineView(.periodic(from: .now, by: 0.1)) { context in
            NavigationStack {
                ScrollView {
                    VStack(alignment: .center, spacing: 0) {
                        Spacer()
                            .frame(height: 30)
                        
                        // Main time display - passes the timeline date to trigger updates
                        timeDisplayView(timelineDate: context.date)
                        
                        Spacer()
                            .frame(height: 20)
                        
                        // Status info
                        statusView
                        
                        Spacer()
                            .frame(height: 30)
                        
                        // About link
                        Button {
                            showingAbout = true
                        } label: {
                            Text("[About]")
                                .font(.terminalSmall)
                                .foregroundColor(.terminalDim)
                        }
                        
                        Spacer()
                            .frame(height: 40)
                    }
                    .frame(maxWidth: .infinity, alignment: .center)
                    .padding(.horizontal, 30)
                }
                .background(Color.black)
                .refreshable {
                    await ntpService.sync()
                }
                .sheet(isPresented: $showingAbout) {
                    AboutView(onDismiss: { showingAbout = false })
                }
                .navigationTitle("")
                .navigationBarTitleDisplayMode(.inline)
                .toolbar {
                    ToolbarItem(placement: .principal) {
                        Text("ExactTime v1.0")
                            .font(.terminalTitle)
                            .foregroundColor(.terminalGreen)
                    }
                }
                .toolbarBackground(Color.black, for: .navigationBar)
                .toolbarBackground(.visible, for: .navigationBar)
            }
        }
        .preferredColorScheme(.dark)
        .task {
            await ntpService.sync()
        }
    }
    
    // MARK: - Time Display
    @ViewBuilder
    private func timeDisplayView(timelineDate: Date) -> some View {
        // Simple box around the time - all same font size for alignment
        // Time with AM/PM is up to 11 chars (e.g. "12:59:59 PM"), + 2 spaces each side + 2 pipes = 17 total
        let time = formattedTime(for: timelineDate)
        VStack(spacing: 0) {
            Text("-----------------")
            Text("|  \(time)  |")
            Text("-----------------")
        }
        .font(.terminal(22))
        .foregroundColor(.terminalBright)
        .monospacedDigit()
        .padding(.vertical, 20)
    }
    
    // MARK: - Status View
    private var statusView: some View {
        VStack(alignment: .leading, spacing: 8) {
            // Date
            Text("> \(formattedDate)")
                .font(.terminalCaption)
                .foregroundColor(.terminalGreen)
            
            // Timezone
            Text("> Zone: \(currentTimezone)")
                .font(.terminalCaption)
                .foregroundColor(.terminalGreen)
            
            // Source
            HStack(spacing: 4) {
                Text("> Source:")
                    .foregroundColor(.terminalGreen)
                Text(ntpService.syncState.displayText)
                    .foregroundColor(sourceColor)
            }
            .font(.terminalCaption)
            
            // Last sync time
            if let timeSince = ntpService.timeSinceSync {
                Text("> Synced: \(timeSince)")
                    .font(.terminalCaption)
                    .foregroundColor(.terminalDim)
            }
            
            // Offset info (for debugging/transparency)
            if ntpService.isSynced {
                let offsetMs = Int(ntpService.timeOffset * 1000)
                Text("> Offset: \(offsetMs > 0 ? "+" : "")\(offsetMs)ms")
                    .font(.terminalSmall)
                    .foregroundColor(.terminalDim)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.horizontal, 40)
        .padding(.bottom, 30)
    }
    
    // MARK: - Time Formatting
    
    /// Format the accurate NTP-synced time
    private func formattedTime(for timelineDate: Date) -> String {
        // Get accurate time from NTP service (applies offset to current time)
        let accurateTime = ntpService.now()
        let formatter = DateFormatter()
        formatter.dateFormat = "h:mm:ss a"
        return formatter.string(from: accurateTime)
    }
    
    private var formattedDate: String {
        let time = ntpService.now()
        let formatter = DateFormatter()
        formatter.dateFormat = "EEE, MMM d, yyyy"
        return formatter.string(from: time)
    }
    
    private var currentTimezone: String {
        TimeZone.current.abbreviation() ?? TimeZone.current.identifier
    }
    
    private var sourceColor: Color {
        switch ntpService.syncState {
        case .synced:
            return .terminalBright
        case .syncing:
            return .terminalGreen
        case .failed:
            return .red
        case .idle:
            return .terminalDim
        }
    }
    
}

#Preview {
    ContentView()
        .environmentObject(NTPService.shared)
}

