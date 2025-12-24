import SwiftUI

struct AboutView: View {
    let onDismiss: () -> Void

    var body: some View {
        NavigationStack {
            ZStack {
                Color.black.ignoresSafeArea()

                ScrollView {
                    VStack(alignment: .leading, spacing: 16) {
                        Text("── About ExactTime ──")
                            .foregroundColor(.terminalDim)

                        Text("What it does:")
                            .foregroundColor(.terminalBright)

                        Text("""
                        Displays precise time synced with NIST servers via NTP, perfect for setting mechanical and automatic wristwatches.

                        Features:

                        • Syncs with NIST time servers
                        • Compensates for network latency
                        • Continuously updating display
                        • Pull down to re-sync
                        • Shows sync status and offset

                        Time source:
                        """)
                            .foregroundColor(.terminalGreen)
                            .fixedSize(horizontal: false, vertical: true)

                        Link("NIST Internet Time Service", destination: URL(string: "https://www.nist.gov/pml/time-and-frequency-division/time-distribution/internet-time-service-its")!)
                            .foregroundColor(.terminalGreen)
                            .underline()

                        Text("")

                        Text("── Credits ──")
                            .foregroundColor(.terminalDim)

                        Text("Created by:")
                            .foregroundColor(.terminalGreen)

                        Text("Andrew Chamberlain, Ph.D.")
                            .foregroundColor(.terminalBright)

                        Link("andrewchamberlain.com", destination: URL(string: "https://andrewchamberlain.com")!)
                            .foregroundColor(.terminalGreen)
                            .underline()

                        Spacer()
                    }
                    .font(.terminalCaption)
                    .padding()
                }
            }
            .navigationTitle("")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .principal) {
                    Text("About")
                        .font(.terminalTitle)
                        .foregroundColor(.terminalGreen)
                }
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button { onDismiss() } label: {
                        Text("[Close]")
                            .font(.terminalCaption)
                            .foregroundColor(.terminalGreen)
                    }
                }
            }
            .toolbarBackground(Color.black, for: .navigationBar)
            .toolbarBackground(.visible, for: .navigationBar)
        }
        .preferredColorScheme(.dark)
    }
}

