import SwiftUI

struct SplashScreenView: View {
    @State private var showClock = false
    @State private var showText = false
    @State private var showStatus = false

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            VStack(spacing: 20) {
                Spacer()

                // ASCII clock art
                if showClock {
                    Text(clockArt)
                        .font(.system(size: 14, weight: .regular, design: .monospaced))
                        .foregroundColor(.terminalGreen)
                        .multilineTextAlignment(.leading)
                        .opacity(showClock ? 1 : 0)
                        .transition(.opacity)
                }

                // App title
                if showText {
                    Text("ExactTime")
                        .font(.system(size: 32, weight: .bold, design: .monospaced))
                        .foregroundColor(.terminalBright)
                        .opacity(showText ? 1 : 0)
                        .transition(.opacity)
                }

                // Status message
                if showStatus {
                    HStack(spacing: 4) {
                        Text(">")
                            .foregroundColor(.terminalGreen)
                        Text("Syncing with NIST time servers...")
                            .foregroundColor(.terminalGreen)
                    }
                    .font(.system(size: 16, weight: .regular, design: .monospaced))
                    .opacity(showStatus ? 1 : 0)
                    .transition(.opacity)
                }

                Spacer()
                Spacer()
            }
            .padding()
        }
        .onAppear {
            // Sequence the animations
            withAnimation(.easeIn(duration: 0.3)) {
                showClock = true
            }

            DispatchQueue.main.asyncAfter(deadline: .now() + 0.4) {
                withAnimation(.easeIn(duration: 0.3)) {
                    showText = true
                }
            }

            DispatchQueue.main.asyncAfter(deadline: .now() + 0.8) {
                withAnimation(.easeIn(duration: 0.3)) {
                    showStatus = true
                }
            }
        }
    }
    
    private var clockArt: String {
        // Each line is 13 characters for proper alignment
        [
            "  .-------.  ",
            " /   12    \\ ",
            "|     |     |",
            "|9    |    3|",
            "|      \\    |",
            "|           |",
            " \\    6    / ",
            "  '-------'  "
        ].joined(separator: "\n")
    }
}

