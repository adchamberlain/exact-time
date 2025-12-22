import Foundation
import Network

/// Service that manages NTP time synchronization with NIST servers
@MainActor
class NTPService: ObservableObject {
    static let shared = NTPService()
    
    /// Current sync state
    @Published var syncState: SyncState = .idle
    
    /// Last successful sync time
    @Published var lastSyncTime: Date?
    
    /// The calculated offset from device time (in seconds)
    @Published var timeOffset: TimeInterval = 0
    
    /// Whether we have a valid time sync
    var isSynced: Bool {
        lastSyncTime != nil
    }
    
    /// NIST NTP servers
    private let ntpServers: [String] = [
        "time.nist.gov",
        "time.apple.com",
        "time-a-wwv.nist.gov",
        "time-b-wwv.nist.gov"
    ]
    
    private var currentServerIndex = 0
    
    enum SyncState: Equatable {
        case idle
        case syncing
        case synced
        case failed(String)
        
        var displayText: String {
            switch self {
            case .idle:
                return "Not synced"
            case .syncing:
                return "Syncing..."
            case .synced:
                return "NIST"
            case .failed(let error):
                return "Error: \(error)"
            }
        }
    }
    
    private init() {}
    
    /// Returns the current accurate time (device time + offset)
    func now() -> Date {
        return Date().addingTimeInterval(timeOffset)
    }
    
    /// Manually trigger a re-sync with NTP servers
    func sync() async {
        syncState = .syncing
        
        // Try each server until one succeeds
        for i in 0..<ntpServers.count {
            let serverIndex = (currentServerIndex + i) % ntpServers.count
            let server = ntpServers[serverIndex]
            
            do {
                let offset = try await fetchNTPOffset(from: server)
                timeOffset = offset
                lastSyncTime = Date()
                syncState = .synced
                currentServerIndex = serverIndex // Remember working server
                return
            } catch {
                // Try next server
                continue
            }
        }
        
        // All servers failed
        if lastSyncTime == nil {
            syncState = .failed("No connection")
        } else {
            // Keep using last known offset
            syncState = .synced
        }
    }
    
    /// Fetch time offset from an NTP server
    private func fetchNTPOffset(from server: String) async throws -> TimeInterval {
        return try await withCheckedThrowingContinuation { continuation in
            let queue = DispatchQueue(label: "ntp.queue")
            
            // Create UDP connection to NTP server (port 123)
            let host = NWEndpoint.Host(server)
            let port = NWEndpoint.Port(integerLiteral: 123)
            let connection = NWConnection(host: host, port: port, using: .udp)
            
            var hasResumed = false
            let timeoutWorkItem = DispatchWorkItem {
                if !hasResumed {
                    hasResumed = true
                    connection.cancel()
                    continuation.resume(throwing: NTPError.timeout)
                }
            }
            
            // Set 5 second timeout
            queue.asyncAfter(deadline: .now() + 5, execute: timeoutWorkItem)
            
            connection.stateUpdateHandler = { state in
                switch state {
                case .ready:
                    // Connection ready, send NTP request
                    let t1 = Date() // Client send time
                    let packet = self.createNTPPacket()
                    
                    connection.send(content: packet, completion: .contentProcessed { error in
                        if let error = error {
                            if !hasResumed {
                                hasResumed = true
                                timeoutWorkItem.cancel()
                                connection.cancel()
                                continuation.resume(throwing: error)
                            }
                            return
                        }
                        
                        // Receive response
                        connection.receive(minimumIncompleteLength: 48, maximumLength: 48) { data, _, _, error in
                            timeoutWorkItem.cancel()
                            
                            if !hasResumed {
                                hasResumed = true
                                
                                if let error = error {
                                    connection.cancel()
                                    continuation.resume(throwing: error)
                                    return
                                }
                                
                                guard let data = data, data.count >= 48 else {
                                    connection.cancel()
                                    continuation.resume(throwing: NTPError.invalidResponse)
                                    return
                                }
                                
                                let t4 = Date() // Client receive time
                                
                                // Parse NTP response
                                let offset = self.parseNTPResponse(data: data, t1: t1, t4: t4)
                                connection.cancel()
                                continuation.resume(returning: offset)
                            }
                        }
                    })
                    
                case .failed(let error):
                    if !hasResumed {
                        hasResumed = true
                        timeoutWorkItem.cancel()
                        connection.cancel()
                        continuation.resume(throwing: error)
                    }
                    
                case .cancelled:
                    break
                    
                default:
                    break
                }
            }
            
            connection.start(queue: queue)
        }
    }
    
    /// Create an NTP request packet (48 bytes)
    /// Marked nonisolated because it's a pure function called from background queue
    nonisolated private func createNTPPacket() -> Data {
        var packet = Data(count: 48)
        
        // LI (0) | VN (3) | Mode (3 = client)
        // Binary: 00 011 011 = 0x1B
        packet[0] = 0x1B
        
        // Rest of packet is zeros for client request
        return packet
    }
    
    /// Parse NTP response and calculate offset
    /// Using the standard NTP offset calculation:
    /// offset = ((t2 - t1) + (t3 - t4)) / 2
    /// where:
    ///   t1 = client send time
    ///   t2 = server receive time
    ///   t3 = server transmit time
    ///   t4 = client receive time
    /// Marked nonisolated because it's a pure function called from background queue
    nonisolated private func parseNTPResponse(data: Data, t1: Date, t4: Date) -> TimeInterval {
        // NTP timestamp is seconds since Jan 1, 1900
        // We need to convert to Unix epoch (Jan 1, 1970)
        let ntpEpochOffset: TimeInterval = 2208988800 // Seconds between 1900 and 1970
        
        // Extract transmit timestamp (bytes 40-47)
        // This is the server's transmit time (t3)
        let seconds = data.withUnsafeBytes { ptr -> UInt32 in
            ptr.load(fromByteOffset: 40, as: UInt32.self).bigEndian
        }
        let fraction = data.withUnsafeBytes { ptr -> UInt32 in
            ptr.load(fromByteOffset: 44, as: UInt32.self).bigEndian
        }
        
        // Convert to TimeInterval
        let t3 = TimeInterval(seconds) - ntpEpochOffset + TimeInterval(fraction) / 4294967296.0
        
        // For simplicity, assume t2 ≈ t3 (server processing time is negligible)
        // offset ≈ t3 - (t1 + t4) / 2
        // This is the simplified offset calculation
        let t1Unix = t1.timeIntervalSince1970
        let t4Unix = t4.timeIntervalSince1970
        
        let offset = t3 - (t1Unix + t4Unix) / 2
        
        return offset
    }
    
    /// Time since last successful sync
    var timeSinceSync: String? {
        guard let lastSync = lastSyncTime else { return nil }
        
        let seconds = Int(Date().timeIntervalSince(lastSync))
        
        if seconds < 60 {
            return "\(seconds)s ago"
        } else if seconds < 3600 {
            return "\(seconds / 60)m ago"
        } else {
            return "\(seconds / 3600)h ago"
        }
    }
}

enum NTPError: Error {
    case timeout
    case invalidResponse
    case connectionFailed
}
