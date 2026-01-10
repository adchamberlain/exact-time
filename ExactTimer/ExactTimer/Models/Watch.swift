import Foundation
import SwiftData

/// Represents a user's mechanical watch that can be tracked for accuracy
@Model
final class Watch {
    var id: UUID
    var name: String
    var brand: String?
    var createdAt: Date

    /// Reference photo data (JPEG)
    @Attribute(.externalStorage)
    var referencePhotoData: Data?

    /// Extracted hour hand mask (PNG with alpha)
    @Attribute(.externalStorage)
    var hourHandMask: Data?

    /// Extracted minute hand mask (PNG with alpha)
    @Attribute(.externalStorage)
    var minuteHandMask: Data?

    /// Extracted second hand mask (PNG with alpha)
    @Attribute(.externalStorage)
    var secondHandMask: Data?

    /// Center point of the watch face (normalized 0-1)
    var centerX: Double?
    var centerY: Double?

    /// Time shown in the reference photo (for ground truth)
    var referenceHour: Int?
    var referenceMinute: Int?
    var referenceSecond: Int?

    /// Path to the trained Core ML model for this watch
    var trainedModelPath: String?

    /// Whether the model has been trained and is ready for inference
    var isModelTrained: Bool {
        trainedModelPath != nil
    }

    /// Training status
    var trainingStatus: TrainingStatus = TrainingStatus.notStarted

    /// All accuracy readings for this watch
    @Relationship(deleteRule: .cascade, inverse: \WatchReading.watch)
    var readings: [WatchReading] = []

    init(name: String, brand: String? = nil) {
        self.id = UUID()
        self.name = name
        self.brand = brand
        self.createdAt = Date()
    }

    /// Average deviation per day based on readings
    var averageDeviationPerDay: Double? {
        guard readings.count >= 2 else { return nil }

        let sortedReadings = readings.sorted { $0.capturedAt < $1.capturedAt }
        guard let first = sortedReadings.first,
              let last = sortedReadings.last else { return nil }

        let totalDeviation = last.cumulativeDeviation - first.cumulativeDeviation
        let daysBetween = last.capturedAt.timeIntervalSince(first.capturedAt) / 86400

        guard daysBetween > 0 else { return nil }
        return totalDeviation / daysBetween
    }
}

/// Training status for a watch's ML model
enum TrainingStatus: String, Codable {
    case notStarted
    case generatingData
    case training
    case completed
    case failed
}
