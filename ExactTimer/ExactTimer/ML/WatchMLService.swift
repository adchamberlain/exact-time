import Foundation
import CoreML
import Vision
import UIKit

/// Service for training and running inference with watch time prediction models
/// Version 2: Uses Vision framework for feature extraction and weighted interpolation for prediction
@MainActor
class WatchMLService: ObservableObject {

    /// Training progress (0-1)
    @Published var trainingProgress: Double = 0

    /// Current status message
    @Published var statusMessage: String = ""

    /// Whether training is in progress
    @Published var isTraining: Bool = false

    /// Error message if training failed
    @Published var errorMessage: String?

    private let dataGenerator = SyntheticDataGenerator()

    /// Feature dimension from Vision framework's VNFeaturePrintObservation
    private let featureDimension = 2048

    /// Number of nearest neighbors for prediction
    private let kNeighbors = 7

    /// Prediction result from the ML model
    struct TimePrediction {
        let hour: Int
        let minute: Int
        let second: Int
        let confidence: Double

        var timeString: String {
            String(format: "%d:%02d:%02d", hour, minute, second)
        }
    }

    // MARK: - Training

    /// Train a model for the given watch
    func trainModel(for watch: Watch) async throws -> URL {
        isTraining = true
        errorMessage = nil

        do {
            // Step 1: Validate watch has required data
            statusMessage = "Validating watch data..."
            trainingProgress = 0.05

            // Give UI a chance to update
            try await Task.sleep(nanoseconds: 100_000_000)

            guard let referenceData = watch.referencePhotoData,
                  let referenceImage = UIImage(data: referenceData),
                  let hourMaskData = watch.hourHandMask,
                  let hourMask = UIImage(data: hourMaskData),
                  let minuteMaskData = watch.minuteHandMask,
                  let minuteMask = UIImage(data: minuteMaskData),
                  let centerX = watch.centerX,
                  let centerY = watch.centerY,
                  let refHour = watch.referenceHour,
                  let refMinute = watch.referenceMinute,
                  let refSecond = watch.referenceSecond else {
                throw WatchMLError.missingWatchData
            }

            let secondMask: UIImage? = watch.secondHandMask.flatMap { UIImage(data: $0) }
            let center = CGPoint(x: centerX, y: centerY)

            // Step 2: Inpaint dial (remove hands) - run off main thread
            statusMessage = "Preparing dial image..."
            trainingProgress = 0.1

            // Give UI a chance to update
            try await Task.sleep(nanoseconds: 50_000_000)

            let cleanDial = await Task.detached { [dataGenerator] in
                return dataGenerator.inpaintDial(
                    referenceImage: referenceImage,
                    hourHandMask: hourMask,
                    minuteHandMask: minuteMask,
                    secondHandMask: secondMask
                )
            }.value

            guard let cleanDial else {
                throw WatchMLError.inpaintingFailed
            }

            // Step 3: Generate synthetic training data
            statusMessage = "Generating training data..."
            trainingProgress = 0.2

            // Give UI a chance to update
            try await Task.sleep(nanoseconds: 50_000_000)

            let samples = await Task.detached { [dataGenerator] in
                return dataGenerator.generateTrainingData(
                    dialImage: cleanDial,
                    hourHandMask: hourMask,
                    minuteHandMask: minuteMask,
                    secondHandMask: secondMask,
                    center: center,
                    referenceTime: (hour: refHour, minute: refMinute, second: refSecond)
                )
            }.value

            guard !samples.isEmpty else {
                throw WatchMLError.dataGenerationFailed
            }

            statusMessage = "Generated \(samples.count) training samples"
            trainingProgress = 0.4

            // Step 4: Create and train the model
            statusMessage = "Training model..."

            let modelURL = try await trainCoreMLModel(
                samples: samples,
                watchId: watch.id
            )

            trainingProgress = 1.0
            statusMessage = "Training complete!"
            isTraining = false

            return modelURL

        } catch {
            isTraining = false
            errorMessage = error.localizedDescription
            throw error
        }
    }

    /// Train a model on the synthetic data using Vision framework features
    private func trainCoreMLModel(
        samples: [SyntheticDataGenerator.TrainingSample],
        watchId: UUID
    ) async throws -> URL {
        // Create directory for watch model
        let documentsDir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let watchDir = documentsDir.appendingPathComponent("watches/\(watchId.uuidString)")

        try FileManager.default.createDirectory(at: watchDir, withIntermediateDirectories: true)

        statusMessage = "Extracting features with Vision framework..."
        trainingProgress = 0.45

        // Extract Vision features from each sample
        var trainingSamples: [ModelDataV2.Sample] = []
        let totalSamples = samples.count
        var processedCount = 0

        // Process samples in batches to avoid memory issues
        let batchSize = 50
        for batchStart in stride(from: 0, to: samples.count, by: batchSize) {
            let batchEnd = min(batchStart + batchSize, samples.count)
            let batch = samples[batchStart..<batchEnd]

            for sample in batch {
                let image = UIImage(cgImage: sample.image)

                // Extract features using Vision framework (off main thread)
                let featuresResult: [Float]? = await Task.detached {
                    return try? await self.extractFeaturesWithVision(from: image)
                }.value

                if let features = featuresResult {
                    trainingSamples.append(ModelDataV2.Sample(
                        features: features,
                        hour: sample.hour,
                        minute: sample.minute,
                        second: sample.second
                    ))
                }

                processedCount += 1
                let progress = 0.45 + (0.45 * Double(processedCount) / Double(totalSamples))
                await MainActor.run {
                    self.trainingProgress = progress
                    if processedCount % 100 == 0 {
                        self.statusMessage = "Extracting features... \(processedCount)/\(totalSamples)"
                    }
                }
            }
        }

        statusMessage = "Saving model..."
        trainingProgress = 0.92

        // Save the v2 model with Vision features and exact time labels
        let modelData = ModelDataV2(
            version: 2,
            featureDimension: featureDimension,
            createdAt: Date(),
            samples: trainingSamples
        )

        let modelURL = watchDir.appendingPathComponent("model.json")
        let encoder = JSONEncoder()
        let data = try encoder.encode(modelData)
        try data.write(to: modelURL)

        trainingProgress = 0.95
        statusMessage = "Finalizing model..."

        return modelURL
    }

    // MARK: - Feature Extraction

    /// Extract features using Vision framework's VNFeaturePrintObservation
    /// This provides 2048-dimensional neural network features that capture semantic content
    private func extractFeaturesWithVision(from image: UIImage) async throws -> [Float] {
        guard let cgImage = image.cgImage else {
            throw WatchMLError.featureExtractionFailed
        }

        return try await withCheckedThrowingContinuation { continuation in
            let request = VNGenerateImageFeaturePrintRequest { request, error in
                if let error = error {
                    continuation.resume(throwing: error)
                    return
                }

                guard let observation = request.results?.first as? VNFeaturePrintObservation else {
                    continuation.resume(throwing: WatchMLError.featureExtractionFailed)
                    return
                }

                // Extract the feature vector data
                let elementCount = observation.elementCount
                var features = [Float](repeating: 0, count: elementCount)

                // Copy feature data
                let data = observation.data
                data.withUnsafeBytes { (buffer: UnsafeRawBufferPointer) in
                    let floatBuffer = buffer.bindMemory(to: Float.self)
                    for i in 0..<elementCount {
                        features[i] = floatBuffer[i]
                    }
                }

                continuation.resume(returning: features)
            }

            let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
            do {
                try handler.perform([request])
            } catch {
                continuation.resume(throwing: error)
            }
        }
    }

    // MARK: - Inference

    /// Predict time from an image using a trained model with weighted interpolation
    func predict(image: UIImage, modelPath: String) async throws -> TimePrediction {
        let modelURL = URL(fileURLWithPath: modelPath)

        guard FileManager.default.fileExists(atPath: modelPath) else {
            throw WatchMLError.modelNotFound
        }

        // Load model (supports both v1 and v2 formats)
        let data = try Data(contentsOf: modelURL)
        let decoder = JSONDecoder()

        // Try v2 format first, fall back to v1
        let modelData: ModelDataV2
        if let v2Data = try? decoder.decode(ModelDataV2.self, from: data), v2Data.version == 2 {
            modelData = v2Data
        } else if let v1Data = try? decoder.decode(KNNModelData.self, from: data) {
            // Convert v1 to v2 format for processing
            modelData = ModelDataV2(
                version: 1,
                featureDimension: 1024,
                createdAt: Date(),
                samples: v1Data.labels.enumerated().map { idx, label in
                    ModelDataV2.Sample(
                        features: v1Data.featureVectors[idx],
                        hour: label.hour,
                        minute: label.minute,
                        second: 0
                    )
                }
            )
        } else {
            throw WatchMLError.modelNotFound
        }

        // Extract features from input image using Vision framework
        let queryFeatures = try await extractFeaturesWithVision(from: image)

        // Find k nearest neighbors with distances
        let neighbors = findKNearestNeighbors(
            queryFeatures: queryFeatures,
            samples: modelData.samples,
            k: kNeighbors
        )

        // Use weighted interpolation to predict continuous time
        let prediction = weightedInterpolation(neighbors: neighbors)

        return prediction
    }

    /// Find k nearest neighbors using Euclidean distance
    private func findKNearestNeighbors(
        queryFeatures: [Float],
        samples: [ModelDataV2.Sample],
        k: Int
    ) -> [(distance: Float, hour: Int, minute: Int, second: Int)] {
        let distances = samples.map { sample -> (distance: Float, hour: Int, minute: Int, second: Int) in
            let dist = euclideanDistance(queryFeatures, sample.features)
            return (dist, sample.hour, sample.minute, sample.second)
        }

        return Array(distances.sorted { $0.distance < $1.distance }.prefix(k))
    }

    /// Weighted interpolation for continuous time prediction
    /// Handles circular nature of time (e.g., 11:59 -> 12:00 wraparound)
    private func weightedInterpolation(
        neighbors: [(distance: Float, hour: Int, minute: Int, second: Int)]
    ) -> TimePrediction {
        guard !neighbors.isEmpty else {
            return TimePrediction(hour: 0, minute: 0, second: 0, confidence: 0)
        }

        // Convert times to angles (radians) for circular interpolation
        // Hour angle: 0-12 hours maps to 0-2π
        // Minute angle: 0-60 minutes maps to 0-2π
        // Second angle: 0-60 seconds maps to 0-2π

        var totalWeight: Double = 0
        var hourSinSum: Double = 0
        var hourCosSum: Double = 0
        var minuteSinSum: Double = 0
        var minuteCosSum: Double = 0
        var secondSinSum: Double = 0
        var secondCosSum: Double = 0

        let minDistance = neighbors.first?.distance ?? 1.0
        let epsilon: Float = 0.0001

        for neighbor in neighbors {
            // Inverse distance weighting (add epsilon to avoid division by zero)
            let weight = Double(1.0 / (neighbor.distance + epsilon))
            totalWeight += weight

            // Convert hour to angle (12-hour clock)
            let hourAngle = Double(neighbor.hour % 12) / 12.0 * 2.0 * .pi
            hourSinSum += sin(hourAngle) * weight
            hourCosSum += cos(hourAngle) * weight

            // Convert minute to angle
            let minuteAngle = Double(neighbor.minute) / 60.0 * 2.0 * .pi
            minuteSinSum += sin(minuteAngle) * weight
            minuteCosSum += cos(minuteAngle) * weight

            // Convert second to angle
            let secondAngle = Double(neighbor.second) / 60.0 * 2.0 * .pi
            secondSinSum += sin(secondAngle) * weight
            secondCosSum += cos(secondAngle) * weight
        }

        // Average the angles using atan2 for proper circular mean
        let avgHourAngle = atan2(hourSinSum / totalWeight, hourCosSum / totalWeight)
        let avgMinuteAngle = atan2(minuteSinSum / totalWeight, minuteCosSum / totalWeight)
        let avgSecondAngle = atan2(secondSinSum / totalWeight, secondCosSum / totalWeight)

        // Convert back to time values
        var hour = Int(round((avgHourAngle < 0 ? avgHourAngle + 2.0 * .pi : avgHourAngle) / (2.0 * .pi) * 12.0))
        var minute = Int(round((avgMinuteAngle < 0 ? avgMinuteAngle + 2.0 * .pi : avgMinuteAngle) / (2.0 * .pi) * 60.0))
        var second = Int(round((avgSecondAngle < 0 ? avgSecondAngle + 2.0 * .pi : avgSecondAngle) / (2.0 * .pi) * 60.0))

        // Normalize values
        hour = hour % 12
        minute = minute % 60
        second = second % 60

        // Calculate confidence based on neighbor agreement
        // Higher confidence when neighbors are close together and agree on time
        let confidence = calculateConfidence(neighbors: neighbors, predictedHour: hour, predictedMinute: minute)

        return TimePrediction(
            hour: hour,
            minute: minute,
            second: second,
            confidence: confidence
        )
    }

    /// Calculate confidence based on how well neighbors agree
    private func calculateConfidence(
        neighbors: [(distance: Float, hour: Int, minute: Int, second: Int)],
        predictedHour: Int,
        predictedMinute: Int
    ) -> Double {
        guard !neighbors.isEmpty else { return 0 }

        // Calculate average time deviation from prediction
        var totalDeviation: Double = 0
        for neighbor in neighbors {
            // Time difference in minutes (handling 12-hour wraparound)
            let neighborMinutes = neighbor.hour * 60 + neighbor.minute
            let predictedMinutes = predictedHour * 60 + predictedMinute

            var diff = abs(neighborMinutes - predictedMinutes)
            // Handle wraparound (12 hours = 720 minutes)
            if diff > 360 {
                diff = 720 - diff
            }
            totalDeviation += Double(diff)
        }

        let avgDeviation = totalDeviation / Double(neighbors.count)

        // Convert deviation to confidence (0-30 minute deviation maps to 1.0-0.0)
        let maxDeviation: Double = 30.0
        let confidence = max(0, 1.0 - (avgDeviation / maxDeviation))

        return confidence
    }

    private func euclideanDistance(_ a: [Float], _ b: [Float]) -> Float {
        var sum: Float = 0
        let count = min(a.count, b.count)
        for i in 0..<count {
            let diff = a[i] - b[i]
            sum += diff * diff
        }
        return sqrt(sum)
    }
}

// MARK: - Supporting Types

/// V2 model format with Vision framework features and exact time labels
struct ModelDataV2: Codable {
    let version: Int
    let featureDimension: Int
    let createdAt: Date
    let samples: [Sample]

    struct Sample: Codable {
        let features: [Float]
        let hour: Int
        let minute: Int
        let second: Int
    }
}

/// V1 model format (for backwards compatibility)
struct KNNModelData: Codable {
    let featureVectors: [[Float]]
    let labels: [TimeLabel]

    struct TimeLabel: Codable {
        let hour: Int
        let minute: Int
    }

    init(featureVectors: [[Float]], labels: [(hour: Int, minute: Int)]) {
        self.featureVectors = featureVectors
        self.labels = labels.map { TimeLabel(hour: $0.hour, minute: $0.minute) }
    }
}

enum WatchMLError: LocalizedError {
    case missingWatchData
    case inpaintingFailed
    case dataGenerationFailed
    case modelNotFound
    case featureExtractionFailed
    case predictionFailed
    case trainingFailed(String)

    var errorDescription: String? {
        switch self {
        case .missingWatchData:
            return "Watch is missing required data (reference photo, hand masks, or center point)"
        case .inpaintingFailed:
            return "Failed to remove hands from dial image"
        case .dataGenerationFailed:
            return "Failed to generate training data"
        case .modelNotFound:
            return "Trained model not found"
        case .featureExtractionFailed:
            return "Failed to extract features from image"
        case .predictionFailed:
            return "Failed to make prediction"
        case .trainingFailed(let message):
            return "Training failed: \(message)"
        }
    }
}
