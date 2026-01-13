import UIKit
import CoreGraphics
import CoreImage
import CoreImage.CIFilterBuiltins
import Accelerate

/// Generates synthetic training data from a reference watch photo
class SyntheticDataGenerator {

    /// Configuration for data generation
    struct Config {
        var outputSize: CGSize = CGSize(width: 224, height: 224)
        var samplesPerHour: Int = 60  // Generate 60 samples per hour = 720 total for 12 hours
        var includeSeconds: Bool = true
        var augmentationEnabled: Bool = true
    }

    private let config: Config

    init(config: Config = Config()) {
        self.config = config
    }

    /// Training sample with image and labels
    struct TrainingSample {
        let image: CGImage
        let hour: Int      // 0-11
        let minute: Int    // 0-59
        let second: Int    // 0-59
    }

    /// Generate synthetic training data from watch components
    /// - Parameters:
    ///   - dialImage: The watch dial with hands removed (inpainted)
    ///   - hourHandMask: Binary mask of the hour hand
    ///   - minuteHandMask: Binary mask of the minute hand
    ///   - secondHandMask: Binary mask of the second hand (optional)
    ///   - center: Center point of the watch (normalized 0-1)
    ///   - referenceTime: The time shown in the reference photo
    /// - Returns: Array of training samples
    func generateTrainingData(
        dialImage: UIImage,
        hourHandMask: UIImage,
        minuteHandMask: UIImage,
        secondHandMask: UIImage?,
        center: CGPoint,
        referenceTime: (hour: Int, minute: Int, second: Int)
    ) -> [TrainingSample] {
        var samples: [TrainingSample] = []

        // Convert to CGImages for processing
        guard let dialCG = dialImage.cgImage,
              let hourMaskCG = hourHandMask.cgImage,
              let minuteMaskCG = minuteHandMask.cgImage else {
            return samples
        }

        let secondMaskCG = secondHandMask?.cgImage

        // Calculate base angles from reference time
        let refHourAngle = hourAngle(hour: referenceTime.hour, minute: referenceTime.minute)
        let refMinuteAngle = minuteAngle(minute: referenceTime.minute, second: referenceTime.second)
        let refSecondAngle = secondAngle(second: referenceTime.second)

        // Generate samples for each hour
        for hour in 0..<12 {
            let samplesForHour = config.samplesPerHour

            for sampleIdx in 0..<samplesForHour {
                // Distribute minutes evenly
                let minute = (sampleIdx * 60) / samplesForHour
                let second = config.includeSeconds ? Int.random(in: 0..<60) : 0

                // Calculate target angles
                let targetHourAngle = hourAngle(hour: hour, minute: minute)
                let targetMinuteAngle = minuteAngle(minute: minute, second: second)
                let targetSecondAngle = secondAngle(second: second)

                // Calculate rotation deltas from reference position
                let hourRotation = targetHourAngle - refHourAngle
                let minuteRotation = targetMinuteAngle - refMinuteAngle
                let secondRotation = targetSecondAngle - refSecondAngle

                // Composite the image
                if let composited = compositeWatchImage(
                    dial: dialCG,
                    hourMask: hourMaskCG,
                    minuteMask: minuteMaskCG,
                    secondMask: secondMaskCG,
                    center: center,
                    hourRotation: hourRotation,
                    minuteRotation: minuteRotation,
                    secondRotation: secondRotation
                ) {
                    // Apply augmentations if enabled
                    let finalImage = config.augmentationEnabled ?
                        applyAugmentations(to: composited) : composited

                    if let resized = resizeImage(finalImage, to: config.outputSize) {
                        samples.append(TrainingSample(
                            image: resized,
                            hour: hour,
                            minute: minute,
                            second: second
                        ))
                    }
                }
            }
        }

        return samples
    }

    // MARK: - Angle Calculations

    /// Calculate hour hand angle (0 = 12 o'clock, clockwise in radians)
    private func hourAngle(hour: Int, minute: Int) -> CGFloat {
        let hourNormalized = CGFloat(hour % 12) + CGFloat(minute) / 60.0
        return (hourNormalized / 12.0) * 2 * .pi
    }

    /// Calculate minute hand angle
    private func minuteAngle(minute: Int, second: Int) -> CGFloat {
        let minuteNormalized = CGFloat(minute) + CGFloat(second) / 60.0
        return (minuteNormalized / 60.0) * 2 * .pi
    }

    /// Calculate second hand angle
    private func secondAngle(second: Int) -> CGFloat {
        return (CGFloat(second) / 60.0) * 2 * .pi
    }

    // MARK: - Image Compositing

    /// Composite watch dial with rotated hands
    private func compositeWatchImage(
        dial: CGImage,
        hourMask: CGImage,
        minuteMask: CGImage,
        secondMask: CGImage?,
        center: CGPoint,
        hourRotation: CGFloat,
        minuteRotation: CGFloat,
        secondRotation: CGFloat
    ) -> CGImage? {
        let width = dial.width
        let height = dial.height
        let centerPixel = CGPoint(x: CGFloat(width) * center.x, y: CGFloat(height) * center.y)

        // Create drawing context
        guard let context = CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else { return nil }

        let rect = CGRect(x: 0, y: 0, width: width, height: height)

        // Draw dial background
        context.draw(dial, in: rect)

        // Draw rotated hour hand
        drawRotatedHand(context: context, hand: hourMask, center: centerPixel, rotation: hourRotation, rect: rect)

        // Draw rotated minute hand
        drawRotatedHand(context: context, hand: minuteMask, center: centerPixel, rotation: minuteRotation, rect: rect)

        // Draw rotated second hand if present
        if let secondMask = secondMask {
            drawRotatedHand(context: context, hand: secondMask, center: centerPixel, rotation: secondRotation, rect: rect)
        }

        return context.makeImage()
    }

    /// Draw a hand rotated around the center
    private func drawRotatedHand(
        context: CGContext,
        hand: CGImage,
        center: CGPoint,
        rotation: CGFloat,
        rect: CGRect
    ) {
        context.saveGState()

        // Translate to center, rotate, translate back
        context.translateBy(x: center.x, y: center.y)
        context.rotate(by: rotation)
        context.translateBy(x: -center.x, y: -center.y)

        // Draw the hand
        context.draw(hand, in: rect)

        context.restoreGState()
    }

    // MARK: - Augmentations

    /// Apply random augmentations to training image
    private func applyAugmentations(to image: CGImage) -> CGImage {
        var result = image

        // Random brightness adjustment (±10%)
        if Bool.random() {
            let factor = CGFloat.random(in: 0.9...1.1)
            if let adjusted = adjustBrightness(image: result, factor: factor) {
                result = adjusted
            }
        }

        // Random slight rotation (±3 degrees) to simulate imperfect camera alignment
        if Bool.random() {
            let angle = CGFloat.random(in: -0.05...0.05)
            if let rotated = rotateImage(result, by: angle) {
                result = rotated
            }
        }

        return result
    }

    private func adjustBrightness(image: CGImage, factor: CGFloat) -> CGImage? {
        let width = image.width
        let height = image.height

        guard let context = CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else { return nil }

        let rect = CGRect(x: 0, y: 0, width: width, height: height)

        // Draw with brightness adjustment using blend mode
        context.setFillColor(gray: factor > 1 ? 1 : 0, alpha: abs(1 - factor))
        context.draw(image, in: rect)
        context.setBlendMode(factor > 1 ? .lighten : .darken)
        context.fill(rect)

        return context.makeImage()
    }

    private func rotateImage(_ image: CGImage, by angle: CGFloat) -> CGImage? {
        let width = image.width
        let height = image.height

        guard let context = CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else { return nil }

        let rect = CGRect(x: 0, y: 0, width: width, height: height)
        let center = CGPoint(x: CGFloat(width) / 2, y: CGFloat(height) / 2)

        context.translateBy(x: center.x, y: center.y)
        context.rotate(by: angle)
        context.translateBy(x: -center.x, y: -center.y)
        context.draw(image, in: rect)

        return context.makeImage()
    }

    private func resizeImage(_ image: CGImage, to size: CGSize) -> CGImage? {
        guard let context = CGContext(
            data: nil,
            width: Int(size.width),
            height: Int(size.height),
            bitsPerComponent: 8,
            bytesPerRow: Int(size.width) * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else { return nil }

        context.interpolationQuality = .high
        context.draw(image, in: CGRect(origin: .zero, size: size))

        return context.makeImage()
    }
}

// MARK: - Dial Inpainting

extension SyntheticDataGenerator {

    private static let ciContext = CIContext(options: [.useSoftwareRenderer: false])

    /// Inpaint dial to remove hands using CoreImage for smoother results
    /// Uses multi-pass blur-blend approach for natural-looking fills
    func inpaintDial(
        referenceImage: UIImage,
        hourHandMask: UIImage,
        minuteHandMask: UIImage,
        secondHandMask: UIImage?
    ) -> UIImage? {
        // Downscale to reasonable size for processing (max 800px on longest side)
        let maxDimension: CGFloat = 800
        let scale = min(maxDimension / referenceImage.size.width, maxDimension / referenceImage.size.height, 1.0)
        let targetSize = CGSize(
            width: referenceImage.size.width * scale,
            height: referenceImage.size.height * scale
        )

        guard let scaledRef = downscale(referenceImage, to: targetSize),
              let scaledHourMask = downscale(hourHandMask, to: targetSize),
              let scaledMinuteMask = downscale(minuteHandMask, to: targetSize) else {
            return nil
        }

        let scaledSecondMask = secondHandMask.flatMap { downscale($0, to: targetSize) }

        guard let refCG = scaledRef.cgImage,
              let hourMaskCG = scaledHourMask.cgImage,
              let minuteMaskCG = scaledMinuteMask.cgImage else {
            return nil
        }

        let width = refCG.width
        let height = refCG.height

        // Create combined mask of all hands
        guard let combinedMask = createCombinedMask(
            hourMask: hourMaskCG,
            minuteMask: minuteMaskCG,
            secondMask: scaledSecondMask?.cgImage,
            width: width,
            height: height
        ) else {
            return nil
        }

        // Use CoreImage-based inpainting for smoother results
        guard let inpainted = inpaintWithCoreImage(image: refCG, mask: combinedMask) else {
            // Fallback to simple inpainting if CoreImage fails
            guard let fallback = simpleInpaint(image: refCG, mask: combinedMask) else {
                return nil
            }
            return UIImage(cgImage: fallback)
        }

        return UIImage(cgImage: inpainted)
    }

    /// Create combined mask from all hand masks
    private func createCombinedMask(
        hourMask: CGImage,
        minuteMask: CGImage,
        secondMask: CGImage?,
        width: Int,
        height: Int
    ) -> CGImage? {
        guard let maskContext = CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width,
            space: CGColorSpaceCreateDeviceGray(),
            bitmapInfo: CGImageAlphaInfo.none.rawValue
        ) else { return nil }

        let rect = CGRect(x: 0, y: 0, width: width, height: height)

        // Clear to black (no mask)
        maskContext.setFillColor(gray: 0, alpha: 1)
        maskContext.fill(rect)

        // Add hand masks in white using lighten blend
        maskContext.setBlendMode(.lighten)
        maskContext.draw(hourMask, in: rect)
        maskContext.draw(minuteMask, in: rect)
        if let secondMask = secondMask {
            maskContext.draw(secondMask, in: rect)
        }

        return maskContext.makeImage()
    }

    /// CoreImage-based inpainting using multi-pass blur and blend
    /// Creates smoother transitions than simple nearest-neighbor fill
    private func inpaintWithCoreImage(image: CGImage, mask: CGImage) -> CGImage? {
        let ciImage = CIImage(cgImage: image)
        let ciMask = CIImage(cgImage: mask)

        // Dilate the mask slightly for better edge coverage
        let dilatedMask = dilateMask(ciMask, radius: 3)

        // Create heavily blurred version of the image
        guard let blurFilter = CIFilter(name: "CIGaussianBlur") else { return nil }
        blurFilter.setValue(ciImage, forKey: kCIInputImageKey)
        blurFilter.setValue(30.0, forKey: kCIInputRadiusKey)

        guard let blurredImage = blurFilter.outputImage else { return nil }

        // First pass: blend blurred into masked areas
        guard let firstPass = blendWithMask(
            background: ciImage,
            foreground: blurredImage,
            mask: dilatedMask
        ) else { return nil }

        // Apply medium blur to first pass result
        blurFilter.setValue(firstPass, forKey: kCIInputImageKey)
        blurFilter.setValue(15.0, forKey: kCIInputRadiusKey)

        guard let mediumBlurred = blurFilter.outputImage else { return nil }

        // Second pass: blend medium blur to smooth edges
        let smallerMask = dilateMask(ciMask, radius: 1)
        guard let secondPass = blendWithMask(
            background: firstPass,
            foreground: mediumBlurred,
            mask: smallerMask
        ) else { return nil }

        // Render final result
        let extent = ciImage.extent
        return Self.ciContext.createCGImage(secondPass, from: extent)
    }

    /// Dilate mask to expand coverage area
    private func dilateMask(_ mask: CIImage, radius: Int) -> CIImage {
        guard let dilateFilter = CIFilter(name: "CIMorphologyMaximum") else {
            return mask
        }
        dilateFilter.setValue(mask, forKey: kCIInputImageKey)
        dilateFilter.setValue(Float(radius), forKey: kCIInputRadiusKey)
        return dilateFilter.outputImage ?? mask
    }

    /// Blend foreground into background using mask
    private func blendWithMask(background: CIImage, foreground: CIImage, mask: CIImage) -> CIImage? {
        guard let blendFilter = CIFilter(name: "CIBlendWithMask") else { return nil }
        blendFilter.setValue(background, forKey: kCIInputBackgroundImageKey)
        blendFilter.setValue(foreground, forKey: kCIInputImageKey)
        blendFilter.setValue(mask, forKey: kCIInputMaskImageKey)
        return blendFilter.outputImage
    }

    /// Fallback simple inpainting using nearest-neighbor fill
    private func simpleInpaint(image: CGImage, mask: CGImage) -> CGImage? {
        let width = image.width
        let height = image.height

        guard let imageContext = CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else { return nil }

        imageContext.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))

        guard let maskContext = CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width,
            space: CGColorSpaceCreateDeviceGray(),
            bitmapInfo: CGImageAlphaInfo.none.rawValue
        ) else { return nil }

        maskContext.draw(mask, in: CGRect(x: 0, y: 0, width: width, height: height))

        guard let imageData = imageContext.data,
              let maskData = maskContext.data else { return nil }

        let imagePixels = imageData.bindMemory(to: UInt8.self, capacity: width * height * 4)
        let maskPixels = maskData.bindMemory(to: UInt8.self, capacity: width * height)

        for y in 0..<height {
            for x in 0..<width {
                let maskIdx = y * width + x
                if maskPixels[maskIdx] > 128 {
                    if let (r, g, b) = findNearestUnmaskedColor(
                        x: x, y: y,
                        imagePixels: imagePixels,
                        maskPixels: maskPixels,
                        width: width,
                        height: height
                    ) {
                        let pixelIdx = (y * width + x) * 4
                        imagePixels[pixelIdx] = r
                        imagePixels[pixelIdx + 1] = g
                        imagePixels[pixelIdx + 2] = b
                    }
                }
            }
        }

        return imageContext.makeImage()
    }

    private func findNearestUnmaskedColor(
        x: Int, y: Int,
        imagePixels: UnsafeMutablePointer<UInt8>,
        maskPixels: UnsafeMutablePointer<UInt8>,
        width: Int, height: Int
    ) -> (UInt8, UInt8, UInt8)? {
        for radius in 1..<50 {
            for dy in -radius...radius {
                for dx in -radius...radius {
                    if abs(dx) != radius && abs(dy) != radius { continue }

                    let nx = x + dx
                    let ny = y + dy

                    if nx >= 0 && nx < width && ny >= 0 && ny < height {
                        let maskIdx = ny * width + nx
                        if maskPixels[maskIdx] < 128 {
                            let pixelIdx = (ny * width + nx) * 4
                            return (
                                imagePixels[pixelIdx],
                                imagePixels[pixelIdx + 1],
                                imagePixels[pixelIdx + 2]
                            )
                        }
                    }
                }
            }
        }
        return nil
    }

    /// Downscale image to target size
    private func downscale(_ image: UIImage, to targetSize: CGSize) -> UIImage? {
        UIGraphicsBeginImageContextWithOptions(targetSize, false, 1.0)
        image.draw(in: CGRect(origin: .zero, size: targetSize))
        let scaledImage = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        return scaledImage
    }
}
