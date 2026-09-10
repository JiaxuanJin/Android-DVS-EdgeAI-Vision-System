package com.example.framesenderapp

import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.Matrix
import androidx.camera.core.ImageProxy
import kotlin.math.abs

// Result returned after processing one camera frame.
// bitmap = DVS-like ON/OFF event map
// grayscaleBitmap = current luminance / grayscale image
// differenceBitmap = motion-compensated gray frame-difference image
// threshold = manual threshold, or average per-pixel threshold in auto mode
data class DvsResult(
    val bitmap: Bitmap,
    val grayscaleBitmap: Bitmap,
    val differenceBitmap: Bitmap,
    val frameCount: Int,
    val width: Int,
    val height: Int,
    val threshold: Int,
    val meanDiff: Double,
    val onEvents: Int,
    val offEvents: Int,
    val totalEvents: Int
)

class DvsProcessor {

    // -----------------------------
    // User threshold control
    // -----------------------------

    @Volatile
    private var autoThresholdEnabled = false

    @Volatile
    private var manualThreshold = 15

    private val manualMinThreshold = 5
    private val manualMaxThreshold = 80

    // Auto mode is independent from the UI slider.
    // When Auto is enabled, per-pixel thresholds start from this fixed value.
    private val autoBaseThreshold = 24.0f
    private val autoMinThreshold = 10.0f
    private val autoMaxThreshold = 60.0f

    // Per-pixel adaptive threshold update.
    // Event -> threshold increases, no event -> threshold decreases.
    private val eventThresholdIncrease = 1.05f
    private val noEventThresholdDecrease = 0.16f

    // Slowly pull thresholds back toward the base value so Auto will not get stuck high.
    private val autoBaseRelaxAlpha = 0.018f

    // -----------------------------
    // Sampling and display parameters
    // -----------------------------

    // 4 means one event pixel represents every 4x4 camera pixels.
    private val sampleStep = 4

    // Gain used to make the gray difference image more visible.
    private val differenceGain = 5.0

    // -----------------------------
    // Global motion compensation
    // -----------------------------

    // These parameters estimate small camera hand-shake as a global x/y translation.
    // It is a lightweight version of video stabilisation for the DVS emulator.
    private val motionMaxShift = 10        // pixels searched in each direction
    private val motionCandidateStep = 2    // search every 2 pixels for speed
    private val motionSampleStride = 16    // sample every 16 pixels for SAD matching

    // Use motion compensation only when the best shifted match is clearly better than no shift.
    private val minShiftForCompensation = 2
    private val minSadImprovementRatio = 0.08
    private val maxBestMeanSadForReliableMatch = 36.0

    // If compensation is not reliable but the frame still looks globally unstable,
    // mildly raise the threshold instead of blanking the image.
    private val fallbackShakeThresholdMultiplier = 1.35f
    private val fallbackDifferenceThresholdMultiplier = 1.25f


    // -----------------------------
    // IMU-based camera motion awareness
    // -----------------------------

    // Updated by MainActivity from CameraMotionDetector.
    // This does not physically stabilise the image. It only makes event generation
    // more conservative when the phone itself is moving.
    @Volatile
    private var cameraIsMoving = false

    @Volatile
    private var cameraMotionLevel = 0.0f

    // Larger value = stronger suppression when the phone shakes.
    private val imuEventThresholdGain = 1.10f
    private val imuDifferenceThresholdGain = 0.90f
    private val imuMaxThresholdMultiplier = 3.20f

    // -----------------------------
    // Internal state
    // -----------------------------

    // Previous luminance frame for Difference Frame Mode.
    private var previousLuma: ByteArray? = null

    // DVS reference frame. DVS events are generated from current frame - reference frame,
    // not simply current frame - previous frame.
    private var referenceLuma: ByteArray? = null

    // Per sampled-pixel threshold array used in Auto mode.
    private var adaptiveThresholds: FloatArray? = null

    private var frameCount = 0

    private data class MotionShift(
        val dx: Int,
        val dy: Int,
        val bestMeanSad: Double,
        val zeroMeanSad: Double
    ) {
        val magnitude: Int
            get() = abs(dx) + abs(dy)

        val improvementRatio: Double
            get() = if (zeroMeanSad > 0.0) (zeroMeanSad - bestMeanSad) / zeroMeanSad else 0.0
    }

    // Called by MainActivity when the user changes manual / auto threshold controls.
    fun setThresholdControl(auto: Boolean, threshold: Int) {
        val wasAuto = autoThresholdEnabled
        autoThresholdEnabled = auto

        if (!auto) {
            manualThreshold = threshold.coerceIn(manualMinThreshold, manualMaxThreshold)
        }

        // Auto mode must not depend on the current SeekBar value.
        // When switching into Auto, reset thresholds to the fixed auto base.
        if (auto && !wasAuto) {
            adaptiveThresholds?.fill(autoBaseThreshold)
        }
    }

    // Called by MainActivity when IMU camera motion detection updates.
    fun setCameraMotion(isMoving: Boolean, motionLevel: Float) {
        cameraIsMoving = isMoving
        cameraMotionLevel = motionLevel.coerceIn(0.0f, 3.0f)
    }

    // Reset processor state when camera starts or stops.
    fun reset() {
        previousLuma = null
        referenceLuma = null
        adaptiveThresholds = null
        frameCount = 0
    }

    fun process(imageProxy: ImageProxy): DvsResult? {
        frameCount++

        val width = imageProxy.width
        val height = imageProxy.height
        val currentLuma = extractLuma(imageProxy)

        val previous = previousLuma
        val reference = referenceLuma

        if (previous == null || reference == null || previous.size != currentLuma.size || reference.size != currentLuma.size) {
            previousLuma = currentLuma.copyOf()
            referenceLuma = currentLuma.copyOf()
            return null
        }

        val eventWidth = (width + sampleStep - 1) / sampleStep
        val eventHeight = (height + sampleStep - 1) / sampleStep
        val sampledPixelCount = eventWidth * eventHeight

        ensureAdaptiveThresholds(sampledPixelCount)
        val thresholdArray = adaptiveThresholds!!

        // ---------------------------------------------------------
        //  estimate global camera motion.
        // shiftToPrevious is mainly used for gray Difference Frame Mode.
        // shiftToReference is used for DVS event generation.
        // The shift means: current(x, y) is compared with target(x + dx, y + dy).
        // ---------------------------------------------------------
        val shiftToPrevious = estimateGlobalShift(
            current = currentLuma,
            target = previous,
            width = width,
            height = height
        )

        val shiftToReference = estimateGlobalShift(
            current = currentLuma,
            target = reference,
            width = width,
            height = height
        )

        val usePreviousMotionComp = isReliableMotionCompensation(shiftToPrevious)
        val useReferenceMotionComp = isReliableMotionCompensation(shiftToReference)

        // ---------------------------------------------------------
        //  mean motion-compensated frame-to-frame difference for statistics.
        // This is more meaningful than raw difference when the phone shakes.
        // ---------------------------------------------------------
        var sumAbsDiff = 0.0
        var sampleCount = 0

        for (y in 0 until height step sampleStep) {
            for (x in 0 until width step sampleStep) {
                val currentIndex = y * width + x
                val previousIndex = getShiftedIndex(
                    x = x,
                    y = y,
                    width = width,
                    height = height,
                    dx = if (usePreviousMotionComp) shiftToPrevious.dx else 0,
                    dy = if (usePreviousMotionComp) shiftToPrevious.dy else 0
                )

                if (previousIndex >= 0) {
                    val currentPixel = currentLuma[currentIndex].toInt() and 0xFF
                    val previousPixel = previous[previousIndex].toInt() and 0xFF
                    sumAbsDiff += abs(currentPixel - previousPixel).toDouble()
                    sampleCount++
                }
            }
        }

        val meanAbsDiff = if (sampleCount > 0) sumAbsDiff / sampleCount else 0.0

        // ---------------------------------------------------------
        // generate DVS event map using the reference frame.
        // If the whole image moved because of phone shake, compare with a shifted
        // reference frame. This prevents static background edges from becoming events.
        // ---------------------------------------------------------
        var onEvents = 0
        var offEvents = 0
        val eventPixels = IntArray(sampledPixelCount) { Color.BLACK }

        val fallbackGlobalMotion =
            !useReferenceMotionComp && shiftToReference.magnitude >= minShiftForCompensation

        for (y in 0 until height step sampleStep) {
            for (x in 0 until width step sampleStep) {
                val currentIndex = y * width + x
                val eventX = x / sampleStep
                val eventY = y / sampleStep
                val eventIndex = eventY * eventWidth + eventX

                val referenceIndex = getShiftedIndex(
                    x = x,
                    y = y,
                    width = width,
                    height = height,
                    dx = if (useReferenceMotionComp) shiftToReference.dx else 0,
                    dy = if (useReferenceMotionComp) shiftToReference.dy else 0
                )

                if (referenceIndex < 0) {
                    updateAutoThreshold(thresholdArray, eventIndex, false)
                    continue
                }

                val currentPixel = currentLuma[currentIndex].toInt() and 0xFF
                val referencePixel = reference[referenceIndex].toInt() and 0xFF
                val diffFromReference = currentPixel - referencePixel

                val localThreshold = if (autoThresholdEnabled) {
                    thresholdArray[eventIndex]
                } else {
                    manualThreshold.toFloat()
                }

                val globalMotionMultiplier = if (fallbackGlobalMotion) {
                    fallbackShakeThresholdMultiplier
                } else {
                    1.0f
                }

                val effectiveThreshold = localThreshold * maxOf(
                    globalMotionMultiplier,
                    getImuEventThresholdMultiplier()
                )

                var eventGenerated = false

                if (diffFromReference >= effectiveThreshold) {
                    onEvents++
                    eventGenerated = true
                    eventPixels[eventIndex] = Color.WHITE
                    reference[referenceIndex] = currentLuma[currentIndex]
                } else if (diffFromReference <= -effectiveThreshold) {
                    offEvents++
                    eventGenerated = true
                    eventPixels[eventIndex] = Color.RED
                    reference[referenceIndex] = currentLuma[currentIndex]
                }

                updateAutoThreshold(thresholdArray, eventIndex, eventGenerated)
            }
        }

        val totalEvents = onEvents + offEvents

        val displayThreshold = if (autoThresholdEnabled) {
            thresholdArray.average().toInt().coerceIn(autoMinThreshold.toInt(), autoMaxThreshold.toInt())
        } else {
            manualThreshold
        }

        val eventBitmap = Bitmap.createBitmap(
            eventPixels,
            eventWidth,
            eventHeight,
            Bitmap.Config.ARGB_8888
        )

        val grayscaleBitmap = createGrayscaleBitmap(currentLuma, width, height)

        // Difference Frame Mode stays as the gray emboss-style display you wanted.
        // It now uses motion compensation so static objects do not create strong edges
        // when the phone shakes slightly.
        val differenceBitmap = createMotionCompensatedDifferenceBitmap(
            currentLuma = currentLuma,
            previousLuma = previous,
            width = width,
            height = height,
            threshold = displayThreshold,
            shift = shiftToPrevious,
            useMotionCompensation = usePreviousMotionComp,
            fallbackGlobalMotion = !usePreviousMotionComp && shiftToPrevious.magnitude >= minShiftForCompensation,
            imuThresholdMultiplier = getImuDifferenceThresholdMultiplier()
        )

        val rotationDegrees = imageProxy.imageInfo.rotationDegrees

        val rotatedEventBitmap = rotateBitmapIfNeeded(eventBitmap, rotationDegrees)
        val rotatedGrayscaleBitmap = rotateBitmapIfNeeded(grayscaleBitmap, rotationDegrees)
        val rotatedDifferenceBitmap = rotateBitmapIfNeeded(differenceBitmap, rotationDegrees)

        previousLuma = currentLuma.copyOf()
        referenceLuma = reference

        return DvsResult(
            bitmap = rotatedEventBitmap,
            grayscaleBitmap = rotatedGrayscaleBitmap,
            differenceBitmap = rotatedDifferenceBitmap,
            frameCount = frameCount,
            width = width,
            height = height,
            threshold = displayThreshold,
            meanDiff = meanAbsDiff,
            onEvents = onEvents,
            offEvents = offEvents,
            totalEvents = totalEvents
        )
    }

    private fun getImuEventThresholdMultiplier(): Float {
        return if (cameraIsMoving) {
            (1.0f + cameraMotionLevel * imuEventThresholdGain)
                .coerceIn(1.0f, imuMaxThresholdMultiplier)
        } else {
            1.0f
        }
    }

    private fun getImuDifferenceThresholdMultiplier(): Float {
        return if (cameraIsMoving) {
            (1.0f + cameraMotionLevel * imuDifferenceThresholdGain)
                .coerceIn(1.0f, imuMaxThresholdMultiplier)
        } else {
            1.0f
        }
    }

    private fun updateAutoThreshold(
        thresholdArray: FloatArray,
        eventIndex: Int,
        eventGenerated: Boolean
    ) {
        if (!autoThresholdEnabled) return

        val localThreshold = thresholdArray[eventIndex]

        // Relax toward base first so Auto cannot remain stuck at a high value.
        val relaxedThreshold = localThreshold + autoBaseRelaxAlpha * (autoBaseThreshold - localThreshold)

        thresholdArray[eventIndex] = if (cameraIsMoving) {
            // If the phone itself is moving, many events are caused by camera motion.
            // Do not let those false/background events push the per-pixel thresholds too high.
            if (eventGenerated) {
                (relaxedThreshold + eventThresholdIncrease * 0.25f)
                    .coerceIn(autoMinThreshold, autoMaxThreshold)
            } else {
                (relaxedThreshold - noEventThresholdDecrease * 0.40f)
                    .coerceIn(autoMinThreshold, autoMaxThreshold)
            }
        } else {
            if (eventGenerated) {
                (relaxedThreshold + eventThresholdIncrease).coerceIn(autoMinThreshold, autoMaxThreshold)
            } else {
                (relaxedThreshold - noEventThresholdDecrease).coerceIn(autoMinThreshold, autoMaxThreshold)
            }
        }
    }

    private fun ensureAdaptiveThresholds(size: Int) {
        if (adaptiveThresholds == null || adaptiveThresholds!!.size != size) {
            adaptiveThresholds = FloatArray(size) { autoBaseThreshold }
        }
    }

    private fun isReliableMotionCompensation(shift: MotionShift): Boolean {
        return shift.magnitude >= minShiftForCompensation &&
                shift.improvementRatio >= minSadImprovementRatio &&
                shift.bestMeanSad <= maxBestMeanSadForReliableMatch
    }

    private fun estimateGlobalShift(
        current: ByteArray,
        target: ByteArray,
        width: Int,
        height: Int
    ): MotionShift {
        var bestDx = 0
        var bestDy = 0
        var bestMeanSad = Double.MAX_VALUE
        var zeroMeanSad = Double.MAX_VALUE

        for (dy in -motionMaxShift..motionMaxShift step motionCandidateStep) {
            for (dx in -motionMaxShift..motionMaxShift step motionCandidateStep) {
                var sad = 0L
                var count = 0

                var y = motionMaxShift
                while (y < height - motionMaxShift) {
                    var x = motionMaxShift
                    while (x < width - motionMaxShift) {
                        val targetX = x + dx
                        val targetY = y + dy

                        if (targetX in 0 until width && targetY in 0 until height) {
                            val currentPixel = current[y * width + x].toInt() and 0xFF
                            val targetPixel = target[targetY * width + targetX].toInt() and 0xFF
                            sad += abs(currentPixel - targetPixel)
                            count++
                        }

                        x += motionSampleStride
                    }
                    y += motionSampleStride
                }

                val meanSad = if (count > 0) sad.toDouble() / count.toDouble() else Double.MAX_VALUE

                if (dx == 0 && dy == 0) {
                    zeroMeanSad = meanSad
                }

                if (meanSad < bestMeanSad) {
                    bestMeanSad = meanSad
                    bestDx = dx
                    bestDy = dy
                }
            }
        }

        return MotionShift(
            dx = bestDx,
            dy = bestDy,
            bestMeanSad = bestMeanSad,
            zeroMeanSad = zeroMeanSad
        )
    }

    private fun getShiftedIndex(
        x: Int,
        y: Int,
        width: Int,
        height: Int,
        dx: Int,
        dy: Int
    ): Int {
        val shiftedX = x + dx
        val shiftedY = y + dy

        if (shiftedX !in 0 until width || shiftedY !in 0 until height) {
            return -1
        }

        return shiftedY * width + shiftedX
    }

    private fun extractLuma(imageProxy: ImageProxy): ByteArray {
        val width = imageProxy.width
        val height = imageProxy.height

        val yPlane = imageProxy.planes[0]
        val buffer = yPlane.buffer

        val rowStride = yPlane.rowStride
        val pixelStride = yPlane.pixelStride

        val luma = ByteArray(width * height)

        for (row in 0 until height) {
            for (col in 0 until width) {
                val bufferIndex = row * rowStride + col * pixelStride
                val outputIndex = row * width + col
                luma[outputIndex] = buffer.get(bufferIndex)
            }
        }

        return luma
    }

    private fun createGrayscaleBitmap(
        luma: ByteArray,
        width: Int,
        height: Int
    ): Bitmap {
        val pixels = IntArray(width * height)

        for (i in luma.indices) {
            val gray = luma[i].toInt() and 0xFF
            pixels[i] = Color.rgb(gray, gray, gray)
        }

        return Bitmap.createBitmap(
            pixels,
            width,
            height,
            Bitmap.Config.ARGB_8888
        )
    }

    private fun createMotionCompensatedDifferenceBitmap(
        currentLuma: ByteArray,
        previousLuma: ByteArray,
        width: Int,
        height: Int,
        threshold: Int,
        shift: MotionShift,
        useMotionCompensation: Boolean,
        fallbackGlobalMotion: Boolean,
        imuThresholdMultiplier: Float
    ): Bitmap {
        val pixels = IntArray(width * height)

        val globalMotionMultiplier = if (fallbackGlobalMotion) {
            fallbackDifferenceThresholdMultiplier
        } else {
            1.0f
        }

        val effectiveThreshold = (threshold * maxOf(
            globalMotionMultiplier,
            imuThresholdMultiplier
        )).toInt().coerceIn(manualMinThreshold, 120)

        val dx = if (useMotionCompensation) shift.dx else 0
        val dy = if (useMotionCompensation) shift.dy else 0

        for (y in 0 until height) {
            for (x in 0 until width) {
                val currentIndex = y * width + x
                val previousIndex = getShiftedIndex(
                    x = x,
                    y = y,
                    width = width,
                    height = height,
                    dx = dx,
                    dy = dy
                )

                val displayValue = if (previousIndex < 0) {
                    128
                } else {
                    val currentPixel = currentLuma[currentIndex].toInt() and 0xFF
                    val previousPixel = previousLuma[previousIndex].toInt() and 0xFF
                    val diff = currentPixel - previousPixel

                    if (abs(diff) < effectiveThreshold) {
                        128
                    } else {
                        (128 + differenceGain * diff)
                            .toInt()
                            .coerceIn(0, 255)
                    }
                }

                pixels[currentIndex] = Color.rgb(displayValue, displayValue, displayValue)
            }
        }

        return Bitmap.createBitmap(
            pixels,
            width,
            height,
            Bitmap.Config.ARGB_8888
        )
    }

    private fun rotateBitmapIfNeeded(bitmap: Bitmap, rotationDegrees: Int): Bitmap {
        if (rotationDegrees == 0) {
            return bitmap
        }

        val matrix = Matrix()
        matrix.postRotate(rotationDegrees.toFloat())

        return Bitmap.createBitmap(
            bitmap,
            0,
            0,
            bitmap.width,
            bitmap.height,
            matrix,
            true
        )
    }
}
