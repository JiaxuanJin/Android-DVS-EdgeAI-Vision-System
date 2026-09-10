package com.example.framesenderapp

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

data class PersonDetection(
    val confidence: Float,
    val left: Float,
    val top: Float,
    val right: Float,
    val bottom: Float
)

data class PersonDetectionResult(
    val annotatedBitmap: Bitmap,
    val detections: List<PersonDetection>,
    val inferenceTimeMs: Long,
    val rawMaxConfidence: Float,
    val candidateCountBeforeNms: Int
)

class PersonDetector(
    context: Context
) : AutoCloseable {

    companion object {
        private const val TAG = "PersonDetector"
        private const val MODEL_NAME = "pedro_person_detector.tflite"

        private const val INPUT_SIZE = 320
        private const val INPUT_CHANNELS = 3
        private const val OUTPUT_CHANNELS = 5
        private const val OUTPUT_CANDIDATES = 2100

        // Phone deployment thresholds.
        // The previous 0.08 diagnostic threshold produced many false boxes.
        private const val CONFIDENCE_THRESHOLD = 0.35f
        private const val IOU_THRESHOLD = 0.40f
        private const val MAX_DETECTIONS = 3

        // Conservative geometric filters for a vertically oriented person.
        private const val MIN_BOX_WIDTH_RATIO = 0.04f
        private const val MIN_BOX_HEIGHT_RATIO = 0.10f
        private const val MAX_BOX_AREA_RATIO = 0.75f
        private const val MIN_PERSON_ASPECT_RATIO = 0.80f
        private const val MAX_PERSON_ASPECT_RATIO = 6.00f
    }

    private val interpreter: Interpreter

    private val inputBuffer: ByteBuffer = ByteBuffer.allocateDirect(
        1 * INPUT_CHANNELS * INPUT_SIZE * INPUT_SIZE * 4
    ).order(ByteOrder.nativeOrder())

    private val outputBuffer = Array(1) {
        Array(OUTPUT_CHANNELS) {
            FloatArray(OUTPUT_CANDIDATES)
        }
    }

    init {
        val modelBuffer = loadModelFile(
            context = context,
            modelName = MODEL_NAME
        )

        val options = Interpreter.Options().apply {
            // Two threads reduce contention with CameraX and DVS processing.
            setNumThreads(2)
        }

        interpreter = Interpreter(
            modelBuffer,
            options
        )

        validateTensorShapes()
        logTensorInformation()
    }

    @Synchronized
    fun detectAccumulated(
        sourceBitmaps: List<Bitmap>
    ): PersonDetectionResult {
        require(sourceBitmaps.isNotEmpty()) {
            "At least one event bitmap is required."
        }

        val accumulatedBitmap =
            accumulateRecentEventMaps(sourceBitmaps)

        return detect(accumulatedBitmap)
    }

    private fun accumulateRecentEventMaps(
        sourceBitmaps: List<Bitmap>
    ): Bitmap {
        val newestBitmap = sourceBitmaps.last()
        val width = newestBitmap.width
        val height = newestBitmap.height

        require(
            sourceBitmaps.all {
                it.width == width &&
                        it.height == height
            }
        ) {
            "All accumulated event maps must have the same size."
        }

        val accumulatedPixels = IntArray(
            width * height
        ) {
            Color.BLACK
        }

        val temporaryPixels = IntArray(
            width * height
        )

        // Work from newest to oldest. The newest non-background event wins,
        // while older maps fill gaps in the current silhouette.
        for (bitmap in sourceBitmaps.asReversed()) {
            bitmap.getPixels(
                temporaryPixels,
                0,
                width,
                0,
                0,
                width,
                height
            )

            for (index in accumulatedPixels.indices) {
                if (
                    accumulatedPixels[index] == Color.BLACK &&
                    isPhoneEventPixel(temporaryPixels[index])
                ) {
                    accumulatedPixels[index] =
                        temporaryPixels[index]
                }
            }
        }

        return Bitmap.createBitmap(
            accumulatedPixels,
            width,
            height,
            Bitmap.Config.ARGB_8888
        )
    }

    private fun isPhoneEventPixel(
        colour: Int
    ): Boolean {
        val red = Color.red(colour)
        val green = Color.green(colour)
        val blue = Color.blue(colour)

        val redEvent =
            red > 160 &&
                    green < 170 &&
                    blue < 170

        val whiteEvent =
            red > 190 &&
                    green > 190 &&
                    blue > 190

        return redEvent || whiteEvent
    }

    @Synchronized
    fun detect(sourceBitmap: Bitmap): PersonDetectionResult {
        require(
            sourceBitmap.width > 0 &&
                    sourceBitmap.height > 0
        ) {
            "The source bitmap has an invalid size."
        }

        val preprocessing = createModelInput(
            sourceBitmap
        )

        inputBuffer.rewind()
        writeNchwInput(
            bitmap = preprocessing.modelBitmap,
            destination = inputBuffer
        )
        inputBuffer.rewind()

        clearOutputBuffer()

        val startTimeNs = System.nanoTime()

        interpreter.run(
            inputBuffer,
            outputBuffer
        )

        val inferenceTimeMs =
            (System.nanoTime() - startTimeNs) / 1_000_000L

        val rawMaxConfidence =
            outputBuffer[0][4].maxOrNull() ?: 0.0f

        val candidates = decodeOutput(
            sourceWidth = sourceBitmap.width,
            sourceHeight = sourceBitmap.height,
            scale = preprocessing.scale,
            padX = preprocessing.padX,
            padY = preprocessing.padY
        )

        val detections = applyNonMaximumSuppression(
            candidates
        )

        val annotatedBitmap = drawDetections(
            sourceBitmap = sourceBitmap,
            detections = detections
        )

        Log.d(
            TAG,
            "Inference completed: " +
                    "rawMax=${
                        String.format(
                            java.util.Locale.US,
                            "%.4f",
                            rawMaxConfidence
                        )
                    }, " +
                    "candidatesBeforeNms=${candidates.size}, " +
                    "detections=${detections.size}, " +
                    "time=${inferenceTimeMs}ms"
        )

        return PersonDetectionResult(
            annotatedBitmap = annotatedBitmap,
            detections = detections,
            inferenceTimeMs = inferenceTimeMs,
            rawMaxConfidence = rawMaxConfidence,
            candidateCountBeforeNms = candidates.size
        )
    }

    private data class PreprocessingResult(
        val modelBitmap: Bitmap,
        val scale: Float,
        val padX: Float,
        val padY: Float
    )

    private fun createModelInput(
        sourceBitmap: Bitmap
    ): PreprocessingResult {
        val mappedBitmap = convertPhoneEventMap(
            sourceBitmap
        )

        val sourceWidth = mappedBitmap.width
        val sourceHeight = mappedBitmap.height

        val scale = min(
            INPUT_SIZE.toFloat() / sourceWidth.toFloat(),
            INPUT_SIZE.toFloat() / sourceHeight.toFloat()
        )

        val resizedWidth = max(
            1,
            (sourceWidth * scale).toInt()
        )

        val resizedHeight = max(
            1,
            (sourceHeight * scale).toInt()
        )

        val padX = (
                INPUT_SIZE - resizedWidth
                ) / 2.0f

        val padY = (
                INPUT_SIZE - resizedHeight
                ) / 2.0f

        val modelBitmap = Bitmap.createBitmap(
            INPUT_SIZE,
            INPUT_SIZE,
            Bitmap.Config.ARGB_8888
        )

        val canvas = Canvas(modelBitmap)
        canvas.drawColor(
            Color.rgb(
                127,
                127,
                127
            )
        )

        val destination = RectF(
            padX,
            padY,
            padX + resizedWidth,
            padY + resizedHeight
        )

        val paint = Paint().apply {
            isAntiAlias = false
            isFilterBitmap = false
        }

        canvas.drawBitmap(
            mappedBitmap,
            null,
            destination,
            paint
        )

        return PreprocessingResult(
            modelBitmap = modelBitmap,
            scale = scale,
            padX = padX,
            padY = padY
        )
    }

    /**
     * Converts the Android DVS display colours into the representation used
     * during PEDRo training:
     *
     * Phone black background -> training grey background
     * Phone red event        -> training white positive event
     * Phone white event      -> training black negative event
     */
    private fun convertPhoneEventMap(
        sourceBitmap: Bitmap
    ): Bitmap {
        val width = sourceBitmap.width
        val height = sourceBitmap.height

        val sourcePixels = IntArray(
            width * height
        )

        sourceBitmap.getPixels(
            sourcePixels,
            0,
            width,
            0,
            0,
            width,
            height
        )

        val mappedPixels = IntArray(
            sourcePixels.size
        )

        for (index in sourcePixels.indices) {
            val colour = sourcePixels[index]

            val red = Color.red(colour)
            val green = Color.green(colour)
            val blue = Color.blue(colour)

            val grayscale = when {
                red < 40 &&
                        green < 40 &&
                        blue < 40 -> {
                    127
                }

                red > 180 &&
                        green < 140 &&
                        blue < 140 -> {
                    255
                }

                red > 200 &&
                        green > 200 &&
                        blue > 200 -> {
                    0
                }

                else -> {
                    127
                }
            }

            mappedPixels[index] = Color.rgb(
                grayscale,
                grayscale,
                grayscale
            )
        }

        return Bitmap.createBitmap(
            mappedPixels,
            width,
            height,
            Bitmap.Config.ARGB_8888
        )
    }

    private fun writeNchwInput(
        bitmap: Bitmap,
        destination: ByteBuffer
    ) {
        val pixels = IntArray(
            INPUT_SIZE * INPUT_SIZE
        )

        bitmap.getPixels(
            pixels,
            0,
            INPUT_SIZE,
            0,
            0,
            INPUT_SIZE,
            INPUT_SIZE
        )

        // The model input is [1, 3, 320, 320], so data must be written in
        // channel-first order rather than the more common NHWC order.
        for (channel in 0 until INPUT_CHANNELS) {
            for (pixel in pixels) {
                val value = Color.red(pixel) / 255.0f
                destination.putFloat(value)
            }
        }
    }

    private fun clearOutputBuffer() {
        for (channel in 0 until OUTPUT_CHANNELS) {
            outputBuffer[0][channel].fill(0.0f)
        }
    }

    private fun decodeOutput(
        sourceWidth: Int,
        sourceHeight: Int,
        scale: Float,
        padX: Float,
        padY: Float
    ): List<PersonDetection> {
        val detections = mutableListOf<PersonDetection>()

        for (candidateIndex in 0 until OUTPUT_CANDIDATES) {
            var centerX =
                outputBuffer[0][0][candidateIndex]

            var centerY =
                outputBuffer[0][1][candidateIndex]

            var boxWidth =
                outputBuffer[0][2][candidateIndex]

            var boxHeight =
                outputBuffer[0][3][candidateIndex]

            val confidence =
                outputBuffer[0][4][candidateIndex]

            if (
                !confidence.isFinite() ||
                confidence < CONFIDENCE_THRESHOLD
            ) {
                continue
            }

            // Most Ultralytics LiteRT exports return pixel coordinates.
            // This fallback also supports normalized coordinates.
            val largestCoordinate = max(
                max(abs(centerX), abs(centerY)),
                max(abs(boxWidth), abs(boxHeight))
            )

            if (largestCoordinate <= 2.0f) {
                centerX *= INPUT_SIZE
                centerY *= INPUT_SIZE
                boxWidth *= INPUT_SIZE
                boxHeight *= INPUT_SIZE
            }

            val modelLeft =
                centerX - boxWidth / 2.0f

            val modelTop =
                centerY - boxHeight / 2.0f

            val modelRight =
                centerX + boxWidth / 2.0f

            val modelBottom =
                centerY + boxHeight / 2.0f

            val sourceLeft = (
                    (modelLeft - padX) / scale
                    ).coerceIn(
                    0.0f,
                    sourceWidth.toFloat()
                )

            val sourceTop = (
                    (modelTop - padY) / scale
                    ).coerceIn(
                    0.0f,
                    sourceHeight.toFloat()
                )

            val sourceRight = (
                    (modelRight - padX) / scale
                    ).coerceIn(
                    0.0f,
                    sourceWidth.toFloat()
                )

            val sourceBottom = (
                    (modelBottom - padY) / scale
                    ).coerceIn(
                    0.0f,
                    sourceHeight.toFloat()
                )

            if (
                sourceRight <= sourceLeft ||
                sourceBottom <= sourceTop
            ) {
                continue
            }

            val sourceBoxWidth =
                sourceRight - sourceLeft

            val sourceBoxHeight =
                sourceBottom - sourceTop

            val widthRatio =
                sourceBoxWidth / sourceWidth.toFloat()

            val heightRatio =
                sourceBoxHeight / sourceHeight.toFloat()

            val areaRatio = (
                    sourceBoxWidth * sourceBoxHeight
                    ) / (
                    sourceWidth.toFloat() *
                            sourceHeight.toFloat()
                    )

            val aspectRatio =
                sourceBoxHeight / sourceBoxWidth

            if (
                widthRatio < MIN_BOX_WIDTH_RATIO ||
                heightRatio < MIN_BOX_HEIGHT_RATIO ||
                areaRatio > MAX_BOX_AREA_RATIO ||
                aspectRatio < MIN_PERSON_ASPECT_RATIO ||
                aspectRatio > MAX_PERSON_ASPECT_RATIO
            ) {
                continue
            }

            detections += PersonDetection(
                confidence = confidence,
                left = sourceLeft,
                top = sourceTop,
                right = sourceRight,
                bottom = sourceBottom
            )
        }

        return detections
    }

    private fun applyNonMaximumSuppression(
        candidates: List<PersonDetection>
    ): List<PersonDetection> {
        val sorted = candidates.sortedByDescending {
            it.confidence
        }.toMutableList()

        val selected = mutableListOf<PersonDetection>()

        while (
            sorted.isNotEmpty() &&
            selected.size < MAX_DETECTIONS
        ) {
            val best = sorted.removeAt(0)
            selected += best

            val iterator = sorted.iterator()

            while (iterator.hasNext()) {
                val candidate = iterator.next()

                if (
                    calculateIou(
                        first = best,
                        second = candidate
                    ) > IOU_THRESHOLD
                ) {
                    iterator.remove()
                }
            }
        }

        return selected
    }

    private fun calculateIou(
        first: PersonDetection,
        second: PersonDetection
    ): Float {
        val intersectionLeft = max(
            first.left,
            second.left
        )

        val intersectionTop = max(
            first.top,
            second.top
        )

        val intersectionRight = min(
            first.right,
            second.right
        )

        val intersectionBottom = min(
            first.bottom,
            second.bottom
        )

        val intersectionWidth = max(
            0.0f,
            intersectionRight - intersectionLeft
        )

        val intersectionHeight = max(
            0.0f,
            intersectionBottom - intersectionTop
        )

        val intersectionArea =
            intersectionWidth * intersectionHeight

        val firstArea =
            (first.right - first.left) *
                    (first.bottom - first.top)

        val secondArea =
            (second.right - second.left) *
                    (second.bottom - second.top)

        val unionArea =
            firstArea + secondArea - intersectionArea

        return if (unionArea > 0.0f) {
            intersectionArea / unionArea
        } else {
            0.0f
        }
    }

    private fun drawDetections(
        sourceBitmap: Bitmap,
        detections: List<PersonDetection>
    ): Bitmap {
        val annotated = sourceBitmap.copy(
            Bitmap.Config.ARGB_8888,
            true
        )

        val canvas = Canvas(annotated)

        val strokeWidth = max(
            2.0f,
            sourceBitmap.width * 0.005f
        )

        val textSize = max(
            14.0f,
            sourceBitmap.width * 0.025f
        )

        val boxPaint = Paint().apply {
            color = Color.GREEN
            style = Paint.Style.STROKE
            this.strokeWidth = strokeWidth
            isAntiAlias = true
        }

        val textPaint = Paint().apply {
            color = Color.WHITE
            style = Paint.Style.FILL
            this.textSize = textSize
            isAntiAlias = true
        }

        val textBackgroundPaint = Paint().apply {
            color = Color.argb(
                210,
                0,
                110,
                0
            )
            style = Paint.Style.FILL
        }

        for (detection in detections) {
            val box = RectF(
                detection.left,
                detection.top,
                detection.right,
                detection.bottom
            )

            canvas.drawRect(
                box,
                boxPaint
            )

            val label = String.format(
                "Person %.1f%%",
                detection.confidence * 100.0f
            )

            val textWidth =
                textPaint.measureText(label)

            val textHeight =
                textPaint.fontMetrics.run {
                    bottom - top
                }

            val labelTop = max(
                0.0f,
                detection.top - textHeight
            )

            canvas.drawRect(
                detection.left,
                labelTop,
                min(
                    sourceBitmap.width.toFloat(),
                    detection.left + textWidth + 12.0f
                ),
                labelTop + textHeight,
                textBackgroundPaint
            )

            canvas.drawText(
                label,
                detection.left + 6.0f,
                labelTop - textPaint.fontMetrics.top,
                textPaint
            )
        }

        return annotated
    }

    private fun validateTensorShapes() {
        val inputShape =
            interpreter.getInputTensor(0).shape()

        val outputShape =
            interpreter.getOutputTensor(0).shape()

        require(
            inputShape.contentEquals(
                intArrayOf(
                    1,
                    INPUT_CHANNELS,
                    INPUT_SIZE,
                    INPUT_SIZE
                )
            )
        ) {
            "Unexpected input tensor shape: " +
                    inputShape.contentToString()
        }

        require(
            outputShape.contentEquals(
                intArrayOf(
                    1,
                    OUTPUT_CHANNELS,
                    OUTPUT_CANDIDATES
                )
            )
        ) {
            "Unexpected output tensor shape: " +
                    outputShape.contentToString()
        }
    }

    private fun logTensorInformation() {
        Log.d(
            TAG,
            "Input tensor count: ${interpreter.inputTensorCount}"
        )

        for (
        index in 0 until interpreter.inputTensorCount
        ) {
            val tensor =
                interpreter.getInputTensor(index)

            Log.d(
                TAG,
                "Input[$index] " +
                        "shape=${tensor.shape().contentToString()}, " +
                        "type=${tensor.dataType()}, " +
                        "bytes=${tensor.numBytes()}"
            )
        }

        Log.d(
            TAG,
            "Output tensor count: ${interpreter.outputTensorCount}"
        )

        for (
        index in 0 until interpreter.outputTensorCount
        ) {
            val tensor =
                interpreter.getOutputTensor(index)

            Log.d(
                TAG,
                "Output[$index] " +
                        "shape=${tensor.shape().contentToString()}, " +
                        "type=${tensor.dataType()}, " +
                        "bytes=${tensor.numBytes()}"
            )
        }
    }

    private fun loadModelFile(
        context: Context,
        modelName: String
    ): MappedByteBuffer {
        val fileDescriptor =
            context.assets.openFd(modelName)

        FileInputStream(
            fileDescriptor.fileDescriptor
        ).use { inputStream ->
            val fileChannel = inputStream.channel

            return fileChannel.map(
                FileChannel.MapMode.READ_ONLY,
                fileDescriptor.startOffset,
                fileDescriptor.declaredLength
            )
        }
    }

    override fun close() {
        interpreter.close()
    }
}
