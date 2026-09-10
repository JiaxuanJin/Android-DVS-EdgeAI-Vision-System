package com.example.framesenderapp

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import org.tensorflow.lite.Interpreter
import java.io.Closeable
import java.nio.ByteBuffer
import java.nio.ByteOrder

data class MotionPrediction(
    val label: String,
    val confidence: Float,
    val probabilities: Map<String, Float>
)

class MotionClassifier(
    context: Context
) : Closeable {

    companion object {
        private const val MODEL_FILE = "saed_motion_classifier.tflite"
        private const val LABEL_FILE = "motion_labels.txt"

        private const val INPUT_WIDTH = 128
        private const val INPUT_HEIGHT = 96
        private const val BYTES_PER_FLOAT = 4

        private const val DARK_MAX = 30
        private const val BRIGHT_MIN = 200
        private const val RED_OTHER_CHANNEL_MAX = 100
    }

    private val labels: List<String> = loadLabels(context)

    private val interpreter: Interpreter = Interpreter(
        loadModelBuffer(context),
        Interpreter.Options().apply {
            setNumThreads(4)
        }
    )

    init {
        require(labels.size == 3) {
            "The model requires exactly three labels."
        }
    }

    /**
     * Classifies one smartphone-generated DVS event map.
     *
     * Model input:
     * [1, 96, 128, 1] float32
     *
     * The SAED training maps used:
     * - inactive background = 127
     * - ON event = 255
     * - OFF event = 0
     *
     * The phone display currently uses:
     * - black background
     * - red event
     * - white event
     *
     * This class converts the phone colours back to the training format.
     * Pixel values are not divided by 255 because the model already contains
     * a Rescaling(1/255) layer.
     */
    fun classify(bitmap: Bitmap): MotionPrediction {
        val resizedBitmap = Bitmap.createScaledBitmap(
            bitmap,
            INPUT_WIDTH,
            INPUT_HEIGHT,
            false
        )

        val inputBuffer = createInputBuffer(resizedBitmap)

        val output = Array(1) {
            FloatArray(labels.size)
        }

        interpreter.run(
            inputBuffer,
            output
        )

        val probabilities = output[0]

        val predictedIndex = probabilities.indices.maxByOrNull { index ->
            probabilities[index]
        } ?: 0

        val probabilityMap = labels.indices.associate { index ->
            labels[index] to probabilities[index]
        }

        if (resizedBitmap !== bitmap) {
            resizedBitmap.recycle()
        }

        return MotionPrediction(
            label = labels[predictedIndex],
            confidence = probabilities[predictedIndex],
            probabilities = probabilityMap
        )
    }

    private fun createInputBuffer(bitmap: Bitmap): ByteBuffer {
        val inputBuffer = ByteBuffer.allocateDirect(
            INPUT_WIDTH *
                    INPUT_HEIGHT *
                    BYTES_PER_FLOAT
        ).apply {
            order(ByteOrder.nativeOrder())
        }

        val pixels = IntArray(
            INPUT_WIDTH * INPUT_HEIGHT
        )

        bitmap.getPixels(
            pixels,
            0,
            INPUT_WIDTH,
            0,
            0,
            INPUT_WIDTH,
            INPUT_HEIGHT
        )

        for (pixel in pixels) {
            val red = Color.red(pixel)
            val green = Color.green(pixel)
            val blue = Color.blue(pixel)

            val modelPixelValue = when {
                // Phone background: black -> SAED inactive background: grey.
                red <= DARK_MAX &&
                        green <= DARK_MAX &&
                        blue <= DARK_MAX -> 127.0f

                // Phone ON event: red -> SAED ON event: white.
                red >= BRIGHT_MIN &&
                        green <= RED_OTHER_CHANNEL_MAX &&
                        blue <= RED_OTHER_CHANNEL_MAX -> 255.0f

                // Phone OFF event: white -> SAED OFF event: black.
                red >= BRIGHT_MIN &&
                        green >= BRIGHT_MIN &&
                        blue >= BRIGHT_MIN -> 0.0f

                // Mid-grey pixels are already compatible with the SAED
                // inactive background. Other unexpected colours are also
                // treated as inactive instead of creating false events.
                else -> 127.0f
            }

            inputBuffer.putFloat(modelPixelValue)
        }

        inputBuffer.rewind()

        return inputBuffer
    }

    private fun loadLabels(context: Context): List<String> {
        return context.assets
            .open(LABEL_FILE)
            .bufferedReader()
            .useLines { lines ->
                lines
                    .map { line -> line.trim() }
                    .filter { line -> line.isNotEmpty() }
                    .toList()
            }
    }

    private fun loadModelBuffer(context: Context): ByteBuffer {
        val modelBytes = context.assets
            .open(MODEL_FILE)
            .use { inputStream ->
                inputStream.readBytes()
            }

        return ByteBuffer.allocateDirect(
            modelBytes.size
        ).apply {
            order(ByteOrder.nativeOrder())
            put(modelBytes)
            rewind()
        }
    }

    override fun close() {
        interpreter.close()
    }
}
