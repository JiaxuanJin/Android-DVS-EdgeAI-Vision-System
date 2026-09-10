package com.example.framesenderapp

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import kotlin.math.sqrt

data class CameraMotionInfo(
    val isMoving: Boolean,
    val motionLevel: Float,
    val gyroMagnitude: Float,
    val linearAccMagnitude: Float
)

class CameraMotionDetector(
    context: Context,
    private val onMotionChanged: (CameraMotionInfo) -> Unit
) : SensorEventListener {

    private val sensorManager =
        context.getSystemService(Context.SENSOR_SERVICE) as SensorManager

    private val gyroscope: Sensor? =
        sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)

    private val linearAcceleration: Sensor? =
        sensorManager.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION)

    private var gyroMagnitude = 0f
    private var linearAccMagnitude = 0f

    // Smooth motion value, avoid rapid flickering
    private var smoothedMotionLevel = 0f

    // Adjust these values if detection is too sensitive or too weak
    private val gyroThreshold = 0.18f
    private val linearAccThreshold = 0.45f

    private val smoothingAlpha = 0.20f

    fun start() {
        gyroscope?.let {
            sensorManager.registerListener(
                this,
                it,
                SensorManager.SENSOR_DELAY_GAME
            )
        }

        linearAcceleration?.let {
            sensorManager.registerListener(
                this,
                it,
                SensorManager.SENSOR_DELAY_GAME
            )
        }
    }

    fun stop() {
        sensorManager.unregisterListener(this)

        gyroMagnitude = 0f
        linearAccMagnitude = 0f
        smoothedMotionLevel = 0f

        onMotionChanged(
            CameraMotionInfo(
                isMoving = false,
                motionLevel = 0f,
                gyroMagnitude = 0f,
                linearAccMagnitude = 0f
            )
        )
    }

    override fun onSensorChanged(event: SensorEvent) {
        when (event.sensor.type) {
            Sensor.TYPE_GYROSCOPE -> {
                val x = event.values[0]
                val y = event.values[1]
                val z = event.values[2]

                gyroMagnitude = sqrt(x * x + y * y + z * z)
            }

            Sensor.TYPE_LINEAR_ACCELERATION -> {
                val x = event.values[0]
                val y = event.values[1]
                val z = event.values[2]

                linearAccMagnitude = sqrt(x * x + y * y + z * z)
            }
        }

        val gyroScore = (gyroMagnitude / gyroThreshold).coerceIn(0f, 3f)
        val accScore = (linearAccMagnitude / linearAccThreshold).coerceIn(0f, 3f)

        val rawMotionLevel = maxOf(gyroScore, accScore)

        smoothedMotionLevel =
            smoothingAlpha * rawMotionLevel +
                    (1f - smoothingAlpha) * smoothedMotionLevel

        val isMoving = smoothedMotionLevel > 1.0f

        onMotionChanged(
            CameraMotionInfo(
                isMoving = isMoving,
                motionLevel = smoothedMotionLevel,
                gyroMagnitude = gyroMagnitude,
                linearAccMagnitude = linearAccMagnitude
            )
        )
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {
        // Not needed for this project
    }
}