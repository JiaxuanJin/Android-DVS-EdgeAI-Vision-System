package com.example.framesenderapp

import android.annotation.SuppressLint
import android.bluetooth.BluetoothDevice
import android.bluetooth.BluetoothSocket
import android.graphics.Bitmap
import android.util.Log
import java.io.ByteArrayOutputStream
import java.io.DataOutputStream
import java.io.IOException
import java.util.UUID
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.atomic.AtomicReference
import kotlin.math.max
import kotlin.math.roundToInt

/**
 * Sends the latest processed DVS frame to a paired Bluetooth Classic device.
 *
 * Frame protocol:
 * 1) Four-byte big-endian JPEG length.
 * 2) JPEG image bytes.
 *
 * Only the newest pending frame is kept. This prevents an old-frame backlog
 * when Bluetooth transmission is slower than CameraX processing.
 */
class BluetoothSenderManager(
    private val statusListener: (message: String, connected: Boolean) -> Unit
) : AutoCloseable {

    companion object {
        private const val TAG = "BluetoothSender"

        // Standard Bluetooth Serial Port Profile UUID used by the Windows
        // incoming COM port.
        private val SPP_UUID: UUID = UUID.fromString(
            "00001101-0000-1000-8000-00805F9B34FB"
        )

        private const val MAX_TRANSMISSION_FPS = 15
        private const val FRAME_INTERVAL_NS =
            1_000_000_000L / MAX_TRANSMISSION_FPS

        private const val MAX_IMAGE_LONG_EDGE = 320
        private const val JPEG_QUALITY = 35
        private const val STATUS_UPDATE_INTERVAL_NS = 2_000_000_000L
    }

    private val ioExecutor: ExecutorService =
        Executors.newSingleThreadExecutor()

    private val latestFrame = AtomicReference<Bitmap?>(null)
    private val sendWorkerRunning = AtomicBoolean(false)
    private val lastFrameAcceptedNs = AtomicLong(0L)

    private val socketLock = Any()

    @Volatile
    private var bluetoothSocket: BluetoothSocket? = null

    @Volatile
    private var outputStream: DataOutputStream? = null

    @Volatile
    private var connectedDeviceName: String? = null

    private var statisticsStartNs = System.nanoTime()
    private var statisticsFrameCount = 0
    private var statisticsByteCount = 0L

    val isConnected: Boolean
        get() = bluetoothSocket?.isConnected == true && outputStream != null

    @SuppressLint("MissingPermission")
    fun connect(device: BluetoothDevice) {
        if (ioExecutor.isShutdown) {
            statusListener("Bluetooth sender is closed.", false)
            return
        }

        // Closing the current socket also aborts a blocking connect or write.
        closeSocket()
        latestFrame.set(null)

        val deviceName = device.name ?: device.address
        statusListener("Bluetooth: Connecting to $deviceName...", false)

        ioExecutor.execute {
            var newSocket: BluetoothSocket? = null

            try {
                val localSocket =
                    device.createRfcommSocketToServiceRecord(
                        SPP_UUID
                    )

                newSocket = localSocket

                synchronized(socketLock) {
                    bluetoothSocket = localSocket
                }

                // connect() is blocking and therefore runs on the I/O thread.
                localSocket.connect()

                val newOutputStream = DataOutputStream(
                    localSocket.outputStream
                )

                synchronized(socketLock) {
                    bluetoothSocket = localSocket
                    outputStream = newOutputStream
                    connectedDeviceName = deviceName
                }

                resetStatistics()

                Log.i(TAG, "Connected to $deviceName")
                statusListener(
                    "Bluetooth: Connected to $deviceName",
                    true
                )
            } catch (exception: Exception) {
                Log.e(
                    TAG,
                    "Failed to connect to $deviceName",
                    exception
                )

                try {
                    newSocket?.close()
                } catch (_: IOException) {
                    // Ignore cleanup errors after a failed connection.
                }

                synchronized(socketLock) {
                    if (bluetoothSocket === newSocket) {
                        bluetoothSocket = null
                        outputStream = null
                        connectedDeviceName = null
                    }
                }

                statusListener(
                    "Bluetooth: Connection failed",
                    false
                )
            }
        }
    }

    /**
     * Queues a processed DVS frame for transmission.
     *
     * The source bitmap is not modified or recycled. The most recent frame
     * replaces any older frame that has not started transmission yet.
     */
    fun submitFrame(bitmap: Bitmap) {
        if (!isConnected || ioExecutor.isShutdown) {
            return
        }

        val nowNs = System.nanoTime()
        val previousNs = lastFrameAcceptedNs.get()

        if (nowNs - previousNs < FRAME_INTERVAL_NS) {
            return
        }

        if (!lastFrameAcceptedNs.compareAndSet(previousNs, nowNs)) {
            return
        }

        latestFrame.set(bitmap)
        startSendWorker()
    }

    private fun startSendWorker() {
        if (!sendWorkerRunning.compareAndSet(false, true)) {
            return
        }

        ioExecutor.execute {
            try {
                while (isConnected) {
                    val frame = latestFrame.getAndSet(null) ?: break
                    sendFrame(frame)
                }
            } catch (exception: Exception) {
                Log.e(TAG, "Bluetooth frame transmission failed", exception)
                closeSocket()
                statusListener(
                    "Bluetooth: Disconnected after send error",
                    false
                )
            } finally {
                sendWorkerRunning.set(false)

                // A newer frame may have arrived just as the worker stopped.
                if (latestFrame.get() != null && isConnected) {
                    startSendWorker()
                }
            }
        }
    }

    @Throws(IOException::class)
    private fun sendFrame(sourceBitmap: Bitmap) {
        val stream = outputStream
            ?: throw IOException("Bluetooth output stream is unavailable.")

        val transmissionBitmap = createTransmissionBitmap(sourceBitmap)
        val jpegBuffer = ByteArrayOutputStream()

        try {
            val compressed = transmissionBitmap.compress(
                Bitmap.CompressFormat.JPEG,
                JPEG_QUALITY,
                jpegBuffer
            )

            if (!compressed) {
                throw IOException("JPEG compression failed.")
            }

            val jpegData = jpegBuffer.toByteArray()

            // DataOutputStream.writeInt() uses big-endian byte order, matching
            // struct.unpack(">I", header) in bluetooth_display.py.
            stream.writeInt(jpegData.size)
            stream.write(jpegData)
            stream.flush()

            updateStatistics(jpegData.size)
        } finally {
            if (transmissionBitmap !== sourceBitmap) {
                transmissionBitmap.recycle()
            }
            jpegBuffer.close()
        }
    }

    private fun createTransmissionBitmap(
        sourceBitmap: Bitmap
    ): Bitmap {
        val sourceWidth = sourceBitmap.width
        val sourceHeight = sourceBitmap.height
        val longEdge = max(sourceWidth, sourceHeight)

        if (longEdge <= MAX_IMAGE_LONG_EDGE) {
            return sourceBitmap
        }

        val scale =
            MAX_IMAGE_LONG_EDGE.toFloat() / longEdge.toFloat()

        val targetWidth = max(
            1,
            (sourceWidth * scale).roundToInt()
        )

        val targetHeight = max(
            1,
            (sourceHeight * scale).roundToInt()
        )

        return Bitmap.createScaledBitmap(
            sourceBitmap,
            targetWidth,
            targetHeight,
            true
        )
    }

    private fun resetStatistics() {
        statisticsStartNs = System.nanoTime()
        statisticsFrameCount = 0
        statisticsByteCount = 0L
    }

    private fun updateStatistics(frameByteCount: Int) {
        statisticsFrameCount += 1
        statisticsByteCount += frameByteCount.toLong()

        val nowNs = System.nanoTime()
        val elapsedNs = nowNs - statisticsStartNs

        if (elapsedNs < STATUS_UPDATE_INTERVAL_NS) {
            return
        }

        val elapsedSeconds = elapsedNs / 1_000_000_000.0
        val fps = statisticsFrameCount / elapsedSeconds
        val kilobytesPerSecond =
            statisticsByteCount / 1024.0 / elapsedSeconds
        val deviceName = connectedDeviceName ?: "device"

        statusListener(
            String.format(
                java.util.Locale.US,
                "Bluetooth: %s | %.1f FPS | %.1f KB/s",
                deviceName,
                fps,
                kilobytesPerSecond
            ),
            true
        )

        resetStatistics()
    }

    fun disconnect() {
        latestFrame.set(null)
        closeSocket()
        statusListener("Bluetooth: Disconnected", false)
    }

    private fun closeSocket() {
        val resourcesToClose = synchronized(socketLock) {
            val socket = bluetoothSocket
            val stream = outputStream

            bluetoothSocket = null
            outputStream = null
            connectedDeviceName = null

            Pair(socket, stream)
        }

        val socketToClose = resourcesToClose.first
        val streamToClose = resourcesToClose.second

        try {
            // Closing the socket first immediately aborts a blocking connect,
            // read, or write operation.
            socketToClose?.close()
        } catch (_: IOException) {
            // Ignore cleanup errors while disconnecting.
        }

        try {
            streamToClose?.close()
        } catch (_: IOException) {
            // The socket has already performed the main cleanup.
        }
    }

    override fun close() {
        latestFrame.set(null)
        closeSocket()
        ioExecutor.shutdownNow()
    }
}
