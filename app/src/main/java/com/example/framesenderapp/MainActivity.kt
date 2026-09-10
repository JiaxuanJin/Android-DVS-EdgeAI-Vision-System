package com.example.framesenderapp

import android.Manifest
import android.annotation.SuppressLint
import android.bluetooth.BluetoothDevice
import android.bluetooth.BluetoothManager
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.drawable.GradientDrawable
import android.os.Build
import android.os.Bundle
import android.util.Log
import android.view.Gravity
import android.view.Surface
import android.view.View
import android.widget.Button
import android.widget.FrameLayout
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.SeekBar
import android.widget.Switch
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.view.isVisible
import java.util.ArrayDeque
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

class MainActivity : AppCompatActivity() {

    // -----------------------------
    // UI components
    // -----------------------------

    private lateinit var previewView: PreviewView
    private lateinit var eventImageView: ImageView

    private lateinit var btnStartCamera: Button
    private lateinit var btnBackHome: Button
    private lateinit var btnMenu: Button

    private lateinit var tvInstruction: TextView
    private lateinit var tvStats: TextView

    private lateinit var cameraContainer: View
    private lateinit var splitLine: View

    private lateinit var menuPanel: LinearLayout
    private lateinit var btnModeCameraOnly: Button
    private lateinit var btnModeGrayscale: Button
    private lateinit var btnModeDifference: Button
    private lateinit var btnModeDvsOnly: Button
    private lateinit var btnModeComparison: Button
    private lateinit var btnModeParameter: Button

    // Threshold control panel
    private lateinit var thresholdPanel: LinearLayout
    private lateinit var tvThresholdValue: TextView
    private lateinit var seekThreshold: SeekBar
    private lateinit var switchAutoThreshold: Switch

    // Floating overlays created in code so activity_main.xml does not need changes.
    private lateinit var btnThresholdToggle: Button
    private lateinit var tvMotionPrediction: TextView
    private lateinit var btnBluetooth: Button
    private lateinit var tvBluetoothStatus: TextView

    // Bluetooth Classic sender used by the Windows COM-port display.
    private lateinit var bluetoothSender: BluetoothSenderManager

    // The threshold panel starts collapsed to avoid blocking the camera preview.
    private var thresholdPanelExpanded = false

    // -----------------------------
    // CameraX and DVS processing
    // -----------------------------

    private var cameraProvider: ProcessCameraProvider? = null
    private lateinit var cameraExecutor: ExecutorService

    // -----------------------------
    // LiteRT motion classification
    // -----------------------------

    private lateinit var motionClassifier: MotionClassifier

    // LiteRT person detector.
    // PersonDetector.kt must be in the same package and the model file must be:
    // app/src/main/assets/pedro_person_detector.tflite
    private lateinit var personDetector: PersonDetector

    private val mlExecutor: ExecutorService = Executors.newSingleThreadExecutor()
    private val isClassificationRunning = AtomicBoolean(false)

    @Volatile
    private var latestPersonDetectionResult: PersonDetectionResult? = null

    @Volatile
    private var latestPersonDetectionFrame = -1

    private val personEventWindow =
        ArrayDeque<Bitmap>()

    private val personEventWindowLock =
        Any()

    // Motion fusion and person inference are updated every 15 frames.
    private val classificationInterval = 15

    // Require the same candidate twice before changing the displayed result.
    // This reduces rapid label flicker caused by frame-to-frame event changes.
    private val requiredStableDecisions = 2
    private var pendingFusionLabel: String? = null
    private var pendingFusionCount = 0
    private var displayedFusionLabel = "Waiting..."

    private data class FusionDecision(
        val label: String,
        val confidence: Float,
        val cameraMoving: Boolean,
        val eventCount: Int
    )

    private val dvsProcessor = DvsProcessor()

    // IMU detector for camera-shake awareness.
    // CameraMotionDetector.kt must be in the same package.
    private lateinit var cameraMotionDetector: CameraMotionDetector

    private var latestCameraMotionInfo = CameraMotionInfo(
        isMoving = false,
        motionLevel = 0f,
        gyroMagnitude = 0f,
        linearAccMagnitude = 0f
    )

    private enum class DisplayMode {
        CAMERA_ONLY,
        GRAYSCALE,
        DIFFERENCE,
        DVS_ONLY,
        COMPARISON,
        PARAMETERS
    }

    private var currentMode = DisplayMode.COMPARISON

    private var currentThreshold = DEFAULT_THRESHOLD
    private var autoThresholdEnabled = false

    companion object {
        private const val REQUEST_CAMERA_PERMISSION = 100
        private const val REQUEST_BLUETOOTH_CONNECT_PERMISSION = 101
        private const val TAG = "SmartphoneDVS"
        private const val MOTION_CLASSIFIER_TAG = "MotionClassifier"
        private const val PERSON_DETECTOR_TAG = "PersonDetector"

        private const val MIN_THRESHOLD = 5
        private const val MAX_THRESHOLD = 80
        private const val DEFAULT_THRESHOLD = 15

        // Sensor-fusion thresholds.
        //
        // 1) When the IMU reports a stable camera, this threshold separates
        //    no significant motion from object motion.
        // 2) When the IMU reports camera movement, this higher threshold
        //    separates camera-only motion from camera + object motion.
        //
        // These are starting values for the current phone pipeline and can be
        // tuned later from Logcat using the reported event counts.
        private const val OBJECT_MOVE_EVENT_THRESHOLD = 3000
        private const val BOTH_MOVE_EVENT_THRESHOLD = 7000

        // The displayed confidence is a heuristic sensor-fusion score rather
        // than the raw CNN softmax probability.
        private const val MIN_FUSION_CONFIDENCE = 0.60f
        private const val MAX_FUSION_CONFIDENCE = 0.99f

        // Three consecutive smartphone DVS maps are accumulated to make the
        // moving-person silhouette more complete before inference.
        private const val PERSON_EVENT_WINDOW_SIZE = 3

        // Detection is allowed below the main 3000-event motion threshold when
        // the camera is stable, so a visible person is not skipped too early.
        private const val PERSON_DETECTION_MIN_EVENTS = 1000

        private const val PERSON_DETECTION_INTERVAL = 15
        private const val PERSON_DETECTION_MAX_AGE_FRAMES = 12
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        cameraExecutor = Executors.newSingleThreadExecutor()

        motionClassifier = MotionClassifier(applicationContext)

        Log.d(
            MOTION_CLASSIFIER_TAG,
            "Motion classifier initialized successfully."
        )

        Log.e(
            PERSON_DETECTOR_TAG,
            "Starting person detector initialization."
        )

        try {
            personDetector = PersonDetector(applicationContext)

            Log.e(
                PERSON_DETECTOR_TAG,
                "Person detector initialized successfully."
            )
        } catch (exception: Exception) {
            Log.e(
                PERSON_DETECTOR_TAG,
                "Failed to initialize the person detector.",
                exception
            )
        }

        bindViews()
        setupBluetoothSender()
        setupFloatingOverlays()
        setupThresholdControls()
        setupButtonListeners()
        setupCameraMotionDetector()
    }

    private fun bindViews() {
        previewView = findViewById(R.id.previewView)
        eventImageView = findViewById(R.id.eventImageView)

        btnStartCamera = findViewById(R.id.btnStartCamera)
        btnBackHome = findViewById(R.id.btnBackHome)
        btnMenu = findViewById(R.id.btnMenu)

        tvInstruction = findViewById(R.id.tvInstruction)
        tvStats = findViewById(R.id.tvStats)

        cameraContainer = findViewById(R.id.cameraContainer)
        splitLine = findViewById(R.id.splitLine)

        menuPanel = findViewById(R.id.menuPanel)
        btnModeCameraOnly = findViewById(R.id.btnModeCameraOnly)
        btnModeGrayscale = findViewById(R.id.btnModeGrayscale)
        btnModeDifference = findViewById(R.id.btnModeDifference)
        btnModeDvsOnly = findViewById(R.id.btnModeDvsOnly)
        btnModeComparison = findViewById(R.id.btnModeComparison)
        btnModeParameter = findViewById(R.id.btnModeParameter)

        thresholdPanel = findViewById(R.id.thresholdPanel)
        tvThresholdValue = findViewById(R.id.tvThresholdValue)
        seekThreshold = findViewById(R.id.seekThreshold)
        switchAutoThreshold = findViewById(R.id.switchAutoThreshold)
    }

    /**
     * Creates a small Controls button and a motion-prediction label as floating
     * overlays. This avoids changing activity_main.xml.
     */
    private fun setupFloatingOverlays() {
        val overlayBackground = GradientDrawable().apply {
            setColor(Color.argb(190, 32, 32, 32))
            cornerRadius = dpToPx(10).toFloat()
        }

        tvMotionPrediction = TextView(this).apply {
            text = "Motion: Waiting...\nFusion confidence: --\nEvents: --"
            setTextColor(Color.WHITE)
            textSize = 14f
            setPadding(
                dpToPx(12),
                dpToPx(8),
                dpToPx(12),
                dpToPx(8)
            )
            background = overlayBackground.constantState?.newDrawable()
            visibility = View.GONE
            elevation = dpToPx(6).toFloat()
        }

        val predictionLayoutParams = FrameLayout.LayoutParams(
            FrameLayout.LayoutParams.WRAP_CONTENT,
            FrameLayout.LayoutParams.WRAP_CONTENT,
            Gravity.TOP or Gravity.START
        ).apply {
            leftMargin = dpToPx(12)
            topMargin = dpToPx(92)
        }

        addContentView(
            tvMotionPrediction,
            predictionLayoutParams
        )

        btnThresholdToggle = Button(this).apply {
            text = "Controls"
            isAllCaps = false
            setTextColor(Color.WHITE)
            textSize = 13f
            minWidth = 0
            minHeight = 0
            setPadding(
                dpToPx(14),
                0,
                dpToPx(14),
                0
            )
            background = GradientDrawable().apply {
                setColor(Color.argb(210, 55, 55, 55))
                cornerRadius = dpToPx(18).toFloat()
            }
            visibility = View.GONE
            elevation = dpToPx(7).toFloat()

            setOnClickListener {
                thresholdPanelExpanded = !thresholdPanelExpanded
                updateThresholdPanelVisibility()
            }
        }

        val toggleLayoutParams = FrameLayout.LayoutParams(
            FrameLayout.LayoutParams.WRAP_CONTENT,
            dpToPx(42),
            Gravity.TOP or Gravity.END
        ).apply {
            rightMargin = dpToPx(12)
            topMargin = dpToPx(92)
        }

        addContentView(
            btnThresholdToggle,
            toggleLayoutParams
        )

        btnBluetooth = Button(this).apply {
            text = "Bluetooth"
            isAllCaps = false
            setTextColor(Color.WHITE)
            textSize = 13f
            minWidth = 0
            minHeight = 0
            setPadding(
                dpToPx(14),
                0,
                dpToPx(14),
                0
            )
            background = GradientDrawable().apply {
                setColor(Color.argb(220, 25, 118, 210))
                cornerRadius = dpToPx(18).toFloat()
            }
            elevation = dpToPx(8).toFloat()

            setOnClickListener {
                handleBluetoothButtonClick()
            }
        }

        val bluetoothButtonLayoutParams = FrameLayout.LayoutParams(
            FrameLayout.LayoutParams.WRAP_CONTENT,
            dpToPx(42),
            Gravity.BOTTOM or Gravity.END
        ).apply {
            rightMargin = dpToPx(16)
            bottomMargin = dpToPx(22)
        }

        addContentView(
            btnBluetooth,
            bluetoothButtonLayoutParams
        )

        tvBluetoothStatus = TextView(this).apply {
            text = "Bluetooth: Disconnected"
            setTextColor(Color.WHITE)
            textSize = 12f
            setPadding(
                dpToPx(10),
                dpToPx(6),
                dpToPx(10),
                dpToPx(6)
            )
            background = GradientDrawable().apply {
                setColor(Color.argb(190, 20, 20, 20))
                cornerRadius = dpToPx(8).toFloat()
            }
            elevation = dpToPx(7).toFloat()
        }

        val bluetoothStatusLayoutParams = FrameLayout.LayoutParams(
            FrameLayout.LayoutParams.WRAP_CONTENT,
            FrameLayout.LayoutParams.WRAP_CONTENT,
            Gravity.BOTTOM or Gravity.START
        ).apply {
            leftMargin = dpToPx(16)
            bottomMargin = dpToPx(26)
        }

        addContentView(
            tvBluetoothStatus,
            bluetoothStatusLayoutParams
        )
    }

    private fun setupBluetoothSender() {
        bluetoothSender = BluetoothSenderManager { message, connected ->
            runOnUiThread {
                if (!isFinishing && !isDestroyed) {
                    if (::tvBluetoothStatus.isInitialized) {
                        tvBluetoothStatus.text = message
                    }

                    if (::btnBluetooth.isInitialized) {
                        btnBluetooth.text = if (connected) {
                            "Disconnect"
                        } else {
                            "Bluetooth"
                        }
                    }
                }
            }
        }
    }

    private fun dpToPx(dp: Int): Int {
        return (dp * resources.displayMetrics.density).toInt()
    }

    private fun setupThresholdControls() {
        currentThreshold = DEFAULT_THRESHOLD
        autoThresholdEnabled = false

        seekThreshold.max = MAX_THRESHOLD - MIN_THRESHOLD
        seekThreshold.progress = DEFAULT_THRESHOLD - MIN_THRESHOLD
        seekThreshold.isEnabled = true
        switchAutoThreshold.isChecked = false

        updateThresholdLabel()
        dvsProcessor.setThresholdControl(auto = false, threshold = currentThreshold)

        seekThreshold.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                currentThreshold = MIN_THRESHOLD + progress
                if (!autoThresholdEnabled) {
                    dvsProcessor.setThresholdControl(auto = false, threshold = currentThreshold)
                }
                updateThresholdLabel()
            }

            override fun onStartTrackingTouch(seekBar: SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: SeekBar?) {}
        })

        switchAutoThreshold.setOnCheckedChangeListener { _, isChecked ->
            autoThresholdEnabled = isChecked
            seekThreshold.isEnabled = !isChecked

            // The threshold argument is ignored by DvsProcessor in Auto mode.
            // It is only passed so the same API can be used for both modes.
            dvsProcessor.setThresholdControl(
                auto = autoThresholdEnabled,
                threshold = currentThreshold
            )

            updateThresholdLabel()
            Log.i(TAG, "Auto threshold: $autoThresholdEnabled")
        }
    }

    private fun setupCameraMotionDetector() {
        cameraMotionDetector = CameraMotionDetector(this) { motionInfo ->
            latestCameraMotionInfo = motionInfo

            dvsProcessor.setCameraMotion(
                isMoving = motionInfo.isMoving,
                motionLevel = motionInfo.motionLevel
            )
        }
    }

    private fun setupButtonListeners() {
        btnStartCamera.setOnClickListener {
            if (hasCameraPermission()) {
                showCameraAndStart()
            } else {
                requestCameraPermission()
            }
        }

        btnBackHome.setOnClickListener {
            stopCameraAndReturnHome()
        }

        btnMenu.setOnClickListener {
            menuPanel.isVisible = !menuPanel.isVisible
            bringOverlayToFront()
            Log.i(TAG, "Menu button clicked")
        }

        btnModeCameraOnly.setOnClickListener {
            Log.i(TAG, "Camera Only Mode selected")
            setDisplayMode(DisplayMode.CAMERA_ONLY)
        }

        btnModeGrayscale.setOnClickListener {
            Log.i(TAG, "Grayscale Mode selected")
            setDisplayMode(DisplayMode.GRAYSCALE)
        }

        btnModeDifference.setOnClickListener {
            Log.i(TAG, "Difference Frame Mode selected")
            setDisplayMode(DisplayMode.DIFFERENCE)
        }

        btnModeDvsOnly.setOnClickListener {
            Log.i(TAG, "DVS Only Mode selected")
            setDisplayMode(DisplayMode.DVS_ONLY)
        }

        btnModeComparison.setOnClickListener {
            Log.i(TAG, "Comparison Mode selected")
            setDisplayMode(DisplayMode.COMPARISON)
        }

        btnModeParameter.setOnClickListener {
            Log.i(TAG, "Parameter Mode selected")
            setDisplayMode(DisplayMode.PARAMETERS)
        }
    }

    private fun handleBluetoothButtonClick() {
        if (bluetoothSender.isConnected) {
            AlertDialog.Builder(this)
                .setTitle("Bluetooth display")
                .setMessage("Disconnect the Windows display?")
                .setPositiveButton("Disconnect") { _, _ ->
                    bluetoothSender.disconnect()
                }
                .setNegativeButton("Cancel", null)
                .show()
            return
        }

        if (hasBluetoothConnectPermission()) {
            showPairedBluetoothDevices()
        } else {
            requestBluetoothConnectPermission()
        }
    }

    private fun hasBluetoothConnectPermission(): Boolean {
        return Build.VERSION.SDK_INT < Build.VERSION_CODES.S ||
                ContextCompat.checkSelfPermission(
                    this,
                    Manifest.permission.BLUETOOTH_CONNECT
                ) == PackageManager.PERMISSION_GRANTED
    }

    private fun requestBluetoothConnectPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            ActivityCompat.requestPermissions(
                this,
                arrayOf(Manifest.permission.BLUETOOTH_CONNECT),
                REQUEST_BLUETOOTH_CONNECT_PERMISSION
            )
        } else {
            showPairedBluetoothDevices()
        }
    }

    @SuppressLint("MissingPermission")
    private fun showPairedBluetoothDevices() {
        if (!hasBluetoothConnectPermission()) {
            return
        }

        val bluetoothManager = getSystemService(
            BluetoothManager::class.java
        )

        val bluetoothAdapter = bluetoothManager.adapter

        if (bluetoothAdapter == null) {
            Toast.makeText(
                this,
                "Bluetooth is not supported on this phone.",
                Toast.LENGTH_LONG
            ).show()
            return
        }

        if (!bluetoothAdapter.isEnabled) {
            Toast.makeText(
                this,
                "Enable Bluetooth in phone settings first.",
                Toast.LENGTH_LONG
            ).show()
            return
        }

        val pairedDevices: List<BluetoothDevice> =
            bluetoothAdapter.bondedDevices
                .sortedWith(
                    compareBy(
                        { it.name ?: "" },
                        { it.address }
                    )
                )

        if (pairedDevices.isEmpty()) {
            Toast.makeText(
                this,
                "No paired Bluetooth devices were found.",
                Toast.LENGTH_LONG
            ).show()
            return
        }

        val deviceLabels = pairedDevices.map { device ->
            val name = device.name ?: "Unknown device"
            "$name\n${device.address}"
        }.toTypedArray()

        AlertDialog.Builder(this)
            .setTitle("Select paired Windows PC")
            .setItems(deviceLabels) { _, selectedIndex ->
                bluetoothSender.connect(
                    pairedDevices[selectedIndex]
                )
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun hasCameraPermission(): Boolean {
        return ContextCompat.checkSelfPermission(
            this,
            Manifest.permission.CAMERA
        ) == PackageManager.PERMISSION_GRANTED
    }

    private fun requestCameraPermission() {
        ActivityCompat.requestPermissions(
            this,
            arrayOf(Manifest.permission.CAMERA),
            REQUEST_CAMERA_PERMISSION
        )
    }

    private fun setDisplayMode(mode: DisplayMode) {
        currentMode = mode
        menuPanel.visibility = View.GONE
        applyDisplayMode()
    }

    private fun showCameraAndStart() {
        dvsProcessor.reset()
        dvsProcessor.setThresholdControl(auto = autoThresholdEnabled, threshold = currentThreshold)

        thresholdPanelExpanded = false
        pendingFusionLabel = null
        pendingFusionCount = 0
        displayedFusionLabel = "Waiting..."
        latestPersonDetectionResult = null
        latestPersonDetectionFrame = -1
        clearPersonEventWindow()
        tvMotionPrediction.text =
            "Motion: Waiting...\n" +
                    "Fusion confidence: --\n" +
                    "Events: --"
        tvMotionPrediction.visibility = View.VISIBLE

        if (::cameraMotionDetector.isInitialized) {
            cameraMotionDetector.start()
        }

        cameraContainer.visibility = View.VISIBLE
        btnBackHome.visibility = View.VISIBLE
        btnMenu.visibility = View.VISIBLE
        menuPanel.visibility = View.GONE

        tvInstruction.visibility = View.GONE
        btnStartCamera.visibility = View.GONE

        applyDisplayMode()
        bringOverlayToFront()

        btnMenu.post {
            btnMenu.visibility = View.VISIBLE
            bringOverlayToFront()
            Log.i(TAG, "Menu button forced visible")
        }

        startCamera()
    }

    private fun bringOverlayToFront() {
        btnMenu.bringToFront()
        menuPanel.bringToFront()
        tvStats.bringToFront()
        btnBackHome.bringToFront()
        thresholdPanel.bringToFront()

        if (::tvMotionPrediction.isInitialized) {
            tvMotionPrediction.bringToFront()
        }

        if (::btnThresholdToggle.isInitialized) {
            btnThresholdToggle.bringToFront()
        }

        if (::btnBluetooth.isInitialized) {
            btnBluetooth.bringToFront()
        }

        if (::tvBluetoothStatus.isInitialized) {
            tvBluetoothStatus.bringToFront()
        }
    }

    private fun applyDisplayMode() {
        val previewParams = previewView.layoutParams as LinearLayout.LayoutParams
        val eventParams = eventImageView.layoutParams as LinearLayout.LayoutParams

        previewParams.height = 0
        eventParams.height = 0

        when (currentMode) {
            DisplayMode.CAMERA_ONLY -> {
                previewView.visibility = View.VISIBLE
                splitLine.visibility = View.GONE
                eventImageView.visibility = View.GONE
                tvStats.visibility = View.GONE

                previewParams.weight = 1f
                eventParams.weight = 0f
            }

            DisplayMode.GRAYSCALE -> {
                previewView.visibility = View.GONE
                splitLine.visibility = View.GONE
                eventImageView.visibility = View.VISIBLE
                tvStats.visibility = View.GONE

                previewParams.weight = 0f
                eventParams.weight = 1f
            }

            DisplayMode.DIFFERENCE -> {
                previewView.visibility = View.GONE
                splitLine.visibility = View.GONE
                eventImageView.visibility = View.VISIBLE
                tvStats.visibility = View.GONE

                previewParams.weight = 0f
                eventParams.weight = 1f
            }

            DisplayMode.DVS_ONLY -> {
                previewView.visibility = View.GONE
                splitLine.visibility = View.GONE
                eventImageView.visibility = View.VISIBLE
                tvStats.visibility = View.GONE

                previewParams.weight = 0f
                eventParams.weight = 1f
            }

            DisplayMode.COMPARISON -> {
                previewView.visibility = View.VISIBLE
                splitLine.visibility = View.VISIBLE
                eventImageView.visibility = View.VISIBLE
                tvStats.visibility = View.GONE

                previewParams.weight = 1f
                eventParams.weight = 1f
            }

            DisplayMode.PARAMETERS -> {
                previewView.visibility = View.VISIBLE
                splitLine.visibility = View.VISIBLE
                eventImageView.visibility = View.VISIBLE
                tvStats.visibility = View.VISIBLE

                previewParams.weight = 1f
                eventParams.weight = 1f
                tvStats.text = "Parameter Mode\nWaiting for DVS data..."
            }
        }

        previewView.layoutParams = previewParams
        eventImageView.layoutParams = eventParams

        updateThresholdPanelVisibility()
        updateThresholdLabel()
        bringOverlayToFront()
    }

    private fun updateThresholdPanelVisibility() {
        val controlsAvailable =
            cameraContainer.visibility == View.VISIBLE &&
                    shouldShowThresholdPanel()

        btnThresholdToggle.visibility = if (controlsAvailable) {
            View.VISIBLE
        } else {
            View.GONE
        }

        thresholdPanel.visibility = if (
            controlsAvailable &&
            thresholdPanelExpanded
        ) {
            View.VISIBLE
        } else {
            View.GONE
        }

        btnThresholdToggle.text = if (thresholdPanelExpanded) {
            "Hide controls"
        } else {
            "Controls"
        }

        bringOverlayToFront()
    }

    private fun shouldShowThresholdPanel(): Boolean {
        return currentMode != DisplayMode.CAMERA_ONLY && currentMode != DisplayMode.GRAYSCALE
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            cameraProvider = cameraProviderFuture.get()
            val targetRotation = previewView.display?.rotation ?: Surface.ROTATION_0

            val preview = Preview.Builder()
                .setTargetRotation(targetRotation)
                .build()
                .also { previewUseCase ->
                    previewUseCase.setSurfaceProvider(previewView.surfaceProvider)
                }

            val imageAnalysis = ImageAnalysis.Builder()
                .setTargetRotation(targetRotation)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()

            imageAnalysis.setAnalyzer(cameraExecutor) { imageProxy ->
                processFrameForDVSEvents(imageProxy)
            }

            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            try {
                cameraProvider?.unbindAll()
                cameraProvider?.bindToLifecycle(
                    this,
                    cameraSelector,
                    preview,
                    imageAnalysis
                )
                bringOverlayToFront()
            } catch (e: Exception) {
                Log.e(TAG, "Camera binding failed", e)
            }

        }, ContextCompat.getMainExecutor(this))
    }

    private fun processFrameForDVSEvents(imageProxy: ImageProxy) {
        try {
            val result = dvsProcessor.process(imageProxy)

            if (result != null) {
                updatePersonEventWindow(
                    result.bitmap
                )

                classifyEventMap(result)
                updateUiWithDvsResult(result)
                logDvsResult(result)
            }
        } catch (e: Exception) {
            Log.e(TAG, "Frame processing error", e)
        } finally {
            imageProxy.close()
        }
    }

    private fun updateUiWithDvsResult(result: DvsResult) {
        // The Windows display always receives the final DVS image, including
        // the most recent person box, label, and confidence when available.
        val dvsDisplayBitmap = getDvsDisplayBitmap(result)

        if (::bluetoothSender.isInitialized) {
            bluetoothSender.submitFrame(dvsDisplayBitmap)
        }

        if (result.frameCount % 2 != 0) return

        runOnUiThread {
            when (currentMode) {
                DisplayMode.CAMERA_ONLY -> {
                    // No output image update is needed.
                }

                DisplayMode.GRAYSCALE -> {
                    eventImageView.setImageBitmap(result.grayscaleBitmap)
                }

                DisplayMode.DIFFERENCE -> {
                    eventImageView.setImageBitmap(result.differenceBitmap)
                }

                DisplayMode.DVS_ONLY,
                DisplayMode.COMPARISON -> {
                    eventImageView.setImageBitmap(
                        dvsDisplayBitmap
                    )
                }

                DisplayMode.PARAMETERS -> {
                    eventImageView.setImageBitmap(
                        dvsDisplayBitmap
                    )
                    tvStats.text = formatStatsText(result)
                    tvStats.visibility = View.VISIBLE
                }
            }

            updateThresholdLabel(result.threshold)
            bringOverlayToFront()
        }
    }

    private fun updatePersonEventWindow(
        bitmap: Bitmap
    ) {
        synchronized(personEventWindowLock) {
            personEventWindow.addLast(bitmap)

            while (
                personEventWindow.size >
                PERSON_EVENT_WINDOW_SIZE
            ) {
                personEventWindow.removeFirst()
            }
        }
    }

    private fun getPersonEventWindowSnapshot():
            List<Bitmap> {
        return synchronized(
            personEventWindowLock
        ) {
            personEventWindow.toList()
        }
    }

    private fun clearPersonEventWindow() {
        synchronized(personEventWindowLock) {
            personEventWindow.clear()
        }
    }

    private fun getDvsDisplayBitmap(
        result: DvsResult
    ): Bitmap {
        val personResult =
            latestPersonDetectionResult

        val detectionAge =
            result.frameCount - latestPersonDetectionFrame

        return if (
            personResult != null &&
            detectionAge in 0..PERSON_DETECTION_MAX_AGE_FRAMES
        ) {
            personResult.annotatedBitmap
        } else {
            result.bitmap
        }
    }

    private fun updateThresholdLabel(autoAverageThreshold: Int? = null) {
        val motionText = if (latestCameraMotionInfo.isMoving) {
            "Camera Motion: DETECTED ${"%.2f".format(latestCameraMotionInfo.motionLevel)}"
        } else {
            "Camera Motion: Stable"
        }

        val thresholdText = if (autoThresholdEnabled) {
            if (autoAverageThreshold != null) {
                "Threshold: Auto avg $autoAverageThreshold"
            } else {
                "Threshold: Auto per-pixel"
            }
        } else {
            "Threshold: $currentThreshold"
        }

        tvThresholdValue.text = "$thresholdText\n$motionText"
    }

    private fun classifyEventMap(result: DvsResult) {
        // Skip the first frame and avoid running inference on every frame.
        if (
            result.frameCount <= 1 ||
            result.frameCount % classificationInterval != 0
        ) {
            return
        }

        // Snapshot IMU information so the background task uses one consistent
        // camera-motion state for this frame.
        val motionSnapshot = latestCameraMotionInfo

        // Prevent inference tasks from building up if processing is slower
        // than the incoming camera frame rate.
        if (!isClassificationRunning.compareAndSet(false, true)) {
            return
        }

        mlExecutor.execute {
            try {
                // Keep the CNN result for logging and later comparison.
                // The final phone decision is made by IMU + residual events,
                // because the CNN was trained on DVXplorer data rather than
                // smartphone frame-difference event maps.
                val prediction = motionClassifier.classify(result.bitmap)
                val mlConfidencePercent = prediction.confidence * 100.0f

                val fusionDecision = createFusionDecision(
                    eventCount = result.totalEvents,
                    cameraMoving = motionSnapshot.isMoving,
                    motionLevel = motionSnapshot.motionLevel
                )

                val stableLabel = updateStableFusionLabel(
                    fusionDecision.label
                )

                val displayConfidence = if (
                    stableLabel == fusionDecision.label
                ) {
                    fusionDecision.confidence
                } else {
                    // While waiting for the second matching decision, show a
                    // deliberately lower confidence rather than a false 100%.
                    minOf(fusionDecision.confidence, 0.69f)
                }

                val enoughResidualEvents =
                    result.totalEvents >=
                            PERSON_DETECTION_MIN_EVENTS

                val stableCameraResidualMotion =
                    !motionSnapshot.isMoving &&
                            enoughResidualEvents

                val fusedObjectMotion =
                    stableLabel == "Object Move" ||
                            stableLabel == "Both Move"

                // This avoids skipping a visible moving person merely because
                // the event count has not yet reached the main 3000 threshold.
                // Camera-only motion remains excluded to reduce background
                // false positives.
                val objectMotionForDetection =
                    enoughResidualEvents &&
                            (
                                    stableCameraResidualMotion ||
                                            fusedObjectMotion
                                    )

                val eventWindowSnapshot =
                    getPersonEventWindowSnapshot()

                val eventWindowReady =
                    eventWindowSnapshot.size >=
                            PERSON_EVENT_WINDOW_SIZE

                val shouldRunPersonDetection =
                    objectMotionForDetection &&
                            eventWindowReady &&
                            result.frameCount %
                            PERSON_DETECTION_INTERVAL == 0

                val personResult = if (
                    shouldRunPersonDetection &&
                    ::personDetector.isInitialized
                ) {
                    personDetector.detectAccumulated(
                        eventWindowSnapshot
                    )
                } else {
                    null
                }

                if (shouldRunPersonDetection) {
                    if (
                        personResult != null &&
                        personResult.detections.isNotEmpty()
                    ) {
                        latestPersonDetectionResult =
                            personResult

                        latestPersonDetectionFrame =
                            result.frameCount
                    } else {
                        latestPersonDetectionResult = null
                        latestPersonDetectionFrame = -1
                    }
                } else if (!objectMotionForDetection) {
                    latestPersonDetectionResult = null
                    latestPersonDetectionFrame = -1
                }

                val displayPersonResult =
                    latestPersonDetectionResult

                val bestPersonConfidence =
                    displayPersonResult
                        ?.detections
                        ?.maxOfOrNull {
                            it.confidence
                        }

                val personText = when {
                    !enoughResidualEvents -> {
                        "Person: Waiting for motion"
                    }

                    !eventWindowReady -> {
                        "Person: Accumulating ${
                            eventWindowSnapshot.size
                        }/$PERSON_EVENT_WINDOW_SIZE"
                    }

                    !objectMotionForDetection -> {
                        "Person: Skipped"
                    }

                    shouldRunPersonDetection &&
                            (
                                    personResult == null ||
                                            personResult.detections.isEmpty()
                                    ) -> {
                        "Person: Not detected"
                    }

                    displayPersonResult == null -> {
                        "Person: Waiting..."
                    }

                    else -> {
                        "Person: ${displayPersonResult.detections.size}, " +
                                "best ${
                                    String.format(
                                        Locale.US,
                                        "%.1f",
                                        (
                                                bestPersonConfidence
                                                    ?: 0.0f
                                                ) * 100.0f
                                    )
                                }%"
                    }
                }

                val probabilityText = prediction.probabilities.entries.joinToString(
                    separator = ", "
                ) { entry ->
                    val probability = String.format(
                        Locale.US,
                        "%.3f",
                        entry.value
                    )

                    "${entry.key}=$probability"
                }

                Log.d(
                    MOTION_CLASSIFIER_TAG,
                    "Frame=${result.frameCount}, " +
                            "Fusion=$stableLabel, " +
                            "fusionConfidence=${
                                String.format(
                                    Locale.US,
                                    "%.1f",
                                    displayConfidence * 100.0f
                                )
                            }%, " +
                            "cameraMoving=${fusionDecision.cameraMoving}, " +
                            "motionLevel=${
                                String.format(
                                    Locale.US,
                                    "%.2f",
                                    motionSnapshot.motionLevel
                                )
                            }, " +
                            "events=${fusionDecision.eventCount}, " +
                            "ML=${prediction.label}, " +
                            "mlConfidence=${
                                String.format(
                                    Locale.US,
                                    "%.1f",
                                    mlConfidencePercent
                                )
                            }%, " +
                            "personWindow=${eventWindowSnapshot.size}, " +
                            "personTrigger=$objectMotionForDetection, " +
                            "personRawMax=${
                                String.format(
                                    Locale.US,
                                    "%.4f",
                                    personResult?.rawMaxConfidence ?: 0.0f
                                )
                            }, " +
                            "personCandidates=${
                                personResult?.candidateCountBeforeNms ?: 0
                            }, " +
                            "personDetections=${
                                personResult?.detections?.size ?: 0
                            }, " +
                            "personInferenceMs=${
                                personResult?.inferenceTimeMs ?: -1
                            }, " +
                            "probabilities=[$probabilityText]"
                )

                runOnUiThread {
                    if (!isFinishing && !isDestroyed) {
                        tvMotionPrediction.text =
                            "Motion: $stableLabel\n" +
                                    "Fusion confidence: ${
                                        String.format(
                                            Locale.US,
                                            "%.1f",
                                            displayConfidence * 100.0f
                                        )
                                    }%\n" +
                                    "Events: ${fusionDecision.eventCount}\n" +
                                    personText

                        if (
                            displayPersonResult != null &&
                            (
                                    currentMode == DisplayMode.DVS_ONLY ||
                                            currentMode == DisplayMode.COMPARISON ||
                                            currentMode == DisplayMode.PARAMETERS
                                    )
                        ) {
                            eventImageView.setImageBitmap(
                                displayPersonResult.annotatedBitmap
                            )
                        }

                        tvMotionPrediction.visibility = View.VISIBLE
                        bringOverlayToFront()
                    }
                }
            } catch (exception: Exception) {
                Log.e(
                    MOTION_CLASSIFIER_TAG,
                    "Motion classification failed.",
                    exception
                )
            } finally {
                isClassificationRunning.set(false)
            }
        }
    }

    /**
     * Combines the IMU camera-motion result with the residual DVS event count.
     *
     * Stable camera + few events  -> No significant motion
     * Stable camera + many events -> Object Move
     * Moving camera + fewer residual events -> Camera Move
     * Moving camera + many residual events  -> Both Move
     */
    private fun createFusionDecision(
        eventCount: Int,
        cameraMoving: Boolean,
        motionLevel: Float
    ): FusionDecision {
        val label = when {
            !cameraMoving &&
                    eventCount < OBJECT_MOVE_EVENT_THRESHOLD -> {
                "No significant motion"
            }

            !cameraMoving -> {
                "Object Move"
            }

            eventCount < BOTH_MOVE_EVENT_THRESHOLD -> {
                "Camera Move"
            }

            else -> {
                "Both Move"
            }
        }

        val confidence = calculateFusionConfidence(
            label = label,
            eventCount = eventCount,
            motionLevel = motionLevel
        )

        return FusionDecision(
            label = label,
            confidence = confidence,
            cameraMoving = cameraMoving,
            eventCount = eventCount
        )
    }

    /**
     * Produces a conservative heuristic confidence for the fused decision.
     * It is intentionally not presented as the CNN probability.
     */
    private fun calculateFusionConfidence(
        label: String,
        eventCount: Int,
        motionLevel: Float
    ): Float {
        val objectRatio = (
                eventCount.toFloat() /
                        OBJECT_MOVE_EVENT_THRESHOLD.toFloat()
                ).coerceIn(0.0f, 2.0f)

        val bothRatio = (
                eventCount.toFloat() /
                        BOTH_MOVE_EVENT_THRESHOLD.toFloat()
                ).coerceIn(0.0f, 2.0f)

        // The current detector commonly reports motion levels around 0-3.
        // Values above 3 are simply treated as strong camera motion.
        val imuStrength = (
                motionLevel / 3.0f
                ).coerceIn(0.0f, 1.0f)

        val rawConfidence = when (label) {
            "No significant motion" -> {
                0.70f + 0.25f * (1.0f - objectRatio.coerceIn(0.0f, 1.0f))
            }

            "Object Move" -> {
                0.68f + 0.27f * ((objectRatio - 1.0f).coerceIn(0.0f, 1.0f))
            }

            "Camera Move" -> {
                val residualSeparation =
                    1.0f - bothRatio.coerceIn(0.0f, 1.0f)

                0.66f +
                        0.16f * imuStrength +
                        0.15f * residualSeparation
            }

            "Both Move" -> {
                val excessResidual =
                    (bothRatio - 1.0f).coerceIn(0.0f, 1.0f)

                0.66f +
                        0.14f * imuStrength +
                        0.17f * excessResidual
            }

            else -> {
                MIN_FUSION_CONFIDENCE
            }
        }

        return rawConfidence.coerceIn(
            MIN_FUSION_CONFIDENCE,
            MAX_FUSION_CONFIDENCE
        )
    }

    /**
     * Requires two consecutive matching candidates before changing the label.
     */
    private fun updateStableFusionLabel(
        candidateLabel: String
    ): String {
        if (candidateLabel == pendingFusionLabel) {
            pendingFusionCount += 1
        } else {
            pendingFusionLabel = candidateLabel
            pendingFusionCount = 1
        }

        if (
            pendingFusionCount >= requiredStableDecisions ||
            displayedFusionLabel == "Waiting..."
        ) {
            displayedFusionLabel = candidateLabel
        }

        return displayedFusionLabel
    }

    private fun logDvsResult(result: DvsResult) {
        if (result.frameCount % 10 != 0) return

        Log.i(
            TAG,
            "Frame: ${result.frameCount}, " +
                    "Size: ${result.width} x ${result.height}, " +
                    "Threshold: ${result.threshold}, " +
                    "MeanDiff: ${"%.2f".format(result.meanDiff)}, " +
                    "ON: ${result.onEvents}, " +
                    "OFF: ${result.offEvents}, " +
                    "Total: ${result.totalEvents}, " +
                    "CameraMoving: ${latestCameraMotionInfo.isMoving}, " +
                    "MotionLevel: ${"%.2f".format(latestCameraMotionInfo.motionLevel)}"
        )
    }

    private fun formatStatsText(result: DvsResult): String {
        val thresholdMode = if (autoThresholdEnabled) {
            "Auto per-pixel avg"
        } else {
            "Manual"
        }

        val cameraMotionText = if (latestCameraMotionInfo.isMoving) {
            "Camera Motion: DETECTED"
        } else {
            "Camera Motion: Stable"
        }

        return "Parameter Mode\n" +
                "Threshold Mode: $thresholdMode\n" +
                "Threshold: ${result.threshold}\n" +
                "$cameraMotionText\n" +
                "Motion Level: ${"%.2f".format(latestCameraMotionInfo.motionLevel)}\n" +
                "Gyro: ${"%.2f".format(latestCameraMotionInfo.gyroMagnitude)}\n" +
                "Linear Acc: ${"%.2f".format(latestCameraMotionInfo.linearAccMagnitude)}\n" +
                "MeanDiff: ${"%.2f".format(result.meanDiff)}\n" +
                "ON Events: ${result.onEvents}\n" +
                "OFF Events: ${result.offEvents}\n" +
                "Total Events: ${result.totalEvents}"
    }

    private fun stopCameraAndReturnHome() {
        if (::cameraMotionDetector.isInitialized) {
            cameraMotionDetector.stop()
        }

        cameraProvider?.unbindAll()
        cameraProvider = null

        dvsProcessor.reset()
        eventImageView.setImageBitmap(null)

        cameraContainer.visibility = View.GONE
        btnBackHome.visibility = View.GONE
        btnMenu.visibility = View.GONE
        menuPanel.visibility = View.GONE
        thresholdPanelExpanded = false
        thresholdPanel.visibility = View.GONE
        btnThresholdToggle.visibility = View.GONE
        tvMotionPrediction.visibility = View.GONE
        pendingFusionLabel = null
        pendingFusionCount = 0
        displayedFusionLabel = "Waiting..."
        tvMotionPrediction.text =
            "Motion: Waiting...\n" +
                    "Fusion confidence: --\n" +
                    "Events: --"
        tvStats.visibility = View.GONE
        tvStats.text = ""

        tvInstruction.visibility = View.VISIBLE
        btnStartCamera.visibility = View.VISIBLE
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)

        if (requestCode == REQUEST_CAMERA_PERMISSION &&
            grantResults.isNotEmpty() &&
            grantResults[0] == PackageManager.PERMISSION_GRANTED
        ) {
            showCameraAndStart()
            return
        }

        if (requestCode == REQUEST_BLUETOOTH_CONNECT_PERMISSION) {
            if (
                grantResults.isNotEmpty() &&
                grantResults[0] == PackageManager.PERMISSION_GRANTED
            ) {
                showPairedBluetoothDevices()
            } else {
                Toast.makeText(
                    this,
                    "Bluetooth permission is required for the Windows display.",
                    Toast.LENGTH_LONG
                ).show()
            }
        }
    }

    override fun onDestroy() {
        if (::cameraMotionDetector.isInitialized) {
            cameraMotionDetector.stop()
        }

        cameraProvider?.unbindAll()

        if (::bluetoothSender.isInitialized) {
            bluetoothSender.close()
        }

        if (::cameraExecutor.isInitialized) {
            cameraExecutor.shutdown()
        }

        // Queue model closure after any inference already in progress.
        if (::motionClassifier.isInitialized && !mlExecutor.isShutdown) {
            mlExecutor.execute {
                try {
                    motionClassifier.close()
                    Log.d(
                        MOTION_CLASSIFIER_TAG,
                        "Motion classifier closed successfully."
                    )
                } catch (exception: Exception) {
                    Log.e(
                        MOTION_CLASSIFIER_TAG,
                        "Failed to close the motion classifier.",
                        exception
                    )
                }
            }
        }

        if (::personDetector.isInitialized) {
            try {
                personDetector.close()

                Log.d(
                    PERSON_DETECTOR_TAG,
                    "Person detector closed successfully."
                )
            } catch (exception: Exception) {
                Log.e(
                    PERSON_DETECTOR_TAG,
                    "Failed to close the person detector.",
                    exception
                )
            }
        }

        mlExecutor.shutdown()
        super.onDestroy()
    }
}
