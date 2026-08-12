package com.example.avtwinresponder

import android.Manifest
import android.app.Activity
import android.content.pm.PackageManager
import android.os.Bundle
import android.text.InputType
import android.view.Gravity
import android.view.ViewGroup
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import android.widget.Toast

class MainActivity : Activity() {
    companion object {
        private const val REQ_AUDIO = 1001
    }

    private lateinit var status: TextView
    private lateinit var host: EditText
    private lateinit var port: EditText
    private lateinit var startStop: Button
    private lateinit var timingTest: Button
    private lateinit var bandTest: Button
    private var responder: AcousticResponder? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        buildUi()
        ensureAudioPermission()
    }

    private fun buildUi() {
        val density = resources.displayMetrics.density
        fun dp(x: Int) = (x * density).toInt()

        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(20), dp(20), dp(20), dp(20))
        }

        val titleView = TextView(this).apply {
            text = "AV-Twin Acoustic Responder v0.4"
            textSize = 22f
            gravity = Gravity.CENTER_HORIZONTAL
        }
        root.addView(titleView, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT))

        val hintView = TextView(this).apply {
            text = "Android responder (B)\nC1 11-19 kHz -> t2 -> C2 -> hardware AudioTrack t3 -> UDP to Linux\nC2 microphone self-detection is diagnostic only\nXiaomi/HyperOS AudioTrack mode: STREAM"
            textSize = 14f
            setPadding(0, dp(10), 0, dp(10))
        }
        root.addView(hintView)

        host = EditText(this).apply {
            hint = "Linux IP, e.g. 192.168.1.100"
            setText("192.168.1.100")
            inputType = InputType.TYPE_CLASS_TEXT
        }
        root.addView(host, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT))

        port = EditText(this).apply {
            hint = "UDP port"
            setText("5005")
            inputType = InputType.TYPE_CLASS_NUMBER
        }
        root.addView(port, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT))

        startStop = Button(this).apply {
            text = "ARM / START LISTENING"
            setOnClickListener { toggleListening() }
        }
        root.addView(startStop, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT))

        timingTest = Button(this).apply {
            text = "TEST C2 HARDWARE TIMESTAMP"
            setOnClickListener { runTimingTest() }
        }
        root.addView(timingTest, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT))

        bandTest = Button(this).apply {
            text = "RUN 4-BAND SPEAKER/MIC DIAGNOSTIC"
            setOnClickListener { runBandTest() }
        }
        root.addView(bandTest, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT))

        status = TextView(this).apply {
            text = "Idle\nStep 1: TEST C2 HARDWARE TIMESTAMP.\nStep 2: RUN 4-BAND DIAGNOSTIC.\nThen ARM and let Linux send C1."
            textSize = 16f
            setTextIsSelectable(true)
            setPadding(0, dp(16), 0, dp(16))
        }
        val scroll = ScrollView(this).apply { addView(status) }
        root.addView(scroll, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f))

        setContentView(root)
    }

    private fun makeResponder(): AcousticResponder {
        return AcousticResponder(
            context = this,
            onStatus = { s ->
                runOnUiThread {
                    status.text = s
                    if (
                        s.startsWith("DONE") ||
                        s.startsWith("C2 TIMESTAMP TEST") ||
                        s.startsWith("BAND DIAGNOSTIC DONE") ||
                        s.startsWith("TIMING TEST ERROR") ||
                        s.startsWith("BAND TEST ERROR") ||
                        s.startsWith("ERROR")
                    ) {
                        setIdleButtons()
                    }
                }
            },
            onResult = { r ->
                runOnUiThread {
                    setIdleButtons()
                    status.append(
                        "\n\nreply duration=${"%.3f".format(r.replyDelayMs)} ms" +
                            "\nt2 sample=${r.t2Sample}" +
                            "\nt3 equivalent sample=${r.t3EquivalentSample}" +
                            "\nt2 score=${"%.3f".format(r.t2Score)}" +
                            "\nC2 self score=${"%.3f".format(r.c2SelfScore)} (diagnostic)"
                    )
                }
            }
        )
    }

    private fun toggleListening() {
        if (!ensurePermissionForAction()) return

        if (responder?.isRunning() == true) {
            responder?.stop()
            setIdleButtons()
            status.text = "Stopping..."
            return
        }

        val udpPort = port.text.toString().toIntOrNull() ?: 5005
        val linuxHost = host.text.toString().trim()
        responder = makeResponder()
        setRunningButtons()
        responder!!.start(linuxHost, udpPort)
    }

    private fun runTimingTest() {
        if (!ensurePermissionForAction()) return
        if (responder?.isRunning() == true) {
            Toast.makeText(this, "Stop the current audio test first", Toast.LENGTH_SHORT).show()
            return
        }
        responder = makeResponder()
        setRunningButtons()
        responder!!.startC2TimingTest()
    }

    private fun runBandTest() {
        if (!ensurePermissionForAction()) return
        if (responder?.isRunning() == true) {
            Toast.makeText(this, "Stop the current audio test first", Toast.LENGTH_SHORT).show()
            return
        }
        responder = makeResponder()
        setRunningButtons()
        responder!!.startBandDiagnostic()
    }

    private fun setRunningButtons() {
        startStop.text = "STOP"
        timingTest.isEnabled = false
        bandTest.isEnabled = false
    }

    private fun setIdleButtons() {
        startStop.text = "ARM / START LISTENING"
        timingTest.isEnabled = true
        bandTest.isEnabled = true
    }

    private fun ensurePermissionForAction(): Boolean {
        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            ensureAudioPermission()
            return false
        }
        return true
    }

    private fun ensureAudioPermission() {
        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(arrayOf(Manifest.permission.RECORD_AUDIO), REQ_AUDIO)
        }
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQ_AUDIO && grantResults.firstOrNull() != PackageManager.PERMISSION_GRANTED) {
            Toast.makeText(this, "Microphone permission is required", Toast.LENGTH_LONG).show()
        }
    }

    override fun onDestroy() {
        responder?.stop()
        super.onDestroy()
    }
}
