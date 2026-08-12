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
    private lateinit var testReply: Button
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
            text = "AV-Twin Acoustic Responder v0.5"
            textSize = 22f
            gravity = Gravity.CENTER_HORIZONTAL
        }
        root.addView(
            titleView,
            LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
            )
        )

        val hintView = TextView(this).apply {
            text = "PRIMARY GOAL: receive C1 -> immediately return C2\n" +
                "C1: 11-19 kHz / 200 ms\n" +
                "C2: 300 Hz-9 kHz / 200 ms\n" +
                "C2 is pre-queued before listening; AudioTrack hardware timestamp is NOT required."
            textSize = 14f
            setPadding(0, dp(10), 0, dp(10))
        }
        root.addView(hintView)

        host = EditText(this).apply {
            hint = "Linux IP, e.g. 192.168.1.100"
            setText("192.168.1.100")
            inputType = InputType.TYPE_CLASS_TEXT
        }
        root.addView(
            host,
            LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
            )
        )

        port = EditText(this).apply {
            hint = "UDP port"
            setText("5005")
            inputType = InputType.TYPE_CLASS_NUMBER
        }
        root.addView(
            port,
            LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
            )
        )

        startStop = Button(this).apply {
            text = "ARM RESPONDER / WAIT FOR C1"
            setOnClickListener { toggleListening() }
        }
        root.addView(
            startStop,
            LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
            )
        )

        testReply = Button(this).apply {
            text = "TEST: PLAY C2 REPLY NOW"
            setOnClickListener { runReplyTest() }
        }
        root.addView(
            testReply,
            LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
            )
        )

        status = TextView(this).apply {
            text = "Idle\n1) You may press TEST to confirm C2 playback.\n2) Press ARM.\n3) Linux sends C1.\n4) Tablet should detect C1 and immediately issue C2."
            textSize = 16f
            setTextIsSelectable(true)
            setPadding(0, dp(16), 0, dp(16))
        }
        val scroll = ScrollView(this).apply { addView(status) }
        root.addView(
            scroll,
            LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                0,
                1f
            )
        )

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
                        s.startsWith("C2 PLAYBACK TEST: DONE") ||
                        s.startsWith("C2 PLAYBACK TEST ERROR") ||
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
                        "\n\nResponder result:" +
                            "\nC1 t2 sample=${r.t2Sample}" +
                            "\nC1 score=${"%.3f".format(r.t2Score)}" +
                            "\nsoftware decision->play=${"%.3f".format(r.softwareDecisionToPlayUs)} us" +
                            "\nC2 reply issued=YES"
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

    private fun runReplyTest() {
        if (!ensurePermissionForAction()) return
        if (responder?.isRunning() == true) {
            Toast.makeText(this, "Stop the current responder first", Toast.LENGTH_SHORT).show()
            return
        }
        responder = makeResponder()
        setRunningButtons()
        responder!!.startReplyPlaybackTest()
    }

    private fun setRunningButtons() {
        startStop.text = "STOP"
        testReply.isEnabled = false
    }

    private fun setIdleButtons() {
        startStop.text = "ARM RESPONDER / WAIT FOR C1"
        testReply.isEnabled = true
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

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
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
