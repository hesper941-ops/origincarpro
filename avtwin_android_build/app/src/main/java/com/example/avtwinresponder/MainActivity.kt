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

        val title = TextView(this).apply {
            text = "AV-Twin Acoustic Responder v0.2"
            textSize = 22f
            gravity = Gravity.CENTER_HORIZONTAL
        }
        root.addView(title, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT))

        val hintView = TextView(this).apply {
            text = "Android responder (B)\nWait for C1 -> record t2 -> play C2 -> self-detect t3 -> UDP to Linux\nAudioTrack compatibility mode: STREAM"
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
            setOnClickListener { toggle() }
        }
        root.addView(startStop, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT))

        status = TextView(this).apply {
            text = "Idle"
            textSize = 16f
            setTextIsSelectable(true)
            setPadding(0, dp(16), 0, dp(16))
        }
        val scroll = ScrollView(this).apply { addView(status) }
        root.addView(scroll, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f))

        setContentView(root)
    }

    private fun toggle() {
        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            ensureAudioPermission()
            return
        }
        if (responder?.isRunning() == true) {
            responder?.stop()
            startStop.text = "ARM / START LISTENING"
            status.text = "Stopping..."
            return
        }

        val udpPort = port.text.toString().toIntOrNull() ?: 5005
        val linuxHost = host.text.toString().trim()
        responder = AcousticResponder(
            context = this,
            onStatus = { s -> runOnUiThread { status.text = s } },
            onResult = { r ->
                runOnUiThread {
                    startStop.text = "ARM AGAIN"
                    status.append("\n\nrefined t2=${r.t2Sample}, t3=${r.t3Sample}\nreply=${"%.3f".format(r.replyDelayMs)} ms\nscore=${"%.3f".format(r.t2Score)}/${"%.3f".format(r.t3Score)}")
                }
            }
        )
        startStop.text = "STOP"
        responder!!.start(linuxHost, udpPort)
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
