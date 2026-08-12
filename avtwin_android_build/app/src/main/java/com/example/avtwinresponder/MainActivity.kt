package com.example.avtwinresponder

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
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
        private const val REQ_C1_FILE = 2001
        private const val REQ_C2_FILE = 2002
        private const val PREFS = "avtwin_probe_prefs"
        private const val PREF_C1_URI = "c1_uri"
        private const val PREF_C2_URI = "c2_uri"
    }

    private lateinit var status: TextView
    private lateinit var host: EditText
    private lateinit var port: EditText
    private lateinit var startStop: Button
    private lateinit var testReply: Button
    private lateinit var c1Info: TextView
    private lateinit var c2Info: TextView
    private lateinit var selectC1: Button
    private lateinit var selectC2: Button
    private lateinit var defaultC1: Button
    private lateinit var defaultC2: Button

    private var responder: AcousticResponder? = null
    private var c1Signal: ProbeSignal = ProbeDefaults.c1()
    private var c2Signal: ProbeSignal = ProbeDefaults.c2()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        buildUi()
        ensureAudioPermission()
        restoreSavedProbeFiles()
    }

    private fun buildUi() {
        val density = resources.displayMetrics.density
        fun dp(x: Int) = (x * density).toInt()

        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(20), dp(16), dp(20), dp(16))
        }

        val titleView = TextView(this).apply {
            text = "AV-Twin Acoustic Responder v0.6"
            textSize = 22f
            gravity = Gravity.CENTER_HORIZONTAL
        }
        root.addView(titleView)

        val hintView = TextView(this).apply {
            text = "PRIMARY GOAL: receive known C1 -> immediately return known C2\n" +
                "Choose the SAME C1 file that Linux will transmit. Choose the SAME C2 file that Linux will later detect.\n" +
                "Supported probe files: RIFF/WAV PCM or 32-bit float; automatically converted to 48 kHz mono."
            textSize = 14f
            setPadding(0, dp(8), 0, dp(8))
        }
        root.addView(hintView)

        c1Info = TextView(this).apply {
            textSize = 14f
            setTextIsSelectable(true)
        }
        root.addView(c1Info)

        val c1Row = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        selectC1 = Button(this).apply {
            text = "SELECT C1 WAV"
            setOnClickListener { chooseProbeFile(REQ_C1_FILE) }
        }
        defaultC1 = Button(this).apply {
            text = "DEFAULT C1"
            setOnClickListener {
                c1Signal = ProbeDefaults.c1()
                getSharedPreferences(PREFS, MODE_PRIVATE).edit().remove(PREF_C1_URI).apply()
                updateProbeInfo()
            }
        }
        c1Row.addView(selectC1, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
        c1Row.addView(defaultC1, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
        root.addView(c1Row)

        c2Info = TextView(this).apply {
            textSize = 14f
            setTextIsSelectable(true)
            setPadding(0, dp(4), 0, 0)
        }
        root.addView(c2Info)

        val c2Row = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        selectC2 = Button(this).apply {
            text = "SELECT C2 WAV"
            setOnClickListener { chooseProbeFile(REQ_C2_FILE) }
        }
        defaultC2 = Button(this).apply {
            text = "DEFAULT C2"
            setOnClickListener {
                c2Signal = ProbeDefaults.c2()
                getSharedPreferences(PREFS, MODE_PRIVATE).edit().remove(PREF_C2_URI).apply()
                updateProbeInfo()
            }
        }
        c2Row.addView(selectC2, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
        c2Row.addView(defaultC2, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
        root.addView(c2Row)

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
            text = "ARM RESPONDER / WAIT FOR C1"
            setOnClickListener { toggleListening() }
        }
        root.addView(startStop, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT))

        testReply = Button(this).apply {
            text = "TEST: PLAY SELECTED C2 NOW"
            setOnClickListener { runReplyTest() }
        }
        root.addView(testReply, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT))

        status = TextView(this).apply {
            text = "Idle\n1) Select C1/C2 WAV, or keep defaults.\n2) TEST C2 if desired.\n3) ARM.\n4) Linux transmits the exact same C1 waveform.\n5) Tablet detects it and immediately plays selected C2."
            textSize = 15f
            setTextIsSelectable(true)
            setPadding(0, dp(10), 0, dp(10))
        }
        val scroll = ScrollView(this).apply { addView(status) }
        root.addView(scroll, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f))

        setContentView(root)
        updateProbeInfo()
    }

    private fun chooseProbeFile(requestCode: Int) {
        if (responder?.isRunning() == true) {
            Toast.makeText(this, "Stop the responder before changing probe files", Toast.LENGTH_SHORT).show()
            return
        }
        val intent = Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
            addCategory(Intent.CATEGORY_OPENABLE)
            type = "audio/*"
            putExtra(
                Intent.EXTRA_MIME_TYPES,
                arrayOf("audio/wav", "audio/x-wav", "audio/wave", "audio/vnd.wave", "application/octet-stream")
            )
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION or Intent.FLAG_GRANT_PERSISTABLE_URI_PERMISSION)
        }
        startActivityForResult(intent, requestCode)
    }

    @Deprecated("Deprecated in Android API, retained for this minimal Activity implementation")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (resultCode != RESULT_OK) return
        if (requestCode != REQ_C1_FILE && requestCode != REQ_C2_FILE) return
        val uri = data?.data ?: return

        try {
            try {
                contentResolver.takePersistableUriPermission(uri, Intent.FLAG_GRANT_READ_URI_PERMISSION)
            } catch (_: Throwable) {
                // Some document providers do not expose persistable grants; current session can still use the file.
            }

            val loaded = WavProbeLoader.load(this, uri)
            if (requestCode == REQ_C1_FILE) {
                c1Signal = loaded
                getSharedPreferences(PREFS, MODE_PRIVATE).edit().putString(PREF_C1_URI, uri.toString()).apply()
                status.text = "C1 loaded successfully:\n${loaded.summary()}\n\nLinux must transmit this exact waveform for matched-filter detection."
            } else {
                c2Signal = loaded
                getSharedPreferences(PREFS, MODE_PRIVATE).edit().putString(PREF_C2_URI, uri.toString()).apply()
                status.text = "C2 loaded successfully:\n${loaded.summary()}\n\nTablet will play this waveform after detecting C1. Linux should use the same C2 as its receive template."
            }
            updateProbeInfo()
        } catch (t: Throwable) {
            Toast.makeText(this, "Cannot load WAV: ${t.message}", Toast.LENGTH_LONG).show()
            status.text = "PROBE FILE ERROR: ${t.javaClass.simpleName}: ${t.message}\nUse a normal RIFF/WAV file."
        }
    }

    private fun restoreSavedProbeFiles() {
        val prefs = getSharedPreferences(PREFS, MODE_PRIVATE)
        restoreOne(prefs.getString(PREF_C1_URI, null), true)
        restoreOne(prefs.getString(PREF_C2_URI, null), false)
        updateProbeInfo()
    }

    private fun restoreOne(uriString: String?, isC1: Boolean) {
        if (uriString.isNullOrBlank()) return
        try {
            val loaded = WavProbeLoader.load(this, Uri.parse(uriString))
            if (isC1) c1Signal = loaded else c2Signal = loaded
        } catch (_: Throwable) {
            getSharedPreferences(PREFS, MODE_PRIVATE).edit()
                .remove(if (isC1) PREF_C1_URI else PREF_C2_URI)
                .apply()
        }
    }

    private fun updateProbeInfo() {
        c1Info.text = "C1 template: ${c1Signal.summary()}"
        c2Info.text = "C2 reply: ${c2Signal.summary()}"
    }

    private fun makeResponder(): AcousticResponder {
        val c1Snapshot = c1Signal
        val c2Snapshot = c2Signal
        return AcousticResponder(
            context = this,
            c1Signal = c1Snapshot,
            c2Signal = c2Snapshot,
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
        selectC1.isEnabled = false
        selectC2.isEnabled = false
        defaultC1.isEnabled = false
        defaultC2.isEnabled = false
    }

    private fun setIdleButtons() {
        startStop.text = "ARM RESPONDER / WAIT FOR C1"
        testReply.isEnabled = true
        selectC1.isEnabled = true
        selectC2.isEnabled = true
        defaultC1.isEnabled = true
        defaultC2.isEnabled = true
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
