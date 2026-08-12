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
import android.view.WindowManager
import android.widget.Button
import android.widget.CheckBox
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
        private const val REQ_RESULT_TREE = 3001
        private const val PREFS = "avtwin_session_prefs"
        private const val PREF_C1_URI = "c1_uri"
        private const val PREF_C2_URI = "c2_uri"
        private const val PREF_TREE_URI = "result_tree_uri"
        private const val PREF_LINUX_IP = "linux_ip"
        private const val PREF_CONTROL_PORT = "control_port"
        private const val PREF_RESULT_PORT = "result_port"
        private const val PREF_DEBUG_AUDIO = "debug_audio"
    }

    private lateinit var c1Info: TextView
    private lateinit var c2Info: TextView
    private lateinit var folderInfo: TextView
    private lateinit var status: TextView
    private lateinit var metrics: TextView
    private lateinit var host: EditText
    private lateinit var controlPort: EditText
    private lateinit var resultPort: EditText
    private lateinit var debugAudio: CheckBox
    private lateinit var selectC1: Button
    private lateinit var selectC2: Button
    private lateinit var defaultC1: Button
    private lateinit var defaultC2: Button
    private lateinit var selectFolder: Button
    private lateinit var startSession: Button
    private lateinit var pauseResume: Button
    private lateinit var stopSession: Button
    private lateinit var udpTest: Button
    private lateinit var c2Test: Button

    private var c1Signal: ProbeSignal = ProbeDefaults.c1()
    private var c2Signal: ProbeSignal = ProbeDefaults.c2()
    private var resultTreeUri: Uri? = null
    private var responder: AcousticResponder? = null
    private var paused = false
    private val logBuffer = StringBuilder()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        buildUi()
        ensureAudioPermission()
        restoreSettings()
        updateProbeInfo()
        updateFolderInfo()
        showIdleMetrics()
    }

    private fun buildUi() {
        val density = resources.displayMetrics.density
        fun dp(x: Int) = (x * density).toInt()
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(18), dp(14), dp(18), dp(14))
        }
        root.addView(TextView(this).apply {
            text = "AV-Twin Continuous Acoustic Responder v0.8"
            textSize = 21f
            gravity = Gravity.CENTER_HORIZONTAL
        })
        root.addView(TextView(this).apply {
            text = "Android Tx: persistent 48 kHz capture -> C1 -> C2 -> audio-frame timing -> UDP -> cooldown -> listen again"
            textSize = 13f
            setPadding(0, dp(5), 0, dp(8))
        })

        c1Info = TextView(this).apply { textSize = 12f; setTextIsSelectable(true) }
        root.addView(c1Info)
        val c1Row = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        selectC1 = Button(this).apply { text = "SELECT C1 WAV"; setOnClickListener { chooseProbeFile(REQ_C1_FILE) } }
        defaultC1 = Button(this).apply { text = "DEFAULT C1"; setOnClickListener { setDefaultProbe(true) } }
        c1Row.addView(selectC1, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
        c1Row.addView(defaultC1, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
        root.addView(c1Row)

        c2Info = TextView(this).apply { textSize = 12f; setTextIsSelectable(true); setPadding(0, dp(3), 0, 0) }
        root.addView(c2Info)
        val c2Row = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        selectC2 = Button(this).apply { text = "SELECT C2 WAV"; setOnClickListener { chooseProbeFile(REQ_C2_FILE) } }
        defaultC2 = Button(this).apply { text = "DEFAULT C2"; setOnClickListener { setDefaultProbe(false) } }
        c2Row.addView(selectC2, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
        c2Row.addView(defaultC2, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
        root.addView(c2Row)

        host = EditText(this).apply { hint = "Linux IP"; inputType = InputType.TYPE_CLASS_TEXT }
        root.addView(host)
        val portRow = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        controlPort = EditText(this).apply { hint = "ARM/control port"; inputType = InputType.TYPE_CLASS_NUMBER }
        resultPort = EditText(this).apply { hint = "result port"; inputType = InputType.TYPE_CLASS_NUMBER }
        portRow.addView(controlPort, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
        portRow.addView(resultPort, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
        root.addView(portRow)

        folderInfo = TextView(this).apply { textSize = 12f; setTextIsSelectable(true); setPadding(0, dp(5), 0, 0) }
        root.addView(folderInfo)
        selectFolder = Button(this).apply { text = "选择结果保存目录"; setOnClickListener { chooseResultFolder() } }
        root.addView(selectFolder)
        debugAudio = CheckBox(this).apply { text = "保存调试音频（默认关闭）" }
        root.addView(debugAudio)

        startSession = Button(this).apply { text = "开始会话"; setOnClickListener { startContinuousSession() } }
        root.addView(startSession)
        val sessionRow = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        pauseResume = Button(this).apply { text = "暂停监听"; isEnabled = false; setOnClickListener { togglePause() } }
        stopSession = Button(this).apply { text = "安全停止并保存"; isEnabled = false; setOnClickListener { safeStop() } }
        sessionRow.addView(pauseResume, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
        sessionRow.addView(stopSession, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
        root.addView(sessionRow)

        val testRow = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        udpTest = Button(this).apply { text = "测试 UDP（无声音）"; setOnClickListener { runUdpTest() } }
        c2Test = Button(this).apply { text = "TEST C2 x20"; setOnClickListener { runC2Test() } }
        testRow.addView(udpTest, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
        testRow.addView(c2Test, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
        root.addView(testRow)

        metrics = TextView(this).apply { textSize = 13f; setTextIsSelectable(true); setPadding(0, dp(6), 0, dp(4)) }
        root.addView(metrics)
        status = TextView(this).apply { textSize = 12f; setTextIsSelectable(true) }
        val scroll = ScrollView(this).apply { addView(status) }
        root.addView(scroll, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f))
        setContentView(root)
    }

    private fun startContinuousSession() {
        if (!ensurePermissionForAction()) return
        if (responder?.isRunning() == true || responder?.isTestRunning() == true) return
        val tree = resultTreeUri
        if (tree == null) {
            Toast.makeText(this, "请先选择结果保存目录", Toast.LENGTH_LONG).show()
            return
        }
        val validation = SafSessionStorage.validateTree(this, tree)
        if (!validation.first) {
            Toast.makeText(this, "保存目录不可用：${validation.second}，请重新授权", Toast.LENGTH_LONG).show()
            return
        }
        val ip = host.text.toString().trim()
        val cp = controlPort.text.toString().toIntOrNull() ?: 5006
        val rp = resultPort.text.toString().toIntOrNull() ?: 5005
        if (ip.isBlank() || cp !in 1..65535 || rp !in 1..65535) {
            Toast.makeText(this, "检查 Linux IP / 端口", Toast.LENGTH_LONG).show()
            return
        }
        saveSettings()
        responder = makeResponder()
        setSessionControls(true)
        paused = false
        pauseResume.text = "暂停监听"
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        appendLog("START requested; validating storage and preparing persistent C2 before LISTENING")
        responder!!.startSession(
            AcousticResponder.SessionConfig(
                linuxHost = ip,
                controlPort = cp,
                resultPort = rp,
                resultTreeUri = tree,
                saveDebugAudio = debugAudio.isChecked
            )
        )
    }

    private fun togglePause() {
        val r = responder ?: return
        if (!r.isRunning()) return
        if (!paused) {
            r.pauseListening()
            paused = true
            pauseResume.text = "继续监听"
        } else {
            r.resumeListening()
            paused = false
            pauseResume.text = "暂停监听"
        }
    }

    private fun safeStop() {
        responder?.stopAndSave()
        appendLog("Safe stop requested; waiting for session finalization")
    }

    private fun runUdpTest() {
        val ip = host.text.toString().trim()
        val rp = resultPort.text.toString().toIntOrNull() ?: 5005
        if (ip.isBlank()) return
        if (responder?.isRunning() == true) responder?.testUdp(ip, rp) else {
            responder = makeResponder()
            responder!!.testUdp(ip, rp)
        }
    }

    private fun runC2Test() {
        if (!ensurePermissionForAction()) return
        if (responder?.isRunning() == true || responder?.isTestRunning() == true) return
        responder = makeResponder()
        setProbeControls(false)
        c2Test.isEnabled = false
        responder!!.startRepeatedPlaybackTest()
    }

    private fun makeResponder(): AcousticResponder = AcousticResponder(
        context = this,
        c1Signal = c1Signal,
        c2Signal = c2Signal,
        onStatus = { message ->
            runOnUiThread {
                appendLog(message)
                if (responder?.isTestRunning() != true && responder?.isRunning() != true) setProbeControls(true)
            }
        },
        onSnapshot = { snap -> runOnUiThread { updateSnapshot(snap) } }
    )

    private fun updateSnapshot(s: AcousticResponder.SessionSnapshot) {
        metrics.text =
            "state=${s.state}\n" +
                "session_id=${s.sessionId ?: "--"}\n" +
                "measurement_id=${s.measurementId ?: "--"} | pending ARM=${s.pendingArmMeasurementId ?: "--"}\n" +
                "成功响应=${s.successResponses} | C1未通过=${s.c1Rejected} | C2失败=${s.c2Failures} | UDP失败=${s.udpFailures}\n" +
                "last reply_delay_samples=${s.lastReplyDelaySamples ?: "--"} | t3_precise=${s.lastT3Precise}\n" +
                "input=${s.inputRoute}\noutput=${s.outputRoute}\n" +
                "note=${s.note}"
        if (s.state == "STOPPED") {
            setSessionControls(false)
            setProbeControls(true)
            window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
            paused = false
            pauseResume.text = "暂停监听"
        }
    }

    private fun showIdleMetrics() {
        metrics.text = "state=STOPPED\nsession_id=--\nmeasurement_id=--\n成功响应=0 | C1未通过=0 | C2失败=0 | UDP失败=0\nt3_precise=false"
    }

    private fun chooseProbeFile(requestCode: Int) {
        if (responder?.isRunning() == true) return
        val intent = Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
            addCategory(Intent.CATEGORY_OPENABLE)
            type = "audio/*"
            putExtra(Intent.EXTRA_MIME_TYPES, arrayOf("audio/wav", "audio/x-wav", "audio/wave", "audio/vnd.wave", "application/octet-stream"))
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION or Intent.FLAG_GRANT_PERSISTABLE_URI_PERMISSION)
        }
        startActivityForResult(intent, requestCode)
    }

    private fun chooseResultFolder() {
        if (responder?.isRunning() == true) return
        val intent = Intent(Intent.ACTION_OPEN_DOCUMENT_TREE).apply {
            addFlags(
                Intent.FLAG_GRANT_READ_URI_PERMISSION or Intent.FLAG_GRANT_WRITE_URI_PERMISSION or
                    Intent.FLAG_GRANT_PERSISTABLE_URI_PERMISSION or Intent.FLAG_GRANT_PREFIX_URI_PERMISSION
            )
        }
        startActivityForResult(intent, REQ_RESULT_TREE)
    }

    @Deprecated("Kept for the minimal Activity implementation")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (resultCode != RESULT_OK) return
        val uri = data?.data ?: return
        try {
            if (requestCode == REQ_RESULT_TREE) {
                val flags = data.flags and (Intent.FLAG_GRANT_READ_URI_PERMISSION or Intent.FLAG_GRANT_WRITE_URI_PERMISSION)
                contentResolver.takePersistableUriPermission(uri, flags)
                resultTreeUri = uri
                getSharedPreferences(PREFS, MODE_PRIVATE).edit().putString(PREF_TREE_URI, uri.toString()).apply()
                updateFolderInfo()
                return
            }
            if (requestCode != REQ_C1_FILE && requestCode != REQ_C2_FILE) return
            try { contentResolver.takePersistableUriPermission(uri, Intent.FLAG_GRANT_READ_URI_PERMISSION) } catch (_: Throwable) {}
            val loaded = WavProbeLoader.load(this, uri)
            if (requestCode == REQ_C1_FILE) {
                c1Signal = loaded
                getSharedPreferences(PREFS, MODE_PRIVATE).edit().putString(PREF_C1_URI, uri.toString()).apply()
                appendLog("C1 loaded: ${loaded.summary()} SHA256=${loaded.sourceSha256}")
            } else {
                c2Signal = loaded
                getSharedPreferences(PREFS, MODE_PRIVATE).edit().putString(PREF_C2_URI, uri.toString()).apply()
                appendLog("C2 loaded: ${loaded.summary()} SHA256=${loaded.sourceSha256}")
            }
            updateProbeInfo()
        } catch (t: Throwable) {
            appendLog("FILE/DIRECTORY ERROR: ${t.javaClass.simpleName}: ${t.message}")
            Toast.makeText(this, t.message ?: "选择失败", Toast.LENGTH_LONG).show()
        }
    }

    private fun setDefaultProbe(c1: Boolean) {
        if (c1) {
            c1Signal = ProbeDefaults.c1()
            getSharedPreferences(PREFS, MODE_PRIVATE).edit().remove(PREF_C1_URI).apply()
        } else {
            c2Signal = ProbeDefaults.c2()
            getSharedPreferences(PREFS, MODE_PRIVATE).edit().remove(PREF_C2_URI).apply()
        }
        updateProbeInfo()
    }

    private fun restoreSettings() {
        val p = getSharedPreferences(PREFS, MODE_PRIVATE)
        host.setText(p.getString(PREF_LINUX_IP, "192.168.1.100"))
        controlPort.setText(p.getInt(PREF_CONTROL_PORT, 5006).toString())
        resultPort.setText(p.getInt(PREF_RESULT_PORT, 5005).toString())
        debugAudio.isChecked = p.getBoolean(PREF_DEBUG_AUDIO, false)
        p.getString(PREF_TREE_URI, null)?.let { resultTreeUri = Uri.parse(it) }
        restoreProbe(p.getString(PREF_C1_URI, null), true)
        restoreProbe(p.getString(PREF_C2_URI, null), false)
    }

    private fun restoreProbe(uriString: String?, c1: Boolean) {
        if (uriString.isNullOrBlank()) return
        try {
            val loaded = WavProbeLoader.load(this, Uri.parse(uriString))
            if (c1) c1Signal = loaded else c2Signal = loaded
        } catch (_: Throwable) {
            getSharedPreferences(PREFS, MODE_PRIVATE).edit().remove(if (c1) PREF_C1_URI else PREF_C2_URI).apply()
        }
    }

    private fun saveSettings() {
        getSharedPreferences(PREFS, MODE_PRIVATE).edit()
            .putString(PREF_LINUX_IP, host.text.toString().trim())
            .putInt(PREF_CONTROL_PORT, controlPort.text.toString().toIntOrNull() ?: 5006)
            .putInt(PREF_RESULT_PORT, resultPort.text.toString().toIntOrNull() ?: 5005)
            .putBoolean(PREF_DEBUG_AUDIO, debugAudio.isChecked)
            .apply()
    }

    private fun updateProbeInfo() {
        c1Info.text = "C1: ${c1Signal.summary()}\n${c1Signal.channelDiagnostics()}"
        c2Info.text = "C2: ${c2Signal.summary()}\n${c2Signal.channelDiagnostics()}"
    }

    private fun updateFolderInfo() {
        val uri = resultTreeUri
        if (uri == null) {
            folderInfo.text = "结果保存目录：未选择"
            return
        }
        val validation = SafSessionStorage.validateTree(this, uri)
        val label = SafSessionStorage.displayName(this, uri) ?: "selected tree"
        folderInfo.text = "结果保存目录：$label\n$uri\npermission=${if (validation.first) "OK" else "INVALID: ${validation.second}"}"
    }

    private fun setSessionControls(active: Boolean) {
        startSession.isEnabled = !active
        pauseResume.isEnabled = active
        stopSession.isEnabled = active
        host.isEnabled = !active
        controlPort.isEnabled = !active
        resultPort.isEnabled = !active
        debugAudio.isEnabled = !active
        selectFolder.isEnabled = !active
        udpTest.isEnabled = !active
        c2Test.isEnabled = !active
        setProbeControls(!active)
    }

    private fun setProbeControls(enabled: Boolean) {
        selectC1.isEnabled = enabled
        selectC2.isEnabled = enabled
        defaultC1.isEnabled = enabled
        defaultC2.isEnabled = enabled
        if (responder?.isRunning() != true) c2Test.isEnabled = enabled
    }

    private fun appendLog(message: String) {
        if (message.isBlank()) return
        logBuffer.append(message).append('\n')
        if (logBuffer.length > 24000) logBuffer.delete(0, 6000)
        status.text = logBuffer.toString()
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
        responder?.stopAndSave()
        super.onDestroy()
    }
}
