package com.example.avtwinresponder

import android.content.Context
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioRecord
import android.media.AudioTrack
import android.media.MediaRecorder
import android.media.audiofx.AcousticEchoCanceler
import android.media.audiofx.AutomaticGainControl
import android.media.audiofx.NoiseSuppressor
import android.os.Process
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.concurrent.thread

class AcousticResponder(
    private val context: Context,
    private val onStatus: (String) -> Unit,
    private val onResult: (Result) -> Unit
) {
    companion object {
        const val SAMPLE_RATE = 48_000
        const val C1_F0 = 11_000.0
        const val C1_F1 = 19_000.0
        const val C2_F0 = 300.0
        const val C2_F1 = 9_000.0
        const val CHIRP_SEC = 0.200
        const val DETECT_PREFIX_SEC = 0.050
        const val HOP_SAMPLES = 240
        const val DETECT_DECIMATION = 16
        const val C1_THRESHOLD = 0.35
        const val C2_THRESHOLD = 0.30
    }

    data class Result(
        val t2Sample: Int,
        val t3Sample: Int,
        val replyDelaySamples: Int,
        val replyDelayMs: Double,
        val t2Score: Double,
        val t3Score: Double,
        val wavPath: String,
        val udpJson: String
    )

    private enum class State { WAIT_C1, WAIT_C2, DONE }

    private val running = AtomicBoolean(false)
    private var record: AudioRecord? = null
    private var track: AudioTrack? = null
    private var aec: AcousticEchoCanceler? = null
    private var ns: NoiseSuppressor? = null
    private var agc: AutomaticGainControl? = null

    private val c1Full = Chirp.linearPcm16(SAMPLE_RATE, CHIRP_SEC, C1_F0, C1_F1)
    private val c2Full = Chirp.linearPcm16(SAMPLE_RATE, CHIRP_SEC, C2_F0, C2_F1)
    private val prefixCount = (SAMPLE_RATE * DETECT_PREFIX_SEC).toInt()
    private val c1Prefix = Chirp.prefix(c1Full, prefixCount)
    private val c2Prefix = Chirp.prefix(c2Full, prefixCount)

    fun start(linuxHost: String, linuxPort: Int) {
        if (!running.compareAndSet(false, true)) return
        thread(name = "AVTwin-Audio") {
            Process.setThreadPriority(Process.THREAD_PRIORITY_AUDIO)
            try {
                runLoop(linuxHost, linuxPort)
            } catch (t: Throwable) {
                post("ERROR: ${t.javaClass.simpleName}: ${t.message}")
            } finally {
                cleanup()
                running.set(false)
            }
        }
    }

    fun stop() {
        running.set(false)
        try { record?.stop() } catch (_: Throwable) {}
    }

    fun isRunning(): Boolean = running.get()

    private fun runLoop(linuxHost: String, linuxPort: Int) {
        val audioManager = context.getSystemService(Context.AUDIO_SERVICE) as AudioManager
        val unprocessed = audioManager.getProperty(AudioManager.PROPERTY_SUPPORT_AUDIO_SOURCE_UNPROCESSED) == "true"
        val source = if (unprocessed) MediaRecorder.AudioSource.UNPROCESSED else MediaRecorder.AudioSource.VOICE_RECOGNITION

        val minBuffer = AudioRecord.getMinBufferSize(
            SAMPLE_RATE,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT
        )
        require(minBuffer > 0) { "48 kHz mono PCM16 AudioRecord is not supported on this device" }
        val bufferBytes = maxOf(minBuffer * 8, HOP_SAMPLES * 2 * 8)

        record = AudioRecord.Builder()
            .setAudioSource(source)
            .setAudioFormat(
                AudioFormat.Builder()
                    .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                    .setSampleRate(SAMPLE_RATE)
                    .setChannelMask(AudioFormat.CHANNEL_IN_MONO)
                    .build()
            )
            .setBufferSizeInBytes(bufferBytes)
            .build()

        require(record?.state == AudioRecord.STATE_INITIALIZED) { "AudioRecord initialization failed" }
        disablePreprocessing(record!!.audioSessionId)
        prepareC2Track()

        val accumulator = ShortAccumulator(SAMPLE_RATE * 8)
        val hop = ShortArray(HOP_SAMPLES)
        var state = State.WAIT_C1
        var coarseT2 = -1
        var coarseT3 = -1
        var lastUiAt = 0L

        post("ARMED\n48 kHz mono PCM16\nsource=${if (unprocessed) "UNPROCESSED" else "VOICE_RECOGNITION"}\noutput=MODE_STREAM\nwaiting for C1 ${C1_F0.toInt()}-${C1_F1.toInt()} Hz")
        record!!.startRecording()

        while (running.get() && state != State.DONE) {
            val n = record!!.read(hop, 0, hop.size, AudioRecord.READ_BLOCKING)
            if (n <= 0) continue
            accumulator.append(hop, n)

            val now = System.currentTimeMillis()
            if (now - lastUiAt > 500) {
                lastUiAt = now
                post("${state.name}\nsamples=${accumulator.size}\ntime=${"%.2f".format(Locale.US, accumulator.size.toDouble() / SAMPLE_RATE)} s")
            }

            when (state) {
                State.WAIT_C1 -> {
                    if (accumulator.size >= c1Prefix.size) {
                        val newest = accumulator.size - c1Prefix.size
                        val (candidate, score) = MatchedFilter.bestRecent(
                            accumulator, c1Prefix, newest, HOP_SAMPLES, startStep = 1, decimation = DETECT_DECIMATION
                        )
                        if (score >= C1_THRESHOLD) {
                            coarseT2 = candidate
                            state = State.WAIT_C2
                            post("C1 DETECTED coarse=$coarseT2 score=${"%.3f".format(Locale.US, score)}\nreplying C2 now...")
                            playC2()
                        }
                    }
                }
                State.WAIT_C2 -> {
                    if (accumulator.size >= c2Prefix.size) {
                        val newest = accumulator.size - c2Prefix.size
                        val (candidate, score) = MatchedFilter.bestRecent(
                            accumulator, c2Prefix, newest, HOP_SAMPLES, startStep = 1, decimation = DETECT_DECIMATION
                        )
                        val safelyAfterC1 = candidate > coarseT2 + (0.020 * SAMPLE_RATE).toInt()
                        if (safelyAfterC1 && score >= C2_THRESHOLD) {
                            coarseT3 = candidate
                            state = State.DONE
                            post("C2 SELF-DETECTED coarse=$coarseT3 score=${"%.3f".format(Locale.US, score)}\nrefining sample indices...")
                        }
                    }
                }
                State.DONE -> Unit
            }
        }

        try { record!!.stop() } catch (_: Throwable) {}
        if (!running.get() || coarseT2 < 0 || coarseT3 < 0) return

        val audio = accumulator.copy()
        val (t2, t2Score) = MatchedFilter.refineStrongest(audio, c1Prefix, coarseT2, marginSamples = 64)
        val (t3, t3Score) = MatchedFilter.refineStrongest(audio, c2Prefix, coarseT3, marginSamples = 64)
        val delta = t3 - t2
        val deltaMs = delta * 1000.0 / SAMPLE_RATE

        val stamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        val baseDir = context.getExternalFilesDir(null) ?: context.filesDir
        val dir = File(baseDir, "recordings")
        val wav = File(dir, "handshake_$stamp.wav")
        WavWriter.writeMonoPcm16(wav, audio, SAMPLE_RATE)

        val json = """{"type":"avtwin_android_reply","sample_rate":$SAMPLE_RATE,"t2_sample":$t2,"t3_sample":$t3,"reply_delay_samples":$delta,"reply_delay_ms":${"%.6f".format(Locale.US, deltaMs)},"t2_score":${"%.5f".format(Locale.US, t2Score)},"t3_score":${"%.5f".format(Locale.US, t3Score)}}"""

        thread(name = "AVTwin-UDP") {
            try {
                UdpReporter.send(linuxHost, linuxPort, json)
                post("DONE\nT2=$t2\nT3=$t3\nDelta=${"%.3f".format(Locale.US, deltaMs)} ms\nUDP -> $linuxHost:$linuxPort\nWAV=${wav.absolutePath}")
            } catch (t: Throwable) {
                post("DONE (UDP failed: ${t.message})\nT2=$t2\nT3=$t3\nDelta=${"%.3f".format(Locale.US, deltaMs)} ms\nWAV=${wav.absolutePath}")
            }
        }

        onResult(Result(t2, t3, delta, deltaMs, t2Score, t3Score, wav.absolutePath, json))
    }

    private fun prepareC2Track() {
        val minOut = AudioTrack.getMinBufferSize(
            SAMPLE_RATE,
            AudioFormat.CHANNEL_OUT_MONO,
            AudioFormat.ENCODING_PCM_16BIT
        )
        require(minOut > 0) { "48 kHz mono PCM16 AudioTrack streaming is not supported on this device: $minOut" }

        val bufferBytes = maxOf(minOut * 2, c2Full.size * 2)
        track = AudioTrack.Builder()
            .setAudioAttributes(
                AudioAttributes.Builder()
                    .setUsage(AudioAttributes.USAGE_MEDIA)
                    .setContentType(AudioAttributes.CONTENT_TYPE_SONIFICATION)
                    .build()
            )
            .setAudioFormat(
                AudioFormat.Builder()
                    .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                    .setSampleRate(SAMPLE_RATE)
                    .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                    .build()
            )
            .setBufferSizeInBytes(bufferBytes)
            .setTransferMode(AudioTrack.MODE_STREAM)
            .build()

        require(track?.state == AudioTrack.STATE_INITIALIZED) { "AudioTrack streaming initialization failed" }
        track!!.setVolume(1.0f)
    }

    private fun playC2() {
        val t = track ?: return
        try { t.pause() } catch (_: Throwable) {}
        try { t.flush() } catch (_: Throwable) {}
        t.play()
        val written = t.write(c2Full, 0, c2Full.size, AudioTrack.WRITE_BLOCKING)
        require(written == c2Full.size) { "Could not stream C2 to AudioTrack: $written/${c2Full.size}" }
    }

    private fun disablePreprocessing(sessionId: Int) {
        try {
            if (AcousticEchoCanceler.isAvailable()) {
                aec = AcousticEchoCanceler.create(sessionId)
                aec?.enabled = false
            }
        } catch (_: Throwable) {}
        try {
            if (NoiseSuppressor.isAvailable()) {
                ns = NoiseSuppressor.create(sessionId)
                ns?.enabled = false
            }
        } catch (_: Throwable) {}
        try {
            if (AutomaticGainControl.isAvailable()) {
                agc = AutomaticGainControl.create(sessionId)
                agc?.enabled = false
            }
        } catch (_: Throwable) {}
    }

    private fun cleanup() {
        try { record?.release() } catch (_: Throwable) {}
        try { track?.stop() } catch (_: Throwable) {}
        try { track?.release() } catch (_: Throwable) {}
        try { aec?.release() } catch (_: Throwable) {}
        try { ns?.release() } catch (_: Throwable) {}
        try { agc?.release() } catch (_: Throwable) {}
        record = null
        track = null
        aec = null
        ns = null
        agc = null
    }

    private fun post(s: String) = onStatus(s)
}
