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
import kotlin.math.PI
import kotlin.math.cos

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
        const val HOP_SAMPLES = 240

        // Robust first-stage settings. v0.5 prioritizes reliable acoustic reply.
        const val C1_PRETRIGGER = 0.18
        const val C1_THRESHOLD = 0.28
        const val C1_MIN_BAND_RATIO = 0.80
    }

    data class Result(
        val t2Sample: Int,
        val t2Score: Double,
        val replyPlayCallNs: Long,
        val softwareDecisionToPlayUs: Double,
        val wavPath: String,
        val udpJson: String
    )

    private val running = AtomicBoolean(false)
    private var record: AudioRecord? = null
    private var track: AudioTrack? = null
    private var aec: AcousticEchoCanceler? = null
    private var ns: NoiseSuppressor? = null
    private var agc: AutomaticGainControl? = null
    private var replyQueued = false

    private val c1Full = Chirp.linearPcm16(SAMPLE_RATE, CHIRP_SEC, C1_F0, C1_F1)
    private val c2Full = Chirp.linearPcm16(SAMPLE_RATE, CHIRP_SEC, C2_F0, C2_F1)

    fun start(linuxHost: String, linuxPort: Int) {
        if (!running.compareAndSet(false, true)) return
        thread(name = "AVTwin-Responder") {
            Process.setThreadPriority(Process.THREAD_PRIORITY_AUDIO)
            try {
                runResponder(linuxHost, linuxPort)
            } catch (t: Throwable) {
                post("ERROR: ${t.javaClass.simpleName}: ${t.message}")
            } finally {
                cleanupAudio()
                running.set(false)
            }
        }
    }

    fun startReplyPlaybackTest() {
        if (!running.compareAndSet(false, true)) return
        thread(name = "AVTwin-ReplyTest") {
            Process.setThreadPriority(Process.THREAD_PRIORITY_AUDIO)
            try {
                val source = prepareAudio()
                post(
                    "C2 PLAYBACK TEST\n" +
                        "source=$source\n" +
                        "output=${outputDescription()}\n" +
                        "playing ${C2_F0.toInt()}-${C2_F1.toInt()} Hz for 200 ms..."
                )
                val callNs = System.nanoTime()
                playQueuedReply()
                Thread.sleep(350)
                post(
                    "C2 PLAYBACK TEST: DONE\n" +
                        "play() called at $callNs ns\n" +
                        "No hardware timestamp required.\n" +
                        "This only verifies that the tablet can issue the acoustic reply."
                )
            } catch (t: Throwable) {
                post("C2 PLAYBACK TEST ERROR: ${t.javaClass.simpleName}: ${t.message}")
            } finally {
                cleanupAudio()
                running.set(false)
            }
        }
    }

    fun stop() {
        running.set(false)
        try { record?.stop() } catch (_: Throwable) {}
    }

    fun isRunning(): Boolean = running.get()

    private fun runResponder(linuxHost: String, linuxPort: Int) {
        val sourceName = prepareAudio()
        val accumulator = ShortAccumulator(SAMPLE_RATE * 12)
        val hop = ShortArray(HOP_SAMPLES)
        var lastUiAt = 0L
        var lastC1Score = 0.0
        var lastBandRatio = 0.0
        var coarseT2 = -1

        post(
            "ARMED v0.5 - RESPONDER MODE\n" +
                "Goal: hear C1 -> immediately play C2\n" +
                "48 kHz mono PCM16 | source=$sourceName\n" +
                "C1=${C1_F0.toInt()}-${C1_F1.toInt()} Hz / 200 ms\n" +
                "C2=${C2_F0.toInt()}-${C2_F1.toInt()} Hz / 200 ms\n" +
                "output=${outputDescription()}\n" +
                "C2 is PRE-QUEUED; hardware timestamp is NOT required"
        )

        record!!.startRecording()

        while (running.get() && coarseT2 < 0) {
            val n = record!!.read(hop, 0, hop.size, AudioRecord.READ_BLOCKING)
            if (n <= 0) continue
            accumulator.append(hop, n)

            if (accumulator.size >= c1Full.size) {
                val newest = accumulator.size - c1Full.size
                val (candidate, score) = MatchedFilter.bestRecent(
                    accumulator,
                    c1Full,
                    newest,
                    HOP_SAMPLES * 2,
                    startStep = 8,
                    decimation = 8
                )
                lastC1Score = score
                lastBandRatio = if (score >= C1_PRETRIGGER) {
                    highToLowBandRatio(accumulator, candidate, c1Full.size)
                } else {
                    0.0
                }

                if (score >= C1_THRESHOLD && lastBandRatio >= C1_MIN_BAND_RATIO) {
                    coarseT2 = candidate
                }
            }

            val now = System.currentTimeMillis()
            if (now - lastUiAt >= 250 && coarseT2 < 0) {
                lastUiAt = now
                post(
                    "WAIT_C1\n" +
                        "time=${fmt(accumulator.size.toDouble() / SAMPLE_RATE, 2)} s\n" +
                        "C1 score=${fmt(lastC1Score)} / ${fmt(C1_THRESHOLD)}\n" +
                        "HF/LF=${if (lastBandRatio > 0.0) fmt(lastBandRatio) else "--"} / ${fmt(C1_MIN_BAND_RATIO)}\n" +
                        "reply status=READY"
                )
            }
        }

        if (!running.get() || coarseT2 < 0) return

        // Refine the arrival sample from audio already captured. This is useful for logging,
        // but v0.5 does NOT wait for any AudioTrack timestamp before replying.
        val preReplyAudio = accumulator.copy()
        val (t2, t2Score) = MatchedFilter.refineStrongest(
            preReplyAudio,
            c1Full,
            coarseT2,
            marginSamples = 96
        )

        val decisionNs = System.nanoTime()
        post(
            "C1 DETECTED\n" +
                "t2 sample=$t2\n" +
                "score=${fmt(t2Score)}\n" +
                "REPLYING C2 NOW..."
        )

        val playCallNs = System.nanoTime()
        playQueuedReply()
        val softwareDecisionToPlayUs = (playCallNs - decisionNs) / 1000.0

        // Tell Linux immediately that the Android responder has issued C2.
        val earlyJson = """{"type":"avtwin_android_reply","version":"0.5","status":"c2_play_issued","sample_rate":$SAMPLE_RATE,"timing_method":"software_play_call_only","t3_precise":false,"t2_sample":$t2,"t2_score":${fmt(t2Score, 5)},"reply_play_call_ns":$playCallNs,"decision_to_play_us":${fmt(softwareDecisionToPlayUs, 3)}}"""
        try {
            UdpReporter.send(linuxHost, linuxPort, earlyJson)
        } catch (_: Throwable) {
            // UDP is informational only; acoustic reply must never depend on Wi-Fi.
        }

        post(
            "REPLIED C2 ✓\n" +
                "C1 t2 sample=$t2 score=${fmt(t2Score)}\n" +
                "C2 play() issued\n" +
                "software decision->play=${fmt(softwareDecisionToPlayUs)} us\n" +
                "NOTE: this is not a precise acoustic t3\n" +
                "capturing a short tail..."
        )

        val tailTarget = accumulator.size + (0.55 * SAMPLE_RATE).toInt()
        while (running.get() && accumulator.size < tailTarget) {
            val n = record!!.read(hop, 0, hop.size, AudioRecord.READ_BLOCKING)
            if (n > 0) accumulator.append(hop, n)
        }
        try { record!!.stop() } catch (_: Throwable) {}

        val wav = saveWav(accumulator.copy(), "responder_v05")
        val finalJson = """{"type":"avtwin_android_reply","version":"0.5","status":"done","sample_rate":$SAMPLE_RATE,"timing_method":"software_play_call_only","t3_precise":false,"t2_sample":$t2,"t2_score":${fmt(t2Score, 5)},"reply_play_call_ns":$playCallNs,"decision_to_play_us":${fmt(softwareDecisionToPlayUs, 3)},"wav":"${escapeJson(wav.absolutePath)}"}"""

        try {
            UdpReporter.send(linuxHost, linuxPort, finalJson)
        } catch (_: Throwable) {}

        post(
            "DONE v0.5\n" +
                "C1 detected ✓\n" +
                "C2 acoustic reply issued ✓\n" +
                "t2=$t2 score=${fmt(t2Score)}\n" +
                "decision->play=${fmt(softwareDecisionToPlayUs)} us\n" +
                "UDP target=$linuxHost:$linuxPort\n" +
                "WAV=${wav.absolutePath}\n\n" +
                "Next: Linux listens for the returned C2."
        )

        onResult(
            Result(
                t2Sample = t2,
                t2Score = t2Score,
                replyPlayCallNs = playCallNs,
                softwareDecisionToPlayUs = softwareDecisionToPlayUs,
                wavPath = wav.absolutePath,
                udpJson = finalJson
            )
        )
    }

    private fun prepareAudio(): String {
        val audioManager = context.getSystemService(Context.AUDIO_SERVICE) as AudioManager
        val unprocessed = audioManager.getProperty(AudioManager.PROPERTY_SUPPORT_AUDIO_SOURCE_UNPROCESSED) == "true"
        val source = if (unprocessed) {
            MediaRecorder.AudioSource.UNPROCESSED
        } else {
            MediaRecorder.AudioSource.VOICE_RECOGNITION
        }
        val sourceName = if (unprocessed) "UNPROCESSED" else "VOICE_RECOGNITION"

        val minBuffer = AudioRecord.getMinBufferSize(
            SAMPLE_RATE,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT
        )
        require(minBuffer > 0) { "48 kHz mono PCM16 AudioRecord is not supported on this device" }

        record = AudioRecord.Builder()
            .setAudioSource(source)
            .setAudioFormat(
                AudioFormat.Builder()
                    .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                    .setSampleRate(SAMPLE_RATE)
                    .setChannelMask(AudioFormat.CHANNEL_IN_MONO)
                    .build()
            )
            .setBufferSizeInBytes(maxOf(minBuffer * 8, HOP_SAMPLES * 24))
            .build()
        require(record?.state == AudioRecord.STATE_INITIALIZED) { "AudioRecord initialization failed" }
        disablePreprocessing(record!!.audioSessionId)

        prepareReplyTrackAndQueueC2()
        return sourceName
    }

    private fun prepareReplyTrackAndQueueC2() {
        fun makeTrack(mask: Int): AudioTrack {
            val minOut = AudioTrack.getMinBufferSize(
                SAMPLE_RATE,
                mask,
                AudioFormat.ENCODING_PCM_16BIT
            )
            require(minOut > 0) { "48 kHz PCM16 AudioTrack not supported for channel mask $mask: $minOut" }

            val requiredSamples = if (mask == AudioFormat.CHANNEL_OUT_MONO) {
                c2Full.size
            } else {
                c2Full.size * 2
            }
            val requiredBytes = requiredSamples * 2

            return AudioTrack.Builder()
                .setAudioAttributes(
                    AudioAttributes.Builder()
                        .setUsage(AudioAttributes.USAGE_MEDIA)
                        .setContentType(AudioAttributes.CONTENT_TYPE_MUSIC)
                        .build()
                )
                .setAudioFormat(
                    AudioFormat.Builder()
                        .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                        .setSampleRate(SAMPLE_RATE)
                        .setChannelMask(mask)
                        .build()
                )
                .setBufferSizeInBytes(maxOf(minOut * 2, requiredBytes + 4096))
                .setTransferMode(AudioTrack.MODE_STREAM)
                .build()
        }

        track = try {
            makeTrack(AudioFormat.CHANNEL_OUT_MONO)
        } catch (_: Throwable) {
            makeTrack(AudioFormat.CHANNEL_OUT_STEREO)
        }
        require(track?.state == AudioTrack.STATE_INITIALIZED) {
            "AudioTrack streaming initialization failed"
        }
        track!!.setVolume(1.0f)

        // Pre-queue the fixed C2 while the track is stopped. Detection then only needs play().
        queueReplySamples(track!!)
        replyQueued = true
    }

    private fun queueReplySamples(t: AudioTrack) {
        try { t.pause() } catch (_: Throwable) {}
        try { t.flush() } catch (_: Throwable) {}

        if (t.channelCount == 1) {
            val written = t.write(c2Full, 0, c2Full.size, AudioTrack.WRITE_BLOCKING)
            require(written == c2Full.size) {
                "Could not pre-queue mono C2: $written/${c2Full.size}"
            }
        } else {
            val stereo = ShortArray(c2Full.size * 2)
            for (i in c2Full.indices) {
                stereo[2 * i] = c2Full[i]
                stereo[2 * i + 1] = c2Full[i]
            }
            val written = t.write(stereo, 0, stereo.size, AudioTrack.WRITE_BLOCKING)
            require(written == stereo.size) {
                "Could not pre-queue stereo C2: $written/${stereo.size}"
            }
        }
    }

    private fun playQueuedReply() {
        val t = track ?: error("AudioTrack not prepared")
        require(replyQueued) { "C2 reply was not pre-queued" }
        t.play()
        replyQueued = false
    }

    private fun highToLowBandRatio(audio: ShortAccumulator, start: Int, length: Int): Double {
        if (start < 0 || start + length > audio.size) return 0.0
        val high = doubleArrayOf(12_000.0, 14_000.0, 16_000.0, 18_000.0)
        val low = doubleArrayOf(2_000.0, 4_000.0, 6_000.0, 8_000.0)
        var hi = 0.0
        var lo = 0.0
        for (f in high) hi += goertzelPower(audio, start, length, f)
        for (f in low) lo += goertzelPower(audio, start, length, f)
        return hi / (lo + 1.0)
    }

    private fun goertzelPower(audio: ShortAccumulator, start: Int, length: Int, freq: Double): Double {
        val omega = 2.0 * PI * freq / SAMPLE_RATE
        val coeff = 2.0 * cos(omega)
        var s1 = 0.0
        var s2 = 0.0
        var i = 0
        while (i < length) {
            val sample = audio[start + i].toDouble()
            val s0 = sample + coeff * s1 - s2
            s2 = s1
            s1 = s0
            i++
        }
        return s1 * s1 + s2 * s2 - coeff * s1 * s2
    }

    private fun outputDescription(): String {
        val channels = track?.channelCount ?: 0
        return "STREAM/${if (channels == 1) "MONO" else if (channels == 2) "STEREO" else "${channels}ch"}"
    }

    private fun saveWav(audio: ShortArray, prefix: String): File {
        val stamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        val baseDir = context.getExternalFilesDir(null) ?: context.filesDir
        val dir = File(baseDir, "recordings")
        val wav = File(dir, "${prefix}_$stamp.wav")
        WavWriter.writeMonoPcm16(wav, audio, SAMPLE_RATE)
        return wav
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

    private fun cleanupAudio() {
        try { record?.stop() } catch (_: Throwable) {}
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
        replyQueued = false
    }

    private fun escapeJson(value: String): String = value.replace("\\", "\\\\").replace("\"", "\\\"")
    private fun fmt(v: Double, digits: Int = 3): String = "% .${digits}f".format(Locale.US, v).trim()
    private fun post(s: String) = onStatus(s)
}
