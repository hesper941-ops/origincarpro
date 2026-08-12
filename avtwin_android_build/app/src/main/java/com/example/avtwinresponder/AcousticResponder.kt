package com.example.avtwinresponder

import android.content.Context
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioRecord
import android.media.AudioTrack
import android.media.MediaRecorder
import android.media.audiofx.AcousticEchoCanceler
import android.media.audiofx.AutomaticGainControl
import android.media.audiofx.NoiseSuppressor
import android.os.Process
import android.util.Log
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
    private val c1Signal: ProbeSignal,
    private val c2Signal: ProbeSignal,
    private val onStatus: (String) -> Unit,
    private val onResult: (Result) -> Unit
) {
    companion object {
        const val SAMPLE_RATE = ProbeSignal.SAMPLE_RATE
        const val HOP_SAMPLES = 240
        const val C1_PRETRIGGER = 0.18
        const val C1_THRESHOLD = 0.28
        const val C1_MIN_BAND_RATIO = 0.80
        private const val TAG = "AVTwinResponder"
        private const val TEST_CYCLES = 20
    }

    data class Result(
        val t2Sample: Int,
        val t2Score: Double,
        val replyPlayCallNs: Long,
        val softwareDecisionToPlayUs: Double,
        val playbackVerified: Boolean,
        val firstValidAudioTimestampNs: Long?,
        val firstValidAudioFramePosition: Long?,
        val wavPath: String,
        val udpJson: String
    )

    private val running = AtomicBoolean(false)
    private var record: AudioRecord? = null
    private var c2Player: PersistentC2Player? = null
    private var aec: AcousticEchoCanceler? = null
    private var ns: NoiseSuppressor? = null
    private var agc: AutomaticGainControl? = null

    private val c1Full = c1Signal.samples
    private val c2Full = c2Signal.samples

    init {
        require(c1Full.isNotEmpty()) { "C1 template is empty" }
        require(c2Full.isNotEmpty()) { "C2 reply is empty" }
    }

    fun start(linuxHost: String, linuxPort: Int) {
        if (!running.compareAndSet(false, true)) return
        thread(name = "AVTwin-Responder") {
            Process.setThreadPriority(Process.THREAD_PRIORITY_AUDIO)
            try {
                runResponder(linuxHost, linuxPort)
            } catch (t: Throwable) {
                post("ERROR: ${t.javaClass.simpleName}: ${t.message}")
                log("ERROR: ${t.javaClass.simpleName}: ${t.message}")
            } finally {
                cleanupAudio()
                running.set(false)
            }
        }
    }

    fun startReplyPlaybackTest() {
        if (!running.compareAndSet(false, true)) return
        thread(name = "AVTwin-C2-x20") {
            Process.setThreadPriority(Process.THREAD_PRIORITY_AUDIO)
            try {
                val source = prepareAudio()
                val prep = c2Player!!.preparationSnapshot()
                val report = StringBuilder()
                report.append("C2 REPEAT PLAYBACK TEST x$TEST_CYCLES\n")
                report.append("source=$source\n")
                report.append("C2=${c2Signal.name}\n")
                report.append(c2Signal.channelDiagnostics()).append('\n')
                report.append(preparationText(prep)).append("\n\n")
                var passCount = 0

                for (i in 1..TEST_CYCLES) {
                    if (!running.get()) return@thread
                    if (i > 1) c2Player!!.rearmForNextPlayback()
                    val verification = c2Player!!.playAndVerify()
                    if (verification.playbackVerified) passCount++
                    report.append("[$i/$TEST_CYCLES] ")
                    report.append(if (verification.playbackVerified) "PASS" else "FAIL")
                    report.append('\n')
                    report.append(verificationText(verification)).append("\n\n")
                    post(report.toString())
                }

                c2Player!!.awaitPlaybackCompletion()
                val overall = passCount == TEST_CYCLES
                report.append(
                    if (overall) {
                        "C2 PLAYBACK TEST: PASS $passCount/$TEST_CYCLES\nC2 HARDWARE PLAYBACK VERIFIED on every cycle"
                    } else {
                        "C2 PLAYBACK TEST: FAIL $passCount/$TEST_CYCLES\nOne or more cycles were not hardware-verified"
                    }
                )
                post(report.toString())
            } catch (t: Throwable) {
                post("C2 PLAYBACK TEST ERROR: ${t.javaClass.simpleName}: ${t.message}")
                log("C2 PLAYBACK TEST ERROR: ${t.message}")
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
        val prep = c2Player!!.preparationSnapshot()
        val accumulator = ShortAccumulator(maxOf(SAMPLE_RATE * 12, c1Full.size * 3))
        val hop = ShortArray(HOP_SAMPLES)
        var lastUiAt = 0L
        var lastC1Score = 0.0
        var lastBandRatio = 0.0
        var coarseT2 = -1

        post(
            "ARMED v0.7 - RESPONDER MODE\n" +
                "Goal: hear known C1 -> immediately return selected C2\n" +
                "48 kHz mono PCM16 | source=$sourceName\n" +
                "C1=${c1Signal.summary()}\n${c1Signal.channelDiagnostics()}\n" +
                "C2=${c2Signal.summary()}\n${c2Signal.channelDiagnostics()}\n" +
                "C1 gate=${if (c1Signal.isBuiltIn) "matched filter + HF/LF" else "matched filter against selected WAV"}\n" +
                preparationText(prep) + "\n" +
                "C2 prepared / AudioTrack initialized / buffer loaded\n" +
                "reply status=READY"
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

                val gateOk = if (c1Signal.isBuiltIn) {
                    lastBandRatio = if (score >= C1_PRETRIGGER) {
                        highToLowBandRatio(accumulator, candidate, c1Full.size)
                    } else 0.0
                    lastBandRatio >= C1_MIN_BAND_RATIO
                } else {
                    true
                }

                if (score >= C1_THRESHOLD && gateOk) coarseT2 = candidate
            }

            val now = System.currentTimeMillis()
            if (now - lastUiAt >= 250 && coarseT2 < 0) {
                lastUiAt = now
                val gateText = if (c1Signal.isBuiltIn) {
                    "HF/LF=${if (lastBandRatio > 0.0) fmt(lastBandRatio) else "--"} / ${fmt(C1_MIN_BAND_RATIO)}"
                } else {
                    "custom C1: waveform correlation only"
                }
                post(
                    "WAIT_C1\n" +
                        "template=${c1Signal.name}\n" +
                        "time=${fmt(accumulator.size.toDouble() / SAMPLE_RATE, 2)} s\n" +
                        "C1 score=${fmt(lastC1Score)} / ${fmt(C1_THRESHOLD)}\n" +
                        "$gateText\n" +
                        "C2 buffer=PRELOADED / reply status=READY"
                )
            }
        }

        if (!running.get() || coarseT2 < 0) return

        val preReplyAudio = accumulator.copy()
        val (t2, t2Score) = MatchedFilter.refineStrongest(
            preReplyAudio,
            c1Full,
            coarseT2,
            marginSamples = 96
        )
        log("C1 detected")

        val decisionNs = System.nanoTime()
        post(
            "C1 DETECTED\n" +
                "c1_detect_sample=$t2\n" +
                "score=${fmt(t2Score)}\n" +
                "decision_time_ns=$decisionNs\n" +
                "REPLYING PRELOADED C2 NOW..."
        )

        var playCallNsFromCallback = 0L
        val verification = c2Player!!.playAndVerify(onPlayIssued = { playNs ->
            playCallNsFromCallback = playNs
            val decisionToPlayUs = (playNs - decisionNs) / 1000.0
            val earlyJson = """{"type":"avtwin_android_reply","version":"0.6","status":"c2_play_issued","sample_rate":$SAMPLE_RATE,"timing_method":"software_play_call_only","t3_precise":false,"c1_template":"${escapeJson(c1Signal.name)}","c2_template":"${escapeJson(c2Signal.name)}","t2_sample":$t2,"t2_score":${fmt(t2Score, 5)},"reply_play_call_ns":$playNs,"decision_to_play_us":${fmt(decisionToPlayUs, 3)}}"""
            try { UdpReporter.send(linuxHost, linuxPort, earlyJson) } catch (_: Throwable) {}
        })

        val playCallNs = if (playCallNsFromCallback != 0L) playCallNsFromCallback else verification.playCallTimeNs
        val softwareDecisionToPlayUs = (playCallNs - decisionNs) / 1000.0
        val playbackStatus = if (verification.playbackVerified) {
            "C2 HARDWARE PLAYBACK VERIFIED"
        } else {
            "PLAY() CALLED — HARDWARE PLAYBACK NOT VERIFIED"
        }

        post(
            "$playbackStatus\n" +
                "c1_detect_sample=$t2\n" +
                "decision_time_ns=$decisionNs\n" +
                "play_call_time_ns=$playCallNs\n" +
                "first_valid_audio_timestamp_ns=${verification.firstValidAudioTimestampNs ?: "NONE"}\n" +
                "first_valid_audio_frame_position=${verification.firstValidAudioFramePosition ?: "NONE"}\n" +
                "playback_verified=${verification.playbackVerified}\n" +
                verificationText(verification) + "\n" +
                "t3_precise=false"
        )

        val tailSamples = maxOf((0.55 * SAMPLE_RATE).toInt(), c2Full.size + (0.25 * SAMPLE_RATE).toInt())
        val tailTarget = accumulator.size + tailSamples
        while (running.get() && accumulator.size < tailTarget) {
            val n = record!!.read(hop, 0, hop.size, AudioRecord.READ_BLOCKING)
            if (n > 0) accumulator.append(hop, n)
        }
        try { record!!.stop() } catch (_: Throwable) {}
        c2Player!!.awaitPlaybackCompletion()

        val wav = saveWav(accumulator.copy(), "responder_v07")
        val finalJson = """{"type":"avtwin_android_reply","version":"0.6","status":"done","sample_rate":$SAMPLE_RATE,"timing_method":"software_play_call_only","t3_precise":false,"c1_template":"${escapeJson(c1Signal.name)}","c2_template":"${escapeJson(c2Signal.name)}","t2_sample":$t2,"t2_score":${fmt(t2Score, 5)},"reply_play_call_ns":$playCallNs,"decision_to_play_us":${fmt(softwareDecisionToPlayUs, 3)},"playback_verified":${verification.playbackVerified},"first_valid_audio_timestamp_ns":${verification.firstValidAudioTimestampNs ?: "null"},"first_valid_audio_frame_position":${verification.firstValidAudioFramePosition ?: "null"},"playback_head_before":${verification.playbackHeadBefore},"playback_head_after":${verification.playbackHeadAfter},"wav":"${escapeJson(wav.absolutePath)}"}"""
        try { UdpReporter.send(linuxHost, linuxPort, finalJson) } catch (_: Throwable) {}

        post(
            "DONE v0.7\n" +
                "C1 detected ✓ (${c1Signal.name})\n" +
                "$playbackStatus\n" +
                "t2=$t2 score=${fmt(t2Score)}\n" +
                "decision->play=${fmt(softwareDecisionToPlayUs)} us\n" +
                "t3_precise=false\n" +
                "UDP target=$linuxHost:$linuxPort\n" +
                "WAV=${wav.absolutePath}"
        )

        onResult(
            Result(
                t2Sample = t2,
                t2Score = t2Score,
                replyPlayCallNs = playCallNs,
                softwareDecisionToPlayUs = softwareDecisionToPlayUs,
                playbackVerified = verification.playbackVerified,
                firstValidAudioTimestampNs = verification.firstValidAudioTimestampNs,
                firstValidAudioFramePosition = verification.firstValidAudioFramePosition,
                wavPath = wav.absolutePath,
                udpJson = finalJson
            )
        )
    }

    private fun prepareAudio(): String {
        val audioManager = context.getSystemService(Context.AUDIO_SERVICE) as AudioManager
        val unprocessed = audioManager.getProperty(AudioManager.PROPERTY_SUPPORT_AUDIO_SOURCE_UNPROCESSED) == "true"
        val source = if (unprocessed) MediaRecorder.AudioSource.UNPROCESSED else MediaRecorder.AudioSource.VOICE_RECOGNITION
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

        c2Player = PersistentC2Player(c2Full, SAMPLE_RATE, ::log)
        c2Player!!.prepare()
        return sourceName
    }

    private fun preparationText(p: PersistentC2Player.Preparation): String =
        "AudioTrack state=${trackStateName(p.trackState)} (${p.trackState})\n" +
            "playState=${playStateName(p.playState)} (${p.playState})\n" +
            "channels=${p.channelCount}\n" +
            "performanceMode=${performanceModeName(p.performanceMode)} (${p.performanceMode})\n" +
            "write=${p.writtenSamples} samples / ${p.writtenFrames} frames / ${p.writtenBytes} bytes\n" +
            "playbackHeadPosition=${p.playbackHeadPosition}"

    private fun verificationText(v: PersistentC2Player.PlaybackVerification): String =
        "AudioTrack state=${trackStateName(v.trackState)} (${v.trackState})\n" +
            "playState=${playStateName(v.playState)} (${v.playState})\n" +
            "write=${v.writeSamples} samples / ${v.writeFrames} frames / ${v.writeBytes} bytes\n" +
            "playbackHeadPosition=${v.playbackHeadBefore} -> ${v.playbackHeadAfter}\n" +
            "playbackHeadAdvanced=${v.playbackHeadAdvanced}\n" +
            "AudioTimestamp valid=${v.audioTimestampValid}\n" +
            "timestamp framePosition=${v.firstValidAudioFramePosition ?: "NONE"}\n" +
            "timestamp nanoTime=${v.firstValidAudioTimestampNs ?: "NONE"}"

    private fun trackStateName(state: Int): String = when (state) {
        AudioTrack.STATE_INITIALIZED -> "INITIALIZED"
        AudioTrack.STATE_UNINITIALIZED -> "UNINITIALIZED"
        else -> "UNKNOWN"
    }

    private fun playStateName(state: Int): String = when (state) {
        AudioTrack.PLAYSTATE_STOPPED -> "STOPPED"
        AudioTrack.PLAYSTATE_PAUSED -> "PAUSED"
        AudioTrack.PLAYSTATE_PLAYING -> "PLAYING"
        else -> "UNKNOWN"
    }

    private fun performanceModeName(mode: Int): String = when (mode) {
        AudioTrack.PERFORMANCE_MODE_LOW_LATENCY -> "LOW_LATENCY"
        AudioTrack.PERFORMANCE_MODE_POWER_SAVING -> "POWER_SAVING"
        AudioTrack.PERFORMANCE_MODE_NONE -> "NONE"
        else -> "UNKNOWN"
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
        try { c2Player?.release() } catch (_: Throwable) {}
        try { aec?.release() } catch (_: Throwable) {}
        try { ns?.release() } catch (_: Throwable) {}
        try { agc?.release() } catch (_: Throwable) {}
        record = null
        c2Player = null
        aec = null
        ns = null
        agc = null
    }

    private fun escapeJson(value: String): String = value
        .replace("\\", "\\\\")
        .replace("\"", "\\\"")
        .replace("\n", "\\n")
        .replace("\r", "\\r")

    private fun fmt(v: Double, digits: Int = 3): String = "% .${digits}f".format(Locale.US, v).trim()
    private fun post(s: String) = onStatus(s)
    private fun log(s: String) = Log.i(TAG, s)
}
