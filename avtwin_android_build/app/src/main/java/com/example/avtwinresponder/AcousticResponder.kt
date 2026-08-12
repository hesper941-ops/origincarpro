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

        // v0.3: detect the complete 200 ms C1 instead of only its first 50 ms.
        // A full-length normalized correlation is much less likely to false-trigger.
        const val C1_PRETRIGGER = 0.18
        const val C1_THRESHOLD = 0.28
        const val C1_MIN_BAND_RATIO = 0.80
        const val C2_THRESHOLD = 0.22
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

    fun startC2SelfTest() {
        if (!running.compareAndSet(false, true)) return
        thread(name = "AVTwin-C2-SelfTest") {
            Process.setThreadPriority(Process.THREAD_PRIORITY_AUDIO)
            try {
                runC2SelfTest()
            } catch (t: Throwable) {
                post("SELF TEST ERROR: ${t.javaClass.simpleName}: ${t.message}")
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
        val sourceName = prepareInputAndOutput()
        val accumulator = ShortAccumulator(SAMPLE_RATE * 12)
        val hop = ShortArray(HOP_SAMPLES)
        var state = State.WAIT_C1
        var coarseT2 = -1
        var coarseT3 = -1
        var lastUiAt = 0L
        var lastC1Score = 0.0
        var lastBandRatio = 0.0
        var t2CoarseScore = 0.0
        var t3CoarseScore = 0.0

        post(
            "ARMED v0.3\n" +
                "48 kHz mono PCM16 | source=$sourceName\n" +
                "output=${outputDescription()}\n" +
                "C1 full chirp=200 ms, ${C1_F0.toInt()}-${C1_F1.toInt()} Hz\n" +
                "threshold=${fmt(C1_THRESHOLD)}, band gate>=${fmt(C1_MIN_BAND_RATIO)}"
        )

        record!!.startRecording()

        while (running.get() && state != State.DONE) {
            val n = record!!.read(hop, 0, hop.size, AudioRecord.READ_BLOCKING)
            if (n <= 0) continue
            accumulator.append(hop, n)

            when (state) {
                State.WAIT_C1 -> {
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

                        if (score >= C1_PRETRIGGER) {
                            lastBandRatio = highToLowBandRatio(accumulator, candidate, c1Full.size)
                        } else {
                            lastBandRatio = 0.0
                        }

                        if (score >= C1_THRESHOLD && lastBandRatio >= C1_MIN_BAND_RATIO) {
                            coarseT2 = candidate
                            t2CoarseScore = score
                            state = State.WAIT_C2
                            post(
                                "C1 DETECTED\n" +
                                    "coarse t2=$coarseT2\n" +
                                    "full-score=${fmt(score)}\n" +
                                    "HF/LF=${fmt(lastBandRatio)}\n" +
                                    "playing C2 now..."
                            )
                            playC2()
                        }
                    }
                }

                State.WAIT_C2 -> {
                    if (accumulator.size >= c2Full.size) {
                        val newest = accumulator.size - c2Full.size
                        val (candidate, score) = MatchedFilter.bestRecent(
                            accumulator,
                            c2Full,
                            newest,
                            HOP_SAMPLES * 3,
                            startStep = 4,
                            decimation = 6
                        )
                        val safelyAfterC1 = candidate > coarseT2 + (0.050 * SAMPLE_RATE).toInt()
                        if (safelyAfterC1 && score >= C2_THRESHOLD) {
                            coarseT3 = candidate
                            t3CoarseScore = score
                            state = State.DONE
                            post(
                                "C2 SELF-DETECTED\n" +
                                    "coarse t3=$coarseT3\n" +
                                    "score=${fmt(score)}\n" +
                                    "refining sample indices..."
                            )
                        }
                    }
                }

                State.DONE -> Unit
            }

            val now = System.currentTimeMillis()
            if (now - lastUiAt >= 250) {
                lastUiAt = now
                if (state == State.WAIT_C1) {
                    post(
                        "WAIT_C1\n" +
                            "time=${fmt(accumulator.size.toDouble() / SAMPLE_RATE, 2)} s\n" +
                            "C1 score=${fmt(lastC1Score)} / ${fmt(C1_THRESHOLD)}\n" +
                            "HF/LF=${if (lastBandRatio > 0.0) fmt(lastBandRatio) else "--"} / ${fmt(C1_MIN_BAND_RATIO)}\n" +
                            "full chirp check=200 ms"
                    )
                } else if (state == State.WAIT_C2) {
                    post(
                        "WAIT_C2\n" +
                            "time=${fmt(accumulator.size.toDouble() / SAMPLE_RATE, 2)} s\n" +
                            "t2 coarse=$coarseT2 score=${fmt(t2CoarseScore)}\n" +
                            "waiting for self C2..."
                    )
                }
            }
        }

        try { record!!.stop() } catch (_: Throwable) {}
        if (!running.get() || coarseT2 < 0 || coarseT3 < 0) return

        val audio = accumulator.copy()
        val (t2, t2Score) = MatchedFilter.refineStrongest(audio, c1Full, coarseT2, marginSamples = 96)
        val (t3, t3Score) = MatchedFilter.refineStrongest(audio, c2Full, coarseT3, marginSamples = 96)
        val delta = t3 - t2
        val deltaMs = delta * 1000.0 / SAMPLE_RATE
        val wav = saveWav(audio, "handshake")

        val json = """{"type":"avtwin_android_reply","version":"0.3","sample_rate":$SAMPLE_RATE,"t2_sample":$t2,"t3_sample":$t3,"reply_delay_samples":$delta,"reply_delay_ms":${fmt(deltaMs, 6)},"t2_score":${fmt(t2Score, 5)},"t3_score":${fmt(t3Score, 5)},"t2_coarse_score":${fmt(t2CoarseScore, 5)},"t3_coarse_score":${fmt(t3CoarseScore, 5)}}"""

        thread(name = "AVTwin-UDP") {
            try {
                UdpReporter.send(linuxHost, linuxPort, json)
                post(
                    "DONE\n" +
                        "T2=$t2 score=${fmt(t2Score)}\n" +
                        "T3=$t3 score=${fmt(t3Score)}\n" +
                        "Delta=${fmt(deltaMs)} ms\n" +
                        "UDP -> $linuxHost:$linuxPort\n" +
                        "WAV=${wav.absolutePath}"
                )
            } catch (t: Throwable) {
                post(
                    "DONE (UDP failed: ${t.message})\n" +
                        "T2=$t2\nT3=$t3\nDelta=${fmt(deltaMs)} ms\n" +
                        "WAV=${wav.absolutePath}"
                )
            }
        }

        onResult(Result(t2, t3, delta, deltaMs, t2Score, t3Score, wav.absolutePath, json))
    }

    private fun runC2SelfTest() {
        val sourceName = prepareInputAndOutput()
        val accumulator = ShortAccumulator(SAMPLE_RATE * 3)
        val hop = ShortArray(HOP_SAMPLES)
        val preRollSamples = (0.25 * SAMPLE_RATE).toInt()
        val totalSamples = (1.30 * SAMPLE_RATE).toInt()
        var played = false

        post(
            "SELF TEST START\n" +
                "source=$sourceName\n" +
                "output=${outputDescription()}\n" +
                "record 250 ms -> play C2 -> self-detect"
        )

        record!!.startRecording()
        while (running.get() && accumulator.size < totalSamples) {
            val n = record!!.read(hop, 0, hop.size, AudioRecord.READ_BLOCKING)
            if (n <= 0) continue
            accumulator.append(hop, n)

            if (!played && accumulator.size >= preRollSamples) {
                played = true
                post("SELF TEST: playing C2 ${C2_F0.toInt()}-${C2_F1.toInt()} Hz for 200 ms...")
                playC2()
            }
        }
        try { record!!.stop() } catch (_: Throwable) {}
        if (!running.get()) return

        val audio = accumulator.copy()
        val first = (0.15 * SAMPLE_RATE).toInt()
        val last = (audio.size - c2Full.size).coerceAtLeast(first)
        var bestStart = first
        var bestScore = -1.0
        var s = first
        while (s <= last) {
            val score = MatchedFilter.score(audio, c2Full, s, decimation = 6)
            if (score > bestScore) {
                bestScore = score
                bestStart = s
            }
            s += 8
        }
        val (fineStart, fineScore) = MatchedFilter.refineStrongest(audio, c2Full, bestStart, marginSamples = 64)
        val wav = saveWav(audio, "c2_selftest")
        val pass = fineScore >= C2_THRESHOLD

        post(
            "SELF TEST DONE: ${if (pass) "PASS" else "FAIL"}\n" +
                "C2 sample=$fineStart\n" +
                "C2 score=${fmt(fineScore)} / ${fmt(C2_THRESHOLD)}\n" +
                "output=${outputDescription()}\n" +
                "WAV=${wav.absolutePath}"
        )
    }

    private fun prepareInputAndOutput(): String {
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
        prepareC2Track()
        return sourceName
    }

    private fun prepareC2Track() {
        fun makeTrack(mask: Int): AudioTrack {
            val minOut = AudioTrack.getMinBufferSize(SAMPLE_RATE, mask, AudioFormat.ENCODING_PCM_16BIT)
            require(minOut > 0) { "48 kHz PCM16 AudioTrack not supported for channel mask $mask: $minOut" }
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
                .setBufferSizeInBytes(maxOf(minOut * 2, c2Full.size * 4))
                .setTransferMode(AudioTrack.MODE_STREAM)
                .build()
        }

        track = try {
            makeTrack(AudioFormat.CHANNEL_OUT_MONO)
        } catch (_: Throwable) {
            makeTrack(AudioFormat.CHANNEL_OUT_STEREO)
        }
        require(track?.state == AudioTrack.STATE_INITIALIZED) { "AudioTrack streaming initialization failed" }
        track!!.setVolume(1.0f)
    }

    private fun playC2() {
        val t = track ?: return
        try { t.pause() } catch (_: Throwable) {}
        try { t.flush() } catch (_: Throwable) {}
        t.play()

        if (t.channelCount == 1) {
            val written = t.write(c2Full, 0, c2Full.size, AudioTrack.WRITE_BLOCKING)
            require(written == c2Full.size) { "Could not stream mono C2: $written/${c2Full.size}" }
        } else {
            val stereo = ShortArray(c2Full.size * 2)
            for (i in c2Full.indices) {
                stereo[2 * i] = c2Full[i]
                stereo[2 * i + 1] = c2Full[i]
            }
            val written = t.write(stereo, 0, stereo.size, AudioTrack.WRITE_BLOCKING)
            require(written == stereo.size) { "Could not stream stereo C2: $written/${stereo.size}" }
        }
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

    private fun fmt(v: Double, digits: Int = 3): String = "% .${digits}f".format(Locale.US, v).trim()
    private fun post(s: String) = onStatus(s)
}
