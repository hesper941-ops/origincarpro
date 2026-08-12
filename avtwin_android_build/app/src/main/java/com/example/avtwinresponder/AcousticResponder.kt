package com.example.avtwinresponder

import android.content.Context
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioRecord
import android.media.AudioTimestamp
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
import kotlin.math.log10
import kotlin.math.sqrt

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

        const val C1_PRETRIGGER = 0.18
        const val C1_THRESHOLD = 0.28
        const val C1_MIN_BAND_RATIO = 0.80
        const val C2_DIAGNOSTIC_THRESHOLD = 0.22
    }

    data class Result(
        val t2Sample: Int,
        val t2MonotonicNs: Long,
        val t3MonotonicNs: Long,
        val t3EquivalentSample: Int,
        val replyDelayNs: Long,
        val replyDelayMs: Double,
        val t2Score: Double,
        val c2SelfScore: Double,
        val wavPath: String,
        val udpJson: String
    )

    private data class BandResult(
        val f0: Int,
        val f1: Int,
        val score: Double,
        val gainDb: Double
    )

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
                runHandshake(linuxHost, linuxPort)
            } catch (t: Throwable) {
                post("ERROR: ${t.javaClass.simpleName}: ${t.message}")
            } finally {
                cleanupAudio()
                running.set(false)
            }
        }
    }

    fun startC2TimingTest() {
        if (!running.compareAndSet(false, true)) return
        thread(name = "AVTwin-C2-TimingTest") {
            Process.setThreadPriority(Process.THREAD_PRIORITY_AUDIO)
            try {
                runC2TimingTest()
            } catch (t: Throwable) {
                post("TIMING TEST ERROR: ${t.javaClass.simpleName}: ${t.message}")
            } finally {
                cleanupAudio()
                running.set(false)
            }
        }
    }

    fun startBandDiagnostic() {
        if (!running.compareAndSet(false, true)) return
        thread(name = "AVTwin-BandDiag") {
            Process.setThreadPriority(Process.THREAD_PRIORITY_AUDIO)
            try {
                runBandDiagnostic()
            } catch (t: Throwable) {
                post("BAND TEST ERROR: ${t.javaClass.simpleName}: ${t.message}")
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

    private fun runHandshake(linuxHost: String, linuxPort: Int) {
        val sourceName = prepareInputAndOutput()
        val accumulator = ShortAccumulator(SAMPLE_RATE * 12)
        val hop = ShortArray(HOP_SAMPLES)
        var lastUiAt = 0L
        var lastC1Score = 0.0
        var lastBandRatio = 0.0
        var coarseT2 = -1

        post(
            "ARMED v0.4\n" +
                "48 kHz mono PCM16 | source=$sourceName\n" +
                "output=${outputDescription()}\n" +
                "C1=11-19 kHz / 200 ms\n" +
                "t3 method=AudioTrack hardware timestamp\n" +
                "waiting for Linux C1..."
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
                } else 0.0

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
                        "t3=hardware timestamp"
                )
            }
        }

        if (!running.get() || coarseT2 < 0) return

        val preReplyAudio = accumulator.copy()
        val (t2, t2Score) = MatchedFilter.refineStrongest(preReplyAudio, c1Full, coarseT2, marginSamples = 96)
        val t2Ns = recordingFrameToMonotonicNs(t2.toLong())

        post(
            "C1 DETECTED\n" +
                "t2 sample=$t2 score=${fmt(t2Score)}\n" +
                "t2 mono=$t2Ns ns\n" +
                "queueing C2 and waiting for AudioTrack timestamp..."
        )

        val t3Ns = playSamplesWithHardwareTimestamp(c2Full)
        val replyDelayNs = t3Ns - t2Ns
        require(replyDelayNs > 0L) { "Audio timestamps produced non-positive t3-t2: $replyDelayNs ns" }
        require(replyDelayNs < 2_000_000_000L) { "Audio timestamps produced implausible t3-t2: $replyDelayNs ns" }
        val replyDelayMs = replyDelayNs / 1_000_000.0
        val t3EquivalentSample = t2 + ((replyDelayNs * SAMPLE_RATE) / 1_000_000_000L).toInt()

        post(
            "C2 PLAYBACK TIMESTAMP OK\n" +
                "t3 mono=$t3Ns ns\n" +
                "t3-t2=${fmt(replyDelayMs)} ms\n" +
                "capturing 450 ms diagnostic tail..."
        )

        val tailEnd = accumulator.size + (0.45 * SAMPLE_RATE).toInt()
        while (running.get() && accumulator.size < tailEnd) {
            val n = record!!.read(hop, 0, hop.size, AudioRecord.READ_BLOCKING)
            if (n > 0) accumulator.append(hop, n)
        }
        try { record!!.stop() } catch (_: Throwable) {}
        if (!running.get()) return

        val audio = accumulator.copy()
        val c2Self = bestScoreAfter(audio, c2Full, (t2 + (0.030 * SAMPLE_RATE).toInt()).coerceAtLeast(0))
        val wav = saveWav(audio, "handshake_v04")

        val json = """{"type":"avtwin_android_reply","version":"0.4","sample_rate":$SAMPLE_RATE,"timing_method":"audio_hardware_timestamps","t2_sample":$t2,"t2_monotonic_ns":$t2Ns,"t3_monotonic_ns":$t3Ns,"t3_equivalent_sample":$t3EquivalentSample,"reply_delay_ns":$replyDelayNs,"reply_delay_ms":${fmt(replyDelayMs, 6)},"t2_score":${fmt(t2Score, 5)},"c2_self_score":${fmt(c2Self, 5)}}"""

        try {
            UdpReporter.send(linuxHost, linuxPort, json)
            post(
                "DONE v0.4\n" +
                    "t2=$t2 score=${fmt(t2Score)}\n" +
                    "t3-t2=${fmt(replyDelayMs)} ms\n" +
                    "C2 self score=${fmt(c2Self)} (diagnostic only)\n" +
                    "UDP -> $linuxHost:$linuxPort\n" +
                    "WAV=${wav.absolutePath}"
            )
        } catch (t: Throwable) {
            post(
                "DONE v0.4 (UDP failed: ${t.message})\n" +
                    "t3-t2=${fmt(replyDelayMs)} ms\n" +
                    "WAV=${wav.absolutePath}"
            )
        }

        onResult(
            Result(
                t2Sample = t2,
                t2MonotonicNs = t2Ns,
                t3MonotonicNs = t3Ns,
                t3EquivalentSample = t3EquivalentSample,
                replyDelayNs = replyDelayNs,
                replyDelayMs = replyDelayMs,
                t2Score = t2Score,
                c2SelfScore = c2Self,
                wavPath = wav.absolutePath,
                udpJson = json
            )
        )
    }

    private fun runC2TimingTest() {
        val sourceName = prepareInputAndOutput()
        val accumulator = ShortAccumulator(SAMPLE_RATE * 2)
        val hop = ShortArray(HOP_SAMPLES)
        val preRoll = (0.25 * SAMPLE_RATE).toInt()

        post(
            "C2 TIMING TEST START\n" +
                "source=$sourceName\n" +
                "output=${outputDescription()}\n" +
                "goal: verify AudioTrack hardware timestamp\n" +
                "self-detection is diagnostic only"
        )

        record!!.startRecording()
        while (running.get() && accumulator.size < preRoll) {
            val n = record!!.read(hop, 0, hop.size, AudioRecord.READ_BLOCKING)
            if (n > 0) accumulator.append(hop, n)
        }
        if (!running.get()) return

        val t3Ns = playSamplesWithHardwareTimestamp(c2Full)
        val target = accumulator.size + (0.65 * SAMPLE_RATE).toInt()
        while (running.get() && accumulator.size < target) {
            val n = record!!.read(hop, 0, hop.size, AudioRecord.READ_BLOCKING)
            if (n > 0) accumulator.append(hop, n)
        }
        try { record!!.stop() } catch (_: Throwable) {}
        if (!running.get()) return

        val audio = accumulator.copy()
        val score = bestScoreAfter(audio, c2Full, (0.15 * SAMPLE_RATE).toInt())
        val wav = saveWav(audio, "c2_timestamp_test")

        post(
            "C2 TIMESTAMP TEST: PASS\n" +
                "AudioTrack start=$t3Ns ns MONOTONIC\n" +
                "C2 self-detect=${if (score >= C2_DIAGNOSTIC_THRESHOLD) "PASS" else "FAIL"} score=${fmt(score)} / ${fmt(C2_DIAGNOSTIC_THRESHOLD)}\n" +
                "NOTE: self-detect does NOT gate v0.4 handshake\n" +
                "WAV=${wav.absolutePath}"
        )
    }

    private fun runBandDiagnostic() {
        val bands = arrayOf(
            1_000.0 to 5_000.0,
            5_000.0 to 9_000.0,
            9_000.0 to 13_000.0,
            13_000.0 to 17_000.0
        )
        val results = ArrayList<BandResult>()

        for ((index, band) in bands.withIndex()) {
            if (!running.get()) return
            cleanupAudio()
            val sourceName = prepareInputAndOutput()
            val f0 = band.first
            val f1 = band.second
            val template = Chirp.linearPcm16(SAMPLE_RATE, 0.150, f0, f1)
            val acc = ShortAccumulator(SAMPLE_RATE * 2)
            val hop = ShortArray(HOP_SAMPLES)
            val preRoll = (0.20 * SAMPLE_RATE).toInt()

            post(
                "BAND DIAGNOSTIC ${index + 1}/4\n" +
                    "source=$sourceName | output=${outputDescription()}\n" +
                    "testing ${f0.toInt()}-${f1.toInt()} Hz..."
            )

            record!!.startRecording()
            while (running.get() && acc.size < preRoll) {
                val n = record!!.read(hop, 0, hop.size, AudioRecord.READ_BLOCKING)
                if (n > 0) acc.append(hop, n)
            }
            if (!running.get()) return

            playSamplesNoTimestamp(template)
            val target = acc.size + (0.60 * SAMPLE_RATE).toInt()
            while (running.get() && acc.size < target) {
                val n = record!!.read(hop, 0, hop.size, AudioRecord.READ_BLOCKING)
                if (n > 0) acc.append(hop, n)
            }
            try { record!!.stop() } catch (_: Throwable) {}

            val audio = acc.copy()
            val score = bestScoreAfter(audio, template, (0.10 * SAMPLE_RATE).toInt())
            val noiseRms = rms(audio, 0, (0.15 * SAMPLE_RATE).toInt().coerceAtMost(audio.size))
            val maxRms = maxWindowRms(audio, preRoll, (0.025 * SAMPLE_RATE).toInt())
            val gainDb = 20.0 * log10((maxRms + 1.0) / (noiseRms + 1.0))
            results.add(BandResult(f0.toInt(), f1.toInt(), score, gainDb))
        }

        val best = results.maxByOrNull { it.score }
        val lines = results.joinToString("\n") {
            "${it.f0 / 1000}-${it.f1 / 1000} kHz: corr=${fmt(it.score)}  level=+${fmt(it.gainDb, 1)} dB"
        }
        post(
            "BAND DIAGNOSTIC DONE\n" + lines +
                "\nBest correlation: ${best?.f0?.div(1000)}-${best?.f1?.div(1000)} kHz\n" +
                "Use this to judge whether 11-19 kHz C1 is realistic on this tablet."
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
        prepareTrack()
        return sourceName
    }

    private fun prepareTrack() {
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

    private fun playSamplesWithHardwareTimestamp(samples: ShortArray): Long {
        val t = track ?: error("AudioTrack not prepared")
        queueSamples(t, samples)
        t.play()

        val ts = AudioTimestamp()
        val deadline = System.nanoTime() + 800_000_000L
        while (System.nanoTime() < deadline && running.get()) {
            if (t.getTimestamp(ts) && ts.framePosition > 0L) {
                return ts.nanoTime - framesToNs(ts.framePosition)
            }
            Thread.sleep(4)
        }
        throw IllegalStateException("AudioTrack hardware timestamp unavailable within 800 ms")
    }

    private fun playSamplesNoTimestamp(samples: ShortArray) {
        val t = track ?: error("AudioTrack not prepared")
        queueSamples(t, samples)
        t.play()
    }

    private fun queueSamples(t: AudioTrack, samples: ShortArray) {
        try { t.pause() } catch (_: Throwable) {}
        try { t.flush() } catch (_: Throwable) {}

        if (t.channelCount == 1) {
            val written = t.write(samples, 0, samples.size, AudioTrack.WRITE_BLOCKING)
            require(written == samples.size) { "Could not queue mono audio: $written/${samples.size}" }
        } else {
            val stereo = ShortArray(samples.size * 2)
            for (i in samples.indices) {
                stereo[2 * i] = samples[i]
                stereo[2 * i + 1] = samples[i]
            }
            val written = t.write(stereo, 0, stereo.size, AudioTrack.WRITE_BLOCKING)
            require(written == stereo.size) { "Could not queue stereo audio: $written/${stereo.size}" }
        }
    }

    private fun recordingFrameToMonotonicNs(frame: Long): Long {
        val r = record ?: error("AudioRecord not prepared")
        val ts = AudioTimestamp()
        var attempts = 0
        while (attempts < 25 && running.get()) {
            val rc = r.getTimestamp(ts, AudioTimestamp.TIMEBASE_MONOTONIC)
            if (rc == AudioRecord.SUCCESS && ts.framePosition > 0L) {
                return ts.nanoTime + framesToNs(frame - ts.framePosition)
            }
            attempts++
            Thread.sleep(2)
        }
        throw IllegalStateException("AudioRecord hardware timestamp unavailable")
    }

    private fun framesToNs(frames: Long): Long = (frames * 1_000_000_000L) / SAMPLE_RATE

    private fun bestScoreAfter(audio: ShortArray, template: ShortArray, firstStart: Int): Double {
        if (audio.size < template.size) return 0.0
        val first = firstStart.coerceIn(0, audio.size - template.size)
        val last = audio.size - template.size
        var best = 0.0
        var s = first
        while (s <= last) {
            val score = MatchedFilter.score(audio, template, s, decimation = 6)
            if (score > best) best = score
            s += 8
        }
        return best
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

    private fun rms(audio: ShortArray, start: Int, length: Int): Double {
        if (length <= 0 || start < 0 || start + length > audio.size) return 0.0
        var sum = 0.0
        for (i in start until start + length) {
            val x = audio[i].toDouble()
            sum += x * x
        }
        return sqrt(sum / length)
    }

    private fun maxWindowRms(audio: ShortArray, start: Int, window: Int): Double {
        if (audio.isEmpty() || window <= 0) return 0.0
        var best = 0.0
        var s = start.coerceAtLeast(0)
        val step = (window / 2).coerceAtLeast(1)
        while (s + window <= audio.size) {
            val value = rms(audio, s, window)
            if (value > best) best = value
            s += step
        }
        return best
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
    }

    private fun fmt(v: Double, digits: Int = 3): String = "% .${digits}f".format(Locale.US, v).trim()
    private fun post(s: String) = onStatus(s)
}
