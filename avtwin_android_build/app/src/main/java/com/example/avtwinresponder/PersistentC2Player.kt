package com.example.avtwinresponder

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTimestamp
import android.media.AudioTrack

internal class PersistentC2Player(
    private val samples: ShortArray,
    private val sampleRate: Int = ProbeSignal.SAMPLE_RATE,
    private val onLog: (String) -> Unit = {}
) {
    data class Preparation(
        val trackState: Int,
        val playState: Int,
        val channelCount: Int,
        val performanceMode: Int,
        val writtenSamples: Int,
        val writtenFrames: Int,
        val writtenBytes: Int,
        val playbackHeadPosition: Long
    )

    data class PlaybackVerification(
        val playCallTimeNs: Long,
        val trackState: Int,
        val playState: Int,
        val writeSamples: Int,
        val writeFrames: Int,
        val writeBytes: Int,
        val playbackHeadBefore: Long,
        val playbackHeadAfter: Long,
        val playbackHeadAdvanced: Boolean,
        val audioTimestampValid: Boolean,
        val firstValidAudioTimestampNs: Long?,
        val firstValidAudioFramePosition: Long?,
        val playbackVerified: Boolean
    )

    private val stateMachine = C2ReplayStateMachine()
    private var track: AudioTrack? = null
    private var outputSamples = ShortArray(0)
    private var outputFrames = 0
    private var lastWrittenSamples = 0
    private var lastWrittenFrames = 0
    private var lastWrittenBytes = 0
    private var lastPlayCallNs = 0L

    fun prepare(): Preparation {
        require(samples.isNotEmpty()) { "C2 PCM is empty" }
        onLog("C2 prepared")
        track = createPersistentTrack()
        val t = track ?: error("AudioTrack creation failed")
        stateMachine.trackInitialized()
        onLog("C2 AudioTrack initialized")

        outputSamples = if (t.channelCount == 1) {
            samples.copyOf()
        } else {
            ShortArray(samples.size * 2).also { stereo ->
                // Internal C2 is mono (RIGHT channel extracted from source WAV).
                // Duplicate it only when the device requires a stereo AudioTrack.
                for (i in samples.indices) {
                    stereo[2 * i] = samples[i]
                    stereo[2 * i + 1] = samples[i]
                }
            }
        }
        outputFrames = outputSamples.size / t.channelCount.coerceAtLeast(1)
        preloadInternal(t, initial = true)
        return preparationSnapshot()
    }

    fun playAndVerify(
        timeoutMs: Long = 800L,
        onPlayIssued: ((Long) -> Unit)? = null
    ): PlaybackVerification {
        val t = track ?: error("C2 AudioTrack is not prepared")
        require(stateMachine.state == C2ReplayStateMachine.State.BUFFER_LOADED) {
            "C2 buffer is not ready: ${stateMachine.state}"
        }

        val headBefore = unsignedHead(t.playbackHeadPosition)
        val baselineTimestamp = AudioTimestamp()
        val baselineTimestampValid = try { t.getTimestamp(baselineTimestamp) } catch (_: Throwable) { false }
        val baselineFrame = if (baselineTimestampValid) baselineTimestamp.framePosition else -1L

        lastPlayCallNs = System.nanoTime()
        t.play()
        stateMachine.playIssued()
        onLog("C2 play() issued")
        onPlayIssued?.invoke(lastPlayCallNs)

        val deadline = System.nanoTime() + timeoutMs * 1_000_000L
        var firstHeadAdvanced = false
        var timestampValid = false
        var firstTimestampNs: Long? = null
        var firstTimestampFrame: Long? = null
        var lastHead = headBefore

        while (System.nanoTime() < deadline) {
            lastHead = unsignedHead(t.playbackHeadPosition)
            val headDelta = unsignedDelta(headBefore, lastHead)
            if (!firstHeadAdvanced && headDelta > 0L) {
                firstHeadAdvanced = true
                onLog("C2 playback head advanced")
            }

            val ts = AudioTimestamp()
            val ok = try { t.getTimestamp(ts) } catch (_: Throwable) { false }
            val timestampAdvanced = ok && ts.framePosition >= 0L &&
                (!baselineTimestampValid || ts.framePosition > baselineFrame)
            if (timestampAdvanced && !timestampValid) {
                timestampValid = true
                firstTimestampNs = ts.nanoTime
                firstTimestampFrame = ts.framePosition
            }

            if (firstHeadAdvanced || timestampValid) break
            Thread.sleep(2)
        }

        val verified = firstHeadAdvanced || timestampValid
        stateMachine.verificationFinished(verified)
        if (verified) onLog("C2 hardware playback verified")

        return PlaybackVerification(
            playCallTimeNs = lastPlayCallNs,
            trackState = t.state,
            playState = t.playState,
            writeSamples = lastWrittenSamples,
            writeFrames = lastWrittenFrames,
            writeBytes = lastWrittenBytes,
            playbackHeadBefore = headBefore,
            playbackHeadAfter = lastHead,
            playbackHeadAdvanced = firstHeadAdvanced,
            audioTimestampValid = timestampValid,
            firstValidAudioTimestampNs = firstTimestampNs,
            firstValidAudioFramePosition = firstTimestampFrame,
            playbackVerified = verified
        )
    }

    fun awaitPlaybackCompletion() {
        waitUntilPlaybackWindowElapsed()
    }

    fun rearmForNextPlayback() {
        val t = track ?: error("C2 AudioTrack is not prepared")
        require(
            stateMachine.state == C2ReplayStateMachine.State.PLAYBACK_VERIFIED ||
                stateMachine.state == C2ReplayStateMachine.State.PLAYBACK_UNVERIFIED
        ) { "Cannot rearm C2 from ${stateMachine.state}" }

        waitUntilPlaybackWindowElapsed()
        try { t.stop() } catch (_: Throwable) {}
        try { t.flush() } catch (_: Throwable) {}
        preloadInternal(t, initial = false)
    }

    fun preparationSnapshot(): Preparation {
        val t = track ?: error("C2 AudioTrack is not prepared")
        return Preparation(
            trackState = t.state,
            playState = t.playState,
            channelCount = t.channelCount,
            performanceMode = try { t.performanceMode } catch (_: Throwable) { -1 },
            writtenSamples = lastWrittenSamples,
            writtenFrames = lastWrittenFrames,
            writtenBytes = lastWrittenBytes,
            playbackHeadPosition = unsignedHead(t.playbackHeadPosition)
        )
    }

    fun release() {
        val t = track
        if (t != null) {
            try { t.stop() } catch (_: Throwable) {}
            try { t.flush() } catch (_: Throwable) {}
            try { t.release() } catch (_: Throwable) {}
        }
        track = null
        stateMachine.release()
    }

    fun completedCycles(): Int = stateMachine.completedCycles

    private fun createPersistentTrack(): AudioTrack {
        val masks = intArrayOf(AudioFormat.CHANNEL_OUT_MONO, AudioFormat.CHANNEL_OUT_STEREO)
        var lastError: Throwable? = null
        for (mask in masks) {
            for (lowLatency in booleanArrayOf(true, false)) {
                try {
                    val minOut = AudioTrack.getMinBufferSize(
                        sampleRate,
                        mask,
                        AudioFormat.ENCODING_PCM_16BIT
                    )
                    require(minOut > 0) { "AudioTrack min buffer failed for mask=$mask: $minOut" }
                    val channels = if (mask == AudioFormat.CHANNEL_OUT_MONO) 1 else 2
                    val pcmBytes = samples.size * channels * 2
                    val builder = AudioTrack.Builder()
                        .setAudioAttributes(
                            AudioAttributes.Builder()
                                .setUsage(AudioAttributes.USAGE_MEDIA)
                                .setContentType(AudioAttributes.CONTENT_TYPE_MUSIC)
                                .build()
                        )
                        .setAudioFormat(
                            AudioFormat.Builder()
                                .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                                .setSampleRate(sampleRate)
                                .setChannelMask(mask)
                                .build()
                        )
                        .setBufferSizeInBytes(maxOf(minOut * 2, pcmBytes + 4096))
                        .setTransferMode(AudioTrack.MODE_STREAM)
                    if (lowLatency) {
                        builder.setPerformanceMode(AudioTrack.PERFORMANCE_MODE_LOW_LATENCY)
                    }
                    val candidate = builder.build()
                    require(candidate.state == AudioTrack.STATE_INITIALIZED) {
                        "AudioTrack state=${candidate.state}"
                    }
                    candidate.setVolume(1.0f)
                    return candidate
                } catch (t: Throwable) {
                    lastError = t
                }
            }
        }
        throw IllegalStateException("Unable to create persistent 48 kHz PCM16 AudioTrack", lastError)
    }

    private fun preloadInternal(t: AudioTrack, initial: Boolean) {
        val written = t.write(outputSamples, 0, outputSamples.size, AudioTrack.WRITE_BLOCKING)
        require(written == outputSamples.size) {
            "C2 buffer write incomplete: $written/${outputSamples.size} samples"
        }
        lastWrittenSamples = written
        lastWrittenFrames = written / t.channelCount.coerceAtLeast(1)
        lastWrittenBytes = written * 2
        stateMachine.bufferLoaded()
        onLog(if (initial) "C2 buffer loaded" else "C2 buffer loaded for next playback")
    }

    private fun waitUntilPlaybackWindowElapsed() {
        if (lastPlayCallNs <= 0L) return
        val desiredNs = ((outputFrames * 1_000_000_000L) / sampleRate) + 100_000_000L
        val elapsed = System.nanoTime() - lastPlayCallNs
        val remaining = desiredNs - elapsed
        if (remaining > 0L) {
            Thread.sleep(remaining / 1_000_000L, (remaining % 1_000_000L).toInt())
        }
    }

    private fun unsignedHead(value: Int): Long = value.toLong() and 0xffffffffL

    private fun unsignedDelta(before: Long, after: Long): Long =
        (after - before) and 0xffffffffL
}
