package com.example.avtwinresponder

import android.media.AudioAttributes
import android.media.AudioDeviceInfo
import android.media.AudioFormat
import android.media.AudioRouting
import android.media.AudioTimestamp
import android.media.AudioTrack

internal class PersistentC2Player(
    private val samples: ShortArray,
    private val sampleRate: Int = ProbeSignal.SAMPLE_RATE,
    private val preferredOutput: AudioDeviceInfo? = null,
    private val onLog: (String) -> Unit = {},
    private val onRouteChanged: (String) -> Unit = {}
) {
    data class Preparation(
        val trackState: Int,
        val playState: Int,
        val channelCount: Int,
        val performanceMode: Int,
        val writtenSamples: Int,
        val writtenFrames: Int,
        val writtenBytes: Int,
        val playbackHeadPosition: Long,
        val preferredRoute: String
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
        val playbackVerified: Boolean,
        val actualOutputRoute: String,
        val actualOutputRouteType: Int?,
        val outputSampleRate: Int,
        val underrunCount: Int
    )

    private val stateMachine = C2ReplayStateMachine()
    private var track: AudioTrack? = null
    private var routeListener: AudioRouting.OnRoutingChangedListener? = null
    private var lastRouteDescription: String? = null
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
        if (preferredOutput != null) {
            val preferred = try { t.setPreferredDevice(preferredOutput) } catch (_: Throwable) { false }
            lastRouteDescription = if (preferred) describeDevice(preferredOutput) else null
            onLog("C2 preferred output=${describeDevice(preferredOutput)} accepted=$preferred")
        }
        routeListener = AudioRouting.OnRoutingChangedListener { router ->
            val route = describeDevice(router.routedDevice)
            val old = lastRouteDescription
            lastRouteDescription = route
            if (old == null || old == "unavailable") {
                // First activation of a previously inactive track is expected and is not a mid-measurement reroute.
                onLog("output route activated: $route")
            } else if (route == old) {
                onLog("output route confirmed: $route")
            } else {
                onLog("output route changed: $old -> $route")
                onRouteChanged(route)
            }
        }.also { t.addOnRoutingChangedListener(it, null) }
        stateMachine.trackInitialized()
        onLog("C2 AudioTrack initialized")

        outputSamples = if (t.channelCount == 1) samples.copyOf() else {
            ShortArray(samples.size * 2).also { stereo ->
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

    fun playAndVerify(timeoutMs: Long = 350L, onPlayIssued: ((Long) -> Unit)? = null): PlaybackVerification {
        val t = track ?: error("C2 AudioTrack is not prepared")
        require(stateMachine.state == C2ReplayStateMachine.State.BUFFER_LOADED) {
            "C2 buffer is not ready: ${stateMachine.state}"
        }

        val headBefore = unsignedHead(t.playbackHeadPosition)
        val baselineTimestamp = AudioTimestamp()
        val baselineTimestampValid = try { t.getTimestamp(baselineTimestamp) } catch (_: Throwable) { false }
        val baselineTimestampNs = if (baselineTimestampValid) baselineTimestamp.nanoTime else -1L

        lastPlayCallNs = System.nanoTime() // diagnostic only; never protocol t3
        t.play()
        stateMachine.playIssued()
        onLog("C2 play() issued")
        onPlayIssued?.invoke(lastPlayCallNs)

        val deadline = System.nanoTime() + timeoutMs * 1_000_000L
        var headAdvanced = false
        var timestampValid = false
        var firstTimestampNs: Long? = null
        var firstTimestampFrame: Long? = null
        var lastHead = headBefore

        while (System.nanoTime() < deadline) {
            lastHead = unsignedHead(t.playbackHeadPosition)
            if (!headAdvanced && unsignedDelta(headBefore, lastHead) > 0L) {
                headAdvanced = true
                onLog("C2 playback head advanced")
            }
            val ts = AudioTimestamp()
            val ok = try { t.getTimestamp(ts) } catch (_: Throwable) { false }
            val fresh = ok && ts.framePosition >= 0L && ts.nanoTime > 0L &&
                (!baselineTimestampValid || ts.nanoTime > baselineTimestampNs)
            if (fresh && !timestampValid) {
                timestampValid = true
                firstTimestampNs = ts.nanoTime
                // Normalize both cumulative and reset-after-flush implementations to this C2 round.
                firstTimestampFrame = if (ts.framePosition >= headBefore) ts.framePosition - headBefore else ts.framePosition
            }
            if (headAdvanced && timestampValid) break
            Thread.sleep(2)
        }

        val verified = headAdvanced || timestampValid
        stateMachine.verificationFinished(verified)
        if (verified) onLog("C2 hardware playback verified")
        else onLog("PLAY() CALLED — HARDWARE PLAYBACK NOT VERIFIED")

        val routed = try { t.routedDevice } catch (_: Throwable) { null }
        val routedDescription = describeDevice(routed)
        lastRouteDescription = routedDescription.takeUnless { it == "unavailable" } ?: lastRouteDescription
        return PlaybackVerification(
            playCallTimeNs = lastPlayCallNs,
            trackState = t.state,
            playState = t.playState,
            writeSamples = lastWrittenSamples,
            writeFrames = lastWrittenFrames,
            writeBytes = lastWrittenBytes,
            playbackHeadBefore = headBefore,
            playbackHeadAfter = lastHead,
            playbackHeadAdvanced = headAdvanced,
            audioTimestampValid = timestampValid,
            firstValidAudioTimestampNs = firstTimestampNs,
            firstValidAudioFramePosition = firstTimestampFrame,
            playbackVerified = verified,
            actualOutputRoute = routedDescription,
            actualOutputRouteType = routed?.type,
            outputSampleRate = t.sampleRate,
            underrunCount = try { t.underrunCount } catch (_: Throwable) { -1 }
        )
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
            playbackHeadPosition = unsignedHead(t.playbackHeadPosition),
            preferredRoute = describeDevice(preferredOutput)
        )
    }

    fun outputSampleRate(): Int = track?.sampleRate ?: sampleRate
    fun preferredOutputDescription(): String = describeDevice(preferredOutput)

    fun release() {
        val t = track
        if (t != null) {
            try { routeListener?.let { t.removeOnRoutingChangedListener(it) } } catch (_: Throwable) {}
            try { t.stop() } catch (_: Throwable) {}
            try { t.flush() } catch (_: Throwable) {}
            try { t.release() } catch (_: Throwable) {}
        }
        routeListener = null
        track = null
        lastRouteDescription = null
        stateMachine.release()
    }

    fun completedCycles(): Int = stateMachine.completedCycles

    private fun createPersistentTrack(): AudioTrack {
        val masks = intArrayOf(AudioFormat.CHANNEL_OUT_MONO, AudioFormat.CHANNEL_OUT_STEREO)
        var lastError: Throwable? = null
        for (mask in masks) {
            for (lowLatency in booleanArrayOf(true, false)) {
                try {
                    val minOut = AudioTrack.getMinBufferSize(sampleRate, mask, AudioFormat.ENCODING_PCM_16BIT)
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
                    if (lowLatency) builder.setPerformanceMode(AudioTrack.PERFORMANCE_MODE_LOW_LATENCY)
                    val candidate = builder.build()
                    require(candidate.state == AudioTrack.STATE_INITIALIZED) { "AudioTrack state=${candidate.state}" }
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
        require(written == outputSamples.size) { "C2 buffer write incomplete: $written/${outputSamples.size} samples" }
        lastWrittenSamples = written
        lastWrittenFrames = written / t.channelCount.coerceAtLeast(1)
        lastWrittenBytes = written * 2
        stateMachine.bufferLoaded()
        onLog(if (initial) "C2 buffer loaded" else "C2 buffer loaded for next playback")
    }

    private fun waitUntilPlaybackWindowElapsed() {
        if (lastPlayCallNs <= 0L) return
        val desiredNs = ((outputFrames * 1_000_000_000L) / sampleRate) + 80_000_000L
        val elapsed = System.nanoTime() - lastPlayCallNs
        val remaining = desiredNs - elapsed
        if (remaining > 0L) Thread.sleep(remaining / 1_000_000L, (remaining % 1_000_000L).toInt())
    }

    private fun unsignedHead(value: Int): Long = value.toLong() and 0xffffffffL
    private fun unsignedDelta(before: Long, after: Long): Long = (after - before) and 0xffffffffL

    companion object {
        fun describeDevice(device: AudioDeviceInfo?): String {
            if (device == null) return "unavailable"
            val rates = device.sampleRates.joinToString(",").ifBlank { "unspecified" }
            return "${typeName(device.type)} id=${device.id} ${device.productName} rates=$rates"
        }

        fun isTrustedBuiltinSpeaker(type: Int?): Boolean = type == AudioDeviceInfo.TYPE_BUILTIN_SPEAKER

        private fun typeName(type: Int): String = when (type) {
            AudioDeviceInfo.TYPE_BUILTIN_SPEAKER -> "BUILTIN_SPEAKER"
            AudioDeviceInfo.TYPE_BUILTIN_EARPIECE -> "BUILTIN_EARPIECE"
            AudioDeviceInfo.TYPE_BLUETOOTH_A2DP -> "BLUETOOTH_A2DP"
            AudioDeviceInfo.TYPE_BLUETOOTH_SCO -> "BLUETOOTH_SCO"
            AudioDeviceInfo.TYPE_USB_DEVICE -> "USB_DEVICE"
            AudioDeviceInfo.TYPE_USB_HEADSET -> "USB_HEADSET"
            AudioDeviceInfo.TYPE_WIRED_HEADPHONES -> "WIRED_HEADPHONES"
            AudioDeviceInfo.TYPE_WIRED_HEADSET -> "WIRED_HEADSET"
            else -> "TYPE_$type"
        }
    }
}
