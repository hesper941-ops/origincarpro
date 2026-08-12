package com.example.avtwinresponder

import kotlin.math.roundToLong

data class CaptureAudioTimestamp(
    val framePosition: Long,
    val nanoTime: Long,
    val observedReadFrameCount: Long
)

data class PlaybackAudioTimestamp(
    val framePosition: Long,
    val nanoTime: Long
)

data class ReplyTimingResult(
    val t3Precise: Boolean,
    val t3Sample: Long?,
    val replyDelaySamples: Long?,
    val playbackFrameZeroNanoTime: Long?,
    val reason: String?
)

object ReplyTimingMapper {
    fun mapToCaptureTimeline(
        t2Sample: Long,
        sampleRate: Int,
        playback: PlaybackAudioTimestamp?,
        capture: CaptureAudioTimestamp?,
        inputSampleRate: Int,
        outputSampleRate: Int,
        routeStable: Boolean,
        inputRouteTrusted: Boolean,
        outputRouteTrusted: Boolean,
        maxReplyDelaySamples: Long = sampleRate.toLong()
    ): ReplyTimingResult {
        if (sampleRate <= 0) return fail("invalid_sample_rate")
        if (playback == null) return fail("audio_track_timestamp_unavailable")
        if (capture == null) return fail("audio_record_timestamp_unavailable")
        if (inputSampleRate != sampleRate || outputSampleRate != sampleRate) return fail("audio_sample_rate_mismatch")
        if (!routeStable) return fail("audio_route_changed_during_measurement")
        if (!inputRouteTrusted) return fail("input_route_not_trusted")
        if (!outputRouteTrusted) return fail("output_route_not_trusted")
        if (playback.nanoTime <= 0L || capture.nanoTime <= 0L || playback.framePosition < 0L || capture.framePosition < 0L) {
            return fail("invalid_audio_timestamp")
        }

        val nsPerFrame = 1_000_000_000.0 / sampleRate.toDouble()
        val playbackFrameZeroNs = playback.nanoTime - (playback.framePosition * nsPerFrame).roundToLong()
        val deltaNs = playbackFrameZeroNs - capture.nanoTime
        val projected = capture.framePosition + (deltaNs * sampleRate.toDouble() / 1_000_000_000.0).roundToLong()
        val replyDelay = projected - t2Sample

        if (replyDelay < 0L) return fail("mapped_t3_before_t2", playbackFrameZeroNs)
        if (replyDelay > maxReplyDelaySamples) return fail("reply_delay_out_of_range", playbackFrameZeroNs)

        return ReplyTimingResult(
            t3Precise = true,
            t3Sample = projected,
            replyDelaySamples = replyDelay,
            playbackFrameZeroNanoTime = playbackFrameZeroNs,
            reason = null
        )
    }

    private fun fail(reason: String, frameZeroNs: Long? = null) = ReplyTimingResult(
        t3Precise = false,
        t3Sample = null,
        replyDelaySamples = null,
        playbackFrameZeroNanoTime = frameZeroNs,
        reason = reason
    )
}
