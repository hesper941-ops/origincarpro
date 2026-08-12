package com.example.avtwinresponder

import kotlin.math.roundToLong

data class CaptureAudioTimestamp(
    val framePosition: Long,
    val nanoTime: Long,
    val observedReadFrameCount: Long
)

data class PlaybackAudioTimestamp(
    val framePosition: Long,
    val nanoTime: Long,
    val roundStartFramePosition: Long = 0L
)

data class ReplyTimingResult(
    val t3Precise: Boolean,
    val t3Sample: Long?,
    val replyDelaySamples: Long?,
    val playbackRoundStartNanoTime: Long?,
    val reason: String?
) {
    // Backward-compatible diagnostic alias; this is the current C2 round start,
    // not necessarily AudioTrack lifetime frame zero.
    val playbackFrameZeroNanoTime: Long?
        get() = playbackRoundStartNanoTime
}

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
        if (
            playback.nanoTime <= 0L || capture.nanoTime <= 0L ||
            playback.framePosition < 0L || playback.roundStartFramePosition < 0L || capture.framePosition < 0L
        ) return fail("invalid_audio_timestamp")

        val framesSinceRoundStart = playback.framePosition - playback.roundStartFramePosition
        if (framesSinceRoundStart < 0L) return fail("playback_timestamp_precedes_round_start")

        // This calculation contains only audio-frame timestamps. System.nanoTime/play() times
        // are diagnostics and never enter the t3 calculation.
        val nsPerFrame = 1_000_000_000.0 / sampleRate.toDouble()
        val playbackRoundStartNs = playback.nanoTime - (framesSinceRoundStart * nsPerFrame).roundToLong()
        val deltaNs = playbackRoundStartNs - capture.nanoTime
        val projected = capture.framePosition +
            (deltaNs * sampleRate.toDouble() / 1_000_000_000.0).roundToLong()
        val replyDelay = projected - t2Sample

        if (replyDelay < 0L) return fail("mapped_t3_before_t2", playbackRoundStartNs)
        if (replyDelay > maxReplyDelaySamples) return fail("reply_delay_out_of_range", playbackRoundStartNs)

        return ReplyTimingResult(
            t3Precise = true,
            t3Sample = projected,
            replyDelaySamples = replyDelay,
            playbackRoundStartNanoTime = playbackRoundStartNs,
            reason = null
        )
    }

    private fun fail(reason: String, roundStartNs: Long? = null) = ReplyTimingResult(
        t3Precise = false,
        t3Sample = null,
        replyDelaySamples = null,
        playbackRoundStartNanoTime = roundStartNs,
        reason = reason
    )
}
