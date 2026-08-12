package com.example.avtwinresponder

import kotlin.math.abs

data class ProbeSignal(
    val samples: ShortArray,
    val name: String,
    val isBuiltIn: Boolean,
    val originalSampleRate: Int = SAMPLE_RATE,
    val originalChannels: Int = 1,
    val sourceChannel: String = if (isBuiltIn) "BUILT_IN_MONO" else "MONO",
    val leftPeak: Double = 0.0,
    val rightPeak: Double = 0.0,
    val internalPeak: Double = peakOf(samples),
    val sourceSha256: String = Sha256.pcm16Hex(samples),
    val internalPcmSha256: String = Sha256.pcm16Hex(samples),
    val sourceUri: String? = null
) {
    val durationMs: Double
        get() = samples.size * 1000.0 / SAMPLE_RATE

    fun summary(): String {
        val origin = if (isBuiltIn) "built-in" else "WAV"
        val sourceInfo = if (isBuiltIn || (originalSampleRate == SAMPLE_RATE && originalChannels == 1)) {
            ""
        } else {
            " | source=${originalSampleRate}Hz/${originalChannels}ch -> 48k mono"
        }
        return "$name | ${"%.1f".format(durationMs)} ms | $origin$sourceInfo | source channel=$sourceChannel"
    }

    fun channelDiagnostics(): String =
        "source channel = $sourceChannel\n" +
            "left peak = ${"%.6f".format(leftPeak)}\n" +
            "right peak = ${"%.6f".format(rightPeak)}\n" +
            "internal peak = ${"%.6f".format(internalPeak)}\n" +
            "source SHA256 = $sourceSha256\n" +
            "internal PCM SHA256 = $internalPcmSha256"

    companion object {
        const val SAMPLE_RATE = 48_000

        fun peakOf(samples: ShortArray): Double {
            if (samples.isEmpty()) return 0.0
            var peak = 0
            for (sample in samples) {
                val value = abs(sample.toInt()).coerceAtMost(32768)
                if (value > peak) peak = value
            }
            return peak / 32768.0
        }
    }
}

object ProbeDefaults {
    fun c1(): ProbeSignal {
        val samples = Chirp.linearPcm16(ProbeSignal.SAMPLE_RATE, 0.200, 11_000.0, 19_000.0)
        return ProbeSignal(
            samples = samples,
            name = "Default C1 11-19 kHz chirp",
            isBuiltIn = true,
            sourceChannel = "BUILT_IN_MONO",
            internalPeak = ProbeSignal.peakOf(samples),
            sourceSha256 = Sha256.pcm16Hex(samples),
            internalPcmSha256 = Sha256.pcm16Hex(samples)
        )
    }

    fun c2(): ProbeSignal {
        val samples = Chirp.linearPcm16(ProbeSignal.SAMPLE_RATE, 0.200, 50.0, 9_000.0)
        return ProbeSignal(
            samples = samples,
            name = "Default C2 50 Hz-9 kHz chirp",
            isBuiltIn = true,
            sourceChannel = "BUILT_IN_MONO",
            internalPeak = ProbeSignal.peakOf(samples),
            sourceSha256 = Sha256.pcm16Hex(samples),
            internalPcmSha256 = Sha256.pcm16Hex(samples)
        )
    }
}
