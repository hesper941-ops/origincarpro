package com.example.avtwinresponder

data class ProbeSignal(
    val samples: ShortArray,
    val name: String,
    val isBuiltIn: Boolean,
    val originalSampleRate: Int = SAMPLE_RATE,
    val originalChannels: Int = 1
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
        return "$name | ${"%.1f".format(durationMs)} ms | $origin$sourceInfo"
    }

    companion object {
        const val SAMPLE_RATE = 48_000
    }
}

object ProbeDefaults {
    fun c1(): ProbeSignal = ProbeSignal(
        samples = Chirp.linearPcm16(ProbeSignal.SAMPLE_RATE, 0.200, 11_000.0, 19_000.0),
        name = "Default C1 11-19 kHz chirp",
        isBuiltIn = true
    )

    fun c2(): ProbeSignal = ProbeSignal(
        samples = Chirp.linearPcm16(ProbeSignal.SAMPLE_RATE, 0.200, 300.0, 9_000.0),
        name = "Default C2 0.3-9 kHz chirp",
        isBuiltIn = true
    )
}
