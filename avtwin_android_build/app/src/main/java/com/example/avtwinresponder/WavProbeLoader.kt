package com.example.avtwinresponder

import android.content.Context
import android.net.Uri
import android.provider.OpenableColumns
import java.nio.charset.StandardCharsets
import kotlin.math.abs
import kotlin.math.floor
import kotlin.math.roundToInt

object WavProbeLoader {
    private const val MAX_FILE_BYTES = 16 * 1024 * 1024
    private const val MAX_DURATION_SEC = 2.0
    private const val MIN_DURATION_SEC = 0.020

    fun load(context: Context, uri: Uri): ProbeSignal {
        val name = displayName(context, uri)
        val bytes = context.contentResolver.openInputStream(uri)?.use { input ->
            val data = input.readBytes()
            require(data.size <= MAX_FILE_BYTES) { "WAV is too large (>16 MB)" }
            data
        } ?: error("Cannot open selected file")
        return decodeBytes(name, bytes, uri.toString())
    }

    internal fun decodeBytes(name: String, bytes: ByteArray, sourceUri: String? = null): ProbeSignal {
        require(bytes.size >= 44) { "File is too small to be a WAV" }
        require(ascii(bytes, 0, 4) == "RIFF" && ascii(bytes, 8, 4) == "WAVE") {
            "Only RIFF/WAVE files are supported"
        }

        var audioFormat = -1
        var channels = -1
        var sampleRate = -1
        var bitsPerSample = -1
        var dataOffset = -1
        var dataSize = -1

        var p = 12
        while (p + 8 <= bytes.size) {
            val id = ascii(bytes, p, 4)
            val sizeLong = u32(bytes, p + 4)
            require(sizeLong <= Int.MAX_VALUE.toLong()) { "Invalid WAV chunk size" }
            val size = sizeLong.toInt()
            val start = p + 8
            require(start <= bytes.size && start + size <= bytes.size) { "Truncated WAV chunk: $id" }

            when (id) {
                "fmt " -> {
                    require(size >= 16) { "Invalid WAV fmt chunk" }
                    audioFormat = u16(bytes, start)
                    channels = u16(bytes, start + 2)
                    sampleRate = u32(bytes, start + 4).toInt()
                    bitsPerSample = u16(bytes, start + 14)
                }
                "data" -> {
                    dataOffset = start
                    dataSize = size
                }
            }

            p = start + size + (size and 1)
        }

        require(dataOffset >= 0 && dataSize > 0) { "WAV has no data chunk" }
        require(channels in 1..8) { "Unsupported WAV channel count: $channels" }
        require(sampleRate in 8_000..192_000) { "Unsupported WAV sample rate: $sampleRate" }

        val bytesPerSample = when (audioFormat) {
            1 -> when (bitsPerSample) {
                8 -> 1
                16 -> 2
                24 -> 3
                32 -> 4
                else -> error("PCM WAV must be 8/16/24/32-bit, got $bitsPerSample-bit")
            }
            3 -> {
                require(bitsPerSample == 32) { "Float WAV must be 32-bit" }
                4
            }
            else -> error("Unsupported WAV encoding format=$audioFormat. Use PCM or IEEE float WAV.")
        }

        val frameBytes = bytesPerSample * channels
        require(frameBytes > 0 && dataSize >= frameBytes) { "Invalid WAV data size" }
        val frameCount = dataSize / frameBytes

        // Project convention: stereo probes carry the chirp in RIGHT and keep LEFT silent.
        // RIGHT must be preserved exactly; averaging L/R would attenuate it by 6 dB.
        val selectedChannel = if (channels >= 2) 1 else 0
        val sourceChannel = if (channels >= 2) "RIGHT" else "MONO"
        val selected = ShortArray(frameCount)
        var leftPeak = 0.0
        var rightPeak = 0.0

        var frameBase = dataOffset
        for (i in 0 until frameCount) {
            var selectedUnit = 0.0
            for (ch in 0 until channels) {
                val o = frameBase + ch * bytesPerSample
                val value = when (audioFormat) {
                    1 -> pcmToUnit(bytes, o, bitsPerSample)
                    3 -> float32ToUnit(bytes, o)
                    else -> 0.0
                }
                if (ch == 0) leftPeak = maxOf(leftPeak, abs(value))
                if (ch == 1) rightPeak = maxOf(rightPeak, abs(value))
                if (ch == selectedChannel) selectedUnit = value
            }
            selected[i] = (selectedUnit.coerceIn(-1.0, 1.0) * 32767.0)
                .roundToInt().coerceIn(-32768, 32767).toShort()
            frameBase += frameBytes
        }
        if (channels == 1) rightPeak = 0.0

        val target = if (sampleRate == ProbeSignal.SAMPLE_RATE) selected else
            resampleLinear(selected, sampleRate, ProbeSignal.SAMPLE_RATE)

        val durationSec = target.size.toDouble() / ProbeSignal.SAMPLE_RATE
        require(durationSec >= MIN_DURATION_SEC) {
            "Probe is too short (${(durationSec * 1000).roundToInt()} ms); use >=20 ms"
        }
        require(durationSec <= MAX_DURATION_SEC) {
            "Probe is too long (${"%.2f".format(durationSec)} s); use <=2 s (paper uses ~0.2 s)"
        }

        return ProbeSignal(
            samples = target,
            name = name,
            isBuiltIn = false,
            originalSampleRate = sampleRate,
            originalChannels = channels,
            sourceChannel = sourceChannel,
            leftPeak = leftPeak,
            rightPeak = rightPeak,
            internalPeak = ProbeSignal.peakOf(target),
            sourceSha256 = Sha256.hex(bytes),
            internalPcmSha256 = Sha256.pcm16Hex(target),
            sourceUri = sourceUri
        )
    }

    private fun resampleLinear(input: ShortArray, sourceRate: Int, targetRate: Int): ShortArray {
        if (input.isEmpty() || sourceRate == targetRate) return input.copyOf()
        val outSize = ((input.size.toDouble() * targetRate) / sourceRate).roundToInt().coerceAtLeast(1)
        val out = ShortArray(outSize)
        val ratio = sourceRate.toDouble() / targetRate
        for (i in 0 until outSize) {
            val src = i * ratio
            val i0 = floor(src).toInt().coerceIn(0, input.lastIndex)
            val i1 = (i0 + 1).coerceAtMost(input.lastIndex)
            val frac = src - i0
            val v = input[i0] * (1.0 - frac) + input[i1] * frac
            out[i] = v.roundToInt().coerceIn(-32768, 32767).toShort()
        }
        return out
    }

    private fun pcmToUnit(b: ByteArray, o: Int, bits: Int): Double = when (bits) {
        8 -> (((b[o].toInt() and 0xff) - 128) / 128.0).coerceIn(-1.0, 1.0)
        16 -> s16(b, o) / 32768.0
        24 -> {
            var v = (b[o].toInt() and 0xff) or
                ((b[o + 1].toInt() and 0xff) shl 8) or
                ((b[o + 2].toInt() and 0xff) shl 16)
            if ((v and 0x800000) != 0) v = v or -0x1000000
            v / 8388608.0
        }
        32 -> s32(b, o) / 2147483648.0
        else -> 0.0
    }

    private fun float32ToUnit(b: ByteArray, o: Int): Double {
        val f = Float.fromBits(s32(b, o))
        return if (f.isFinite()) f.toDouble().coerceIn(-1.0, 1.0) else 0.0
    }

    private fun displayName(context: Context, uri: Uri): String {
        context.contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use { c ->
            if (c.moveToFirst()) {
                val index = c.getColumnIndex(OpenableColumns.DISPLAY_NAME)
                if (index >= 0) return c.getString(index) ?: "selected_probe.wav"
            }
        }
        return uri.lastPathSegment ?: "selected_probe.wav"
    }

    private fun ascii(b: ByteArray, o: Int, n: Int): String = String(b, o, n, StandardCharsets.US_ASCII)
    private fun u16(b: ByteArray, o: Int): Int = (b[o].toInt() and 0xff) or ((b[o + 1].toInt() and 0xff) shl 8)
    private fun s16(b: ByteArray, o: Int): Int {
        val v = u16(b, o)
        return if ((v and 0x8000) != 0) v - 0x10000 else v
    }
    private fun s32(b: ByteArray, o: Int): Int =
        (b[o].toInt() and 0xff) or ((b[o + 1].toInt() and 0xff) shl 8) or
            ((b[o + 2].toInt() and 0xff) shl 16) or (b[o + 3].toInt() shl 24)
    private fun u32(b: ByteArray, o: Int): Long = s32(b, o).toLong() and 0xffffffffL
}
