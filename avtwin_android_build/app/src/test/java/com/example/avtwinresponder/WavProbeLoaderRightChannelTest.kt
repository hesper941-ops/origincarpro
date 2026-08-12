package com.example.avtwinresponder

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.nio.ByteBuffer
import java.nio.ByteOrder

class WavProbeLoaderRightChannelTest {
    @Test
    fun stereoLeftSilentRightSignal_usesRightWithoutAveraging() {
        val frames = 1_200
        val right = ShortArray(frames) { i -> if (i % 2 == 0) 16_000 else -16_000 }
        val wav = stereoPcm16(left = ShortArray(frames), right = right, sampleRate = 48_000)

        val signal = WavProbeLoader.decodeBytes("right_only.wav", wav)

        assertEquals("RIGHT", signal.sourceChannel)
        assertEquals(2, signal.originalChannels)
        assertEquals(48_000, signal.originalSampleRate)
        assertEquals(frames, signal.samples.size)
        assertEquals(0.0, signal.leftPeak, 1e-9)
        assertTrue("right peak should remain strong", signal.rightPeak > 0.48)
        assertTrue("internal mono must retain RIGHT amplitude, not half it", signal.internalPeak > 0.48)
        assertTrue(signal.samples.any { kotlin.math.abs(it.toInt()) > 15_000 })
    }

    @Test
    fun monoWav_remainsMono() {
        val frames = 1_200
        val mono = ShortArray(frames) { 8_000 }
        val wav = monoPcm16(mono, 48_000)

        val signal = WavProbeLoader.decodeBytes("mono.wav", wav)

        assertEquals("MONO", signal.sourceChannel)
        assertEquals(1, signal.originalChannels)
        assertTrue(signal.internalPeak > 0.24)
    }

    private fun stereoPcm16(left: ShortArray, right: ShortArray, sampleRate: Int): ByteArray {
        require(left.size == right.size)
        val channels = 2
        val dataBytes = left.size * channels * 2
        val buffer = ByteBuffer.allocate(44 + dataBytes).order(ByteOrder.LITTLE_ENDIAN)
        putHeader(buffer, channels, sampleRate, dataBytes)
        for (i in left.indices) {
            buffer.putShort(left[i])
            buffer.putShort(right[i])
        }
        return buffer.array()
    }

    private fun monoPcm16(samples: ShortArray, sampleRate: Int): ByteArray {
        val dataBytes = samples.size * 2
        val buffer = ByteBuffer.allocate(44 + dataBytes).order(ByteOrder.LITTLE_ENDIAN)
        putHeader(buffer, 1, sampleRate, dataBytes)
        samples.forEach { buffer.putShort(it) }
        return buffer.array()
    }

    private fun putHeader(buffer: ByteBuffer, channels: Int, sampleRate: Int, dataBytes: Int) {
        buffer.put("RIFF".toByteArray(Charsets.US_ASCII))
        buffer.putInt(36 + dataBytes)
        buffer.put("WAVE".toByteArray(Charsets.US_ASCII))
        buffer.put("fmt ".toByteArray(Charsets.US_ASCII))
        buffer.putInt(16)
        buffer.putShort(1)
        buffer.putShort(channels.toShort())
        buffer.putInt(sampleRate)
        buffer.putInt(sampleRate * channels * 2)
        buffer.putShort((channels * 2).toShort())
        buffer.putShort(16)
        buffer.put("data".toByteArray(Charsets.US_ASCII))
        buffer.putInt(dataBytes)
    }
}
