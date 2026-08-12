package com.example.avtwinresponder

import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.sqrt

class StreamingC1Detector(
    private val fullTemplate: ShortArray,
    private val sampleRate: Int = ProbeSignal.SAMPLE_RATE,
    private val threshold: Double = 0.28,
    private val pretrigger: Double = 0.18,
    private val useHighFrequencyGate: Boolean = false,
    private val minBandRatio: Double = 0.80,
    detectionMs: Int = 60
) {
    data class Detection(
        val detected: Boolean,
        val t2Sample: Long?,
        val score: Double,
        val candidateSample: Long?,
        val rejectionReason: String?,
        val bandRatio: Double?,
        val detectionCompletedAtSample: Long
    )

    private val activeStart = findActiveStart(fullTemplate)
    private val segmentLength = minOf(
        fullTemplate.size - activeStart,
        (sampleRate * detectionMs / 1000).coerceAtLeast(512)
    ).coerceAtLeast(1)
    private val segment = fullTemplate.copyOfRange(activeStart, activeStart + segmentLength)
    private val ring = RollingShortBuffer(maxOf(segmentLength * 3, sampleRate / 2))
    private var lastRejectReportedAtSample = Long.MIN_VALUE
    private var seenArmGeneration = StrictArmGate.generation()

    init {
        require(fullTemplate.isNotEmpty()) { "C1 template is empty" }
        require(segment.any { it.toInt() != 0 }) { "C1 detection segment is silent" }
    }

    fun reset(nextAbsoluteSample: Long) {
        ring.clear(nextAbsoluteSample)
        lastRejectReportedAtSample = Long.MIN_VALUE
        seenArmGeneration = StrictArmGate.generation()
    }

    @Synchronized
    fun appendOnly(samples: ShortArray, length: Int, absoluteStartSample: Long) {
        ring.append(samples, length, absoluteStartSample)
    }

    @Synchronized
    fun process(samples: ShortArray, length: Int, absoluteStartSample: Long): Detection? {
        val armGeneration = StrictArmGate.generation()

        // Formal experiment mode is fail-closed: without a fresh Linux ARM we do not even run
        // C1 correlation. This guarantees that ambient/echo detections cannot autonomously play C2.
        if (!StrictArmGate.isArmed()) {
            ring.clear(absoluteStartSample + length)
            lastRejectReportedAtSample = Long.MIN_VALUE
            seenArmGeneration = armGeneration
            return null
        }

        // A newly accepted/superseding ARM defines a new measurement. Drop every sample buffered
        // before it so an old C1 cannot be attached to the new measurement merely because its
        // correlation peak is discovered later. This gate is protocol-only; t2 remains an audio
        // sample index from the AudioRecord stream once post-ARM samples are processed.
        if (armGeneration != seenArmGeneration) {
            ring.clear(absoluteStartSample)
            lastRejectReportedAtSample = Long.MIN_VALUE
            seenArmGeneration = armGeneration
        }

        ring.append(samples, length, absoluteStartSample)
        if (ring.size < segment.size) return null

        val snapshot = ring.copyChronological()
        val newestStart = snapshot.size - segment.size
        val first = (newestStart - maxOf(length * 2, 128)).coerceAtLeast(0)
        var bestLocal = first
        var bestScore = -1.0
        var s = first
        while (s <= newestStart) {
            val score = normalizedScore(snapshot, segment, s, decimation = 4)
            if (score > bestScore) {
                bestScore = score
                bestLocal = s
            }
            s += 4
        }

        val fineStart = (bestLocal - 8).coerceAtLeast(first)
        val fineEnd = (bestLocal + 8).coerceAtMost(newestStart)
        s = fineStart
        while (s <= fineEnd) {
            val score = normalizedScore(snapshot, segment, s, decimation = 1)
            if (score > bestScore) {
                bestScore = score
                bestLocal = s
            }
            s++
        }

        val absoluteSegmentStart = ring.baseSample + bestLocal
        val c1Start = (absoluteSegmentStart - activeStart).coerceAtLeast(0L)
        val completedAt = absoluteStartSample + length
        val bandRatio = if (useHighFrequencyGate && bestScore >= pretrigger) {
            highToLowBandRatio(snapshot, bestLocal, segment.size)
        } else null
        val gateOk = !useHighFrequencyGate || (bandRatio != null && bandRatio >= minBandRatio)

        if (bestScore >= threshold && gateOk) {
            return Detection(
                detected = true,
                t2Sample = c1Start,
                score = bestScore,
                candidateSample = c1Start,
                rejectionReason = null,
                bandRatio = bandRatio,
                detectionCompletedAtSample = completedAt
            )
        }

        if (bestScore >= pretrigger && completedAt - lastRejectReportedAtSample >= sampleRate / 10) {
            lastRejectReportedAtSample = completedAt
            return Detection(
                detected = false,
                t2Sample = null,
                score = bestScore,
                candidateSample = c1Start,
                rejectionReason = if (!gateOk) "high_frequency_gate_failed" else "score_below_threshold",
                bandRatio = bandRatio,
                detectionCompletedAtSample = completedAt
            )
        }
        return null
    }

    @Synchronized
    fun debugWindow(centerSample: Long, beforeSamples: Int, afterSamples: Int): ShortArray? =
        ring.window(centerSample, beforeSamples, afterSamples)

    private fun findActiveStart(template: ShortArray): Int {
        var peak = 0
        for (s in template) peak = maxOf(peak, abs(s.toInt()))
        if (peak <= 0) return 0
        val threshold = maxOf(16, (peak * 0.02).toInt())
        for (i in template.indices) if (abs(template[i].toInt()) >= threshold) return i
        return 0
    }

    private fun normalizedScore(audio: ShortArray, template: ShortArray, start: Int, decimation: Int): Double {
        if (start < 0 || start + template.size > audio.size) return 0.0
        var dot = 0.0
        var ex = 0.0
        var et = 0.0
        var i = 0
        while (i < template.size) {
            val x = audio[start + i].toDouble()
            val t = template[i].toDouble()
            dot += x * t
            ex += x * x
            et += t * t
            i += decimation
        }
        if (ex <= 1e-12 || et <= 1e-12) return 0.0
        return abs(dot) / sqrt(ex * et)
    }

    private fun highToLowBandRatio(audio: ShortArray, start: Int, length: Int): Double {
        if (start < 0 || start + length > audio.size) return 0.0
        val high = doubleArrayOf(12_000.0, 14_000.0, 16_000.0, 18_000.0)
        val low = doubleArrayOf(2_000.0, 4_000.0, 6_000.0, 8_000.0)
        var hi = 0.0
        var lo = 0.0
        for (f in high) hi += goertzelPower(audio, start, length, f)
        for (f in low) lo += goertzelPower(audio, start, length, f)
        return hi / (lo + 1.0)
    }

    private fun goertzelPower(audio: ShortArray, start: Int, length: Int, freq: Double): Double {
        val omega = 2.0 * PI * freq / sampleRate
        val coeff = 2.0 * cos(omega)
        var s1 = 0.0
        var s2 = 0.0
        for (i in 0 until length) {
            val s0 = audio[start + i] + coeff * s1 - s2
            s2 = s1
            s1 = s0
        }
        return s1 * s1 + s2 * s2 - coeff * s1 * s2
    }
}

internal class RollingShortBuffer(private val capacity: Int) {
    private val data = ShortArray(capacity.coerceAtLeast(1024))
    private var head = 0
    var size: Int = 0
        private set
    var baseSample: Long = 0L
        private set

    fun clear(nextBaseSample: Long) {
        head = 0
        size = 0
        baseSample = nextBaseSample
    }

    fun append(src: ShortArray, length: Int, absoluteStartSample: Long) {
        require(length in 0..src.size)
        if (size == 0) baseSample = absoluteStartSample
        for (i in 0 until length) {
            if (size < data.size) {
                data[(head + size) % data.size] = src[i]
                size++
            } else {
                data[head] = src[i]
                head = (head + 1) % data.size
                baseSample++
            }
        }
    }

    fun copyChronological(): ShortArray {
        val out = ShortArray(size)
        for (i in 0 until size) out[i] = data[(head + i) % data.size]
        return out
    }

    fun window(centerSample: Long, beforeSamples: Int, afterSamples: Int): ShortArray? {
        if (size == 0) return null
        val startAbs = centerSample - beforeSamples
        val endAbs = centerSample + afterSamples
        val bufferEnd = baseSample + size
        val clampedStart = maxOf(baseSample, startAbs)
        val clampedEnd = minOf(bufferEnd, endAbs)
        if (clampedEnd <= clampedStart) return null
        val out = ShortArray((clampedEnd - clampedStart).toInt())
        val localStart = (clampedStart - baseSample).toInt()
        for (i in out.indices) out[i] = data[(head + localStart + i) % data.size]
        return out
    }
}
