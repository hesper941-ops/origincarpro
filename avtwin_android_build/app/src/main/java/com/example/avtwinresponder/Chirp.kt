package com.example.avtwinresponder

import kotlin.math.PI
import kotlin.math.min
import kotlin.math.sin

object Chirp {
    fun linearPcm16(
        sampleRate: Int,
        durationSec: Double,
        f0Hz: Double,
        f1Hz: Double,
        amplitude: Double = 0.65,
        fadeSec: Double = 0.005
    ): ShortArray {
        val n = (sampleRate * durationSec).toInt()
        val out = ShortArray(n)
        val k = (f1Hz - f0Hz) / durationSec
        val fadeN = (sampleRate * fadeSec).toInt().coerceAtLeast(1)

        for (i in 0 until n) {
            val t = i.toDouble() / sampleRate
            val phase = 2.0 * PI * (f0Hz * t + 0.5 * k * t * t)
            var env = 1.0
            if (i < fadeN) {
                env = 0.5 - 0.5 * kotlin.math.cos(PI * i.toDouble() / fadeN)
            } else if (i >= n - fadeN) {
                val j = n - 1 - i
                env = min(env, 0.5 - 0.5 * kotlin.math.cos(PI * j.toDouble() / fadeN))
            }
            val v = amplitude * env * sin(phase)
            out[i] = (v * Short.MAX_VALUE).toInt().coerceIn(Short.MIN_VALUE.toInt(), Short.MAX_VALUE.toInt()).toShort()
        }
        return out
    }

    fun prefix(x: ShortArray, count: Int): ShortArray = x.copyOfRange(0, count.coerceAtMost(x.size))
}
