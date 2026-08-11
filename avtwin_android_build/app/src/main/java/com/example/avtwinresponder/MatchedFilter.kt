package com.example.avtwinresponder

import kotlin.math.abs
import kotlin.math.sqrt

object MatchedFilter {
    fun score(
        audio: ShortAccumulator,
        template: ShortArray,
        start: Int,
        decimation: Int = 8
    ): Double {
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

    fun bestRecent(
        audio: ShortAccumulator,
        template: ShortArray,
        newestStart: Int,
        searchSamples: Int,
        startStep: Int = 8,
        decimation: Int = 8
    ): Pair<Int, Double> {
        val first = (newestStart - searchSamples).coerceAtLeast(0)
        val last = newestStart.coerceAtMost(audio.size - template.size)
        if (last < first) return newestStart to 0.0
        var bestStart = first
        var bestScore = -1.0
        var s = first
        while (s <= last) {
            val sc = score(audio, template, s, decimation)
            if (sc > bestScore) {
                bestScore = sc
                bestStart = s
            }
            s += startStep
        }
        return bestStart to bestScore
    }

    fun score(
        audio: ShortArray,
        template: ShortArray,
        start: Int,
        decimation: Int = 1
    ): Double {
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

    fun refineStrongest(
        audio: ShortArray,
        template: ShortArray,
        coarseStart: Int,
        marginSamples: Int = 480
    ): Pair<Int, Double> {
        val minStart = (coarseStart - marginSamples).coerceAtLeast(0)
        val maxStart = (coarseStart + marginSamples).coerceAtMost(audio.size - template.size)
        if (maxStart < minStart) return coarseStart to 0.0

        var bestStart = minStart
        var bestScore = -1.0
        var s = minStart
        while (s <= maxStart) {
            val sc = score(audio, template, s, decimation = 4)
            if (sc > bestScore) {
                bestScore = sc
                bestStart = s
            }
            s += 4
        }

        val fineMin = (bestStart - 6).coerceAtLeast(minStart)
        val fineMax = (bestStart + 6).coerceAtMost(maxStart)
        s = fineMin
        while (s <= fineMax) {
            val sc = score(audio, template, s, decimation = 1)
            if (sc > bestScore) {
                bestScore = sc
                bestStart = s
            }
            s++
        }
        return bestStart to bestScore
    }
}
