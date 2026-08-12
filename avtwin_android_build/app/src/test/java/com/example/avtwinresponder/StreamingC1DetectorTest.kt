package com.example.avtwinresponder

import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.random.Random

class StreamingC1DetectorTest {
    @Test
    fun unarmedC1CannotProduceDetection() {
        ArmPairingManager("session") // resets strict gate to unarmed
        val detector = StreamingC1Detector(
            fullTemplate = ProbeDefaults.c1().samples,
            threshold = 0.25,
            pretrigger = 0.15,
            useHighFrequencyGate = false,
            detectionMs = 60
        )
        val c1 = ProbeDefaults.c1().samples
        detector.reset(0L)
        var absolute = 0L
        var found: StreamingC1Detector.Detection? = null
        var offset = 0
        while (offset < c1.size) {
            val n = minOf(240, c1.size - offset)
            val chunk = c1.copyOfRange(offset, offset + n)
            val d = detector.process(chunk, n, absolute)
            if (d?.detected == true) found = d
            offset += n
            absolute += n
        }
        assertNull(found)
    }

    @Test
    fun detectsKnownC1AcrossStreamingChunksWithinOneHundredMs_afterArm() {
        val pairing = ArmPairingManager("session")
        val c1 = ProbeDefaults.c1().samples
        val detector = StreamingC1Detector(
            fullTemplate = c1,
            threshold = 0.25,
            pretrigger = 0.15,
            useHighFrequencyGate = false,
            detectionMs = 60
        )
        val prefix = 2_000
        val signal = ShortArray(prefix + c1.size + 1_000)
        val random = Random(7)
        for (i in signal.indices) signal[i] = random.nextInt(-120, 121).toShort()
        for (i in c1.indices) {
            val mixed = signal[prefix + i].toInt() + c1[i].toInt()
            signal[prefix + i] = mixed.coerceIn(-32768, 32767).toShort()
        }

        detector.reset(0L)
        assertTrue(pairing.accept(ArmCommand(1, "session", 1), 100).accepted)
        var absolute = 0L
        var found: StreamingC1Detector.Detection? = null
        val hop = 240
        var offset = 0
        while (offset < signal.size && found == null) {
            val n = minOf(hop, signal.size - offset)
            val chunk = signal.copyOfRange(offset, offset + n)
            val d = detector.process(chunk, n, absolute)
            if (d?.detected == true) found = d
            offset += n
            absolute += n
        }

        assertNotNull(found)
        val d = found!!
        assertTrue(kotlin.math.abs((d.t2Sample ?: -1L) - prefix.toLong()) <= 12L)
        val latency = d.detectionCompletedAtSample - (d.t2Sample ?: 0L)
        assertTrue("streaming detector should complete under 100 ms", latency <= 4_800L)
        assertTrue(d.score >= 0.25)

        // Claim consumes this ARM. A second physical/correlation event cannot be authorized by it.
        val claim = pairing.claimNext(200)
        assertTrue(claim.measurementId == 1L)
        assertTrue(!StrictArmGate.isArmed())
    }
}
