package com.example.avtwinresponder

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class UdpRetryPolicyTest {
    @Test
    fun productionReplyTiming_isAlwaysSingleShot() {
        assertEquals(1, UdpReporter.productionAttemptCount(1))
        assertEquals(1, UdpReporter.productionAttemptCount(3))
        assertEquals(1, UdpReporter.productionAttemptCount(99))
    }

    @Test
    fun retryPrimitive_canStillReuseIdenticalEventPayloadForFutureAckProtocol() {
        val eventId = "event-123"
        val payload = JsonWire.obj(
            "type" to "reply_timing",
            "android_event_id" to eventId,
            "measurement_id" to 12
        )
        val seen = ArrayList<String>()
        var calls = 0
        val transport = UdpReporter.Transport { _, _, json ->
            calls++
            seen += json
            if (calls < 3) throw IllegalStateException("simulated network send failure")
        }

        val result = UdpReporter.sendRepeatedWithTransport(
            host = "192.0.2.1",
            port = 5005,
            json = payload,
            repeats = 3,
            spacingMs = 0,
            transport = transport,
            sleeper = {}
        )

        assertEquals(3, result.attempts.size)
        assertEquals(listOf(false, false, true), result.attempts.map { it.success })
        assertTrue(result.success)
        assertEquals(3, seen.size)
        assertTrue(seen.all { it == payload })
        assertTrue(seen.all { JsonWire.stringField(it, "android_event_id") == eventId })
    }
}
