package com.example.avtwinresponder

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class PauseCooldownTest {
    @Test
    fun pauseDuringMeasurement_doesNotBypassCooldownOnResume() {
        val sm = ContinuousResponderStateMachine(cooldownSamples = 4_800, minimumRearmSamples = 0)
        sm.start()
        assertTrue(sm.acceptC1())
        sm.c2Scheduled()
        sm.requestPause()
        sm.c2Playing()
        sm.reporting()
        sm.enterCooldown(10_000)
        assertEquals(ContinuousResponderStateMachine.State.PAUSED, sm.state)
        sm.updateCaptureSample(11_000)
        sm.resume()
        assertEquals(ContinuousResponderStateMachine.State.COOLDOWN, sm.state)
        assertFalse(sm.acceptC1())
        assertFalse(sm.updateCaptureSample(14_799))
        assertTrue(sm.updateCaptureSample(14_800))
        assertEquals(ContinuousResponderStateMachine.State.LISTENING, sm.state)
    }
}
