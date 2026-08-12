package com.example.avtwinresponder

import org.junit.Assert.assertEquals
import org.junit.Test

class C2ReplayStateMachineTest {
    @Test
    fun twentyVerifiedReplayCyclesReuseOneLifecycle() {
        val sm = C2ReplayStateMachine()
        sm.trackInitialized()
        sm.bufferLoaded()

        repeat(20) { index ->
            sm.playIssued()
            sm.verificationFinished(true)
            assertEquals(index + 1, sm.completedCycles)
            if (index != 19) sm.bufferLoaded()
        }

        assertEquals(C2ReplayStateMachine.State.PLAYBACK_VERIFIED, sm.state)
        assertEquals(20, sm.completedCycles)
        sm.release()
        assertEquals(C2ReplayStateMachine.State.RELEASED, sm.state)
    }

    @Test
    fun unverifiedCycleCanStillBeRearmedWithoutRecreatingTrack() {
        val sm = C2ReplayStateMachine()
        sm.trackInitialized()
        sm.bufferLoaded()
        sm.playIssued()
        sm.verificationFinished(false)
        assertEquals(C2ReplayStateMachine.State.PLAYBACK_UNVERIFIED, sm.state)

        sm.bufferLoaded()
        sm.playIssued()
        sm.verificationFinished(true)

        assertEquals(2, sm.completedCycles)
        assertEquals(C2ReplayStateMachine.State.PLAYBACK_VERIFIED, sm.state)
    }

    @Test(expected = IllegalArgumentException::class)
    fun cannotPlayBeforeBufferIsLoaded() {
        val sm = C2ReplayStateMachine()
        sm.trackInitialized()
        sm.playIssued()
    }
}
