package com.example.avtwinresponder

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class ContinuousSessionLogicTest {
    @Test
    fun multipleC1Cycles_doNotRetriggerUntilCooldownEnds() {
        val sm = ContinuousResponderStateMachine(cooldownSamples = 4800, minimumRearmSamples = 0)
        sm.start()
        assertTrue(sm.acceptC1())
        assertFalse("same C1 cannot trigger twice", sm.acceptC1())
        sm.c2Scheduled()
        sm.c2Playing()
        sm.reporting()
        sm.enterCooldown(10_000)
        assertEquals(ContinuousResponderStateMachine.State.COOLDOWN, sm.state)
        assertFalse(sm.acceptC1())
        assertFalse(sm.updateCaptureSample(14_799))
        assertTrue(sm.updateCaptureSample(14_800))
        assertEquals(ContinuousResponderStateMachine.State.LISTENING, sm.state)
        assertTrue("next C1 after cooldown is legal", sm.acceptC1())
    }

    @Test
    fun defaultAcousticRearmGuard_blocksSamePhysicalC1TailFor800ms() {
        val sm = ContinuousResponderStateMachine(cooldownSamples = 4_800)
        sm.start()
        assertTrue(sm.acceptC1())
        sm.c2Scheduled()
        sm.c2Playing()
        sm.reporting()
        sm.enterCooldown(100_000)

        assertEquals(138_400L, sm.cooldownUntilSample)
        assertFalse(sm.updateCaptureSample(138_399))
        assertFalse(sm.acceptC1())
        assertTrue(sm.updateCaptureSample(138_400))
        assertTrue(sm.acceptC1())
    }

    @Test
    fun pauseResumeAndSafeStop_areDeterministic() {
        val sm = ContinuousResponderStateMachine(100, minimumRearmSamples = 0)
        sm.start()
        sm.requestPause()
        assertEquals(ContinuousResponderStateMachine.State.PAUSED, sm.state)
        assertFalse(sm.acceptC1())
        sm.resume()
        assertEquals(ContinuousResponderStateMachine.State.LISTENING, sm.state)
        sm.stop()
        assertEquals(ContinuousResponderStateMachine.State.STOPPED, sm.state)
    }

    @Test
    fun armPairing_matchesOnlyCurrentFreshMeasurement() {
        val p = ArmPairingManager("s1", maxArmAgeMs = 1000)
        assertFalse(p.accept(ArmCommand(1, "wrong", 1), 10).accepted)
        assertTrue(p.accept(ArmCommand(1, "s1", 12), 100).accepted)
        assertFalse("duplicate pending ARM rejected", p.accept(ArmCommand(1, "s1", 12), 110).accepted)
        val claim = p.claimNext(200)
        assertEquals("armed", claim.pairingMode)
        assertEquals(12L, claim.measurementId)
        assertFalse("old ARM cannot be reused", p.accept(ArmCommand(1, "s1", 12), 210).accepted)
        assertTrue(p.accept(ArmCommand(1, "s1", 13), 300).accepted)
        val next = p.claimNext(301)
        assertEquals(13L, next.measurementId)
    }

    @Test
    fun unarmedMode_usesChronologicalLocalSequence() {
        val p = ArmPairingManager("s")
        val a = p.claimNext(1)
        val b = p.claimNext(2)
        assertEquals("chronological_unarmed", a.pairingMode)
        assertEquals(1L, a.measurementId)
        assertEquals(2L, b.measurementId)
    }

    @Test
    fun expiredArm_isNotAttachedToLaterC1() {
        val p = ArmPairingManager("s", maxArmAgeMs = 50)
        assertTrue(p.accept(ArmCommand(1, "s", 99), 100).accepted)
        val claim = p.claimNext(200)
        assertEquals("chronological_unarmed", claim.pairingMode)
        assertEquals(1L, claim.measurementId)
    }

    @Test
    fun armJsonParser_parsesProtocolOne() {
        val a = ArmCommand.parse("""{"type":"arm","protocol_version":1,"session_id":"abc","measurement_id":12}""")
        assertNotNull(a)
        assertEquals("abc", a!!.sessionId)
        assertEquals(12L, a.measurementId)
        assertNull(ArmCommand.parse("""{"type":"other"}"""))
    }

    @Test
    fun audioTimestampProjection_producesSameCaptureTimelineT3() {
        val sampleRate = 48_000
        val capture = CaptureAudioTimestamp(framePosition = 50_000, nanoTime = 2_000_000_000L, observedReadFrameCount = 50_000)
        val playback = PlaybackAudioTimestamp(framePosition = 480, nanoTime = 2_020_000_000L)
        val result = ReplyTimingMapper.mapToCaptureTimeline(
            t2Sample = 49_000,
            sampleRate = sampleRate,
            playback = playback,
            capture = capture,
            inputSampleRate = sampleRate,
            outputSampleRate = sampleRate,
            routeStable = true,
            inputRouteTrusted = true,
            outputRouteTrusted = true
        )
        assertTrue(result.t3Precise)
        assertEquals(50_480L, result.t3Sample)
        assertEquals(1_480L, result.replyDelaySamples)
    }

    @Test
    fun audioTimestampProjection_worksForLaterPersistentTrackRound() {
        val sampleRate = 48_000
        val capture = CaptureAudioTimestamp(framePosition = 200_000, nanoTime = 5_000_000_000L, observedReadFrameCount = 200_000)
        val playback = PlaybackAudioTimestamp(
            framePosition = 96_480,
            nanoTime = 5_020_000_000L,
            roundStartFramePosition = 96_000
        )
        val result = ReplyTimingMapper.mapToCaptureTimeline(
            t2Sample = 199_000,
            sampleRate = sampleRate,
            playback = playback,
            capture = capture,
            inputSampleRate = sampleRate,
            outputSampleRate = sampleRate,
            routeStable = true,
            inputRouteTrusted = true,
            outputRouteTrusted = true
        )
        assertTrue(result.t3Precise)
        assertEquals(200_480L, result.t3Sample)
        assertEquals(1_480L, result.replyDelaySamples)
    }

    @Test
    fun t3IsFalseWhenTrackTimestampOrRouteIsUntrusted() {
        val capture = CaptureAudioTimestamp(10_000, 1_000_000_000L, 10_000)
        val noTrack = ReplyTimingMapper.mapToCaptureTimeline(
            t2Sample = 9_000,
            sampleRate = 48_000,
            playback = null,
            capture = capture,
            inputSampleRate = 48_000,
            outputSampleRate = 48_000,
            routeStable = true,
            inputRouteTrusted = true,
            outputRouteTrusted = true
        )
        assertFalse(noTrack.t3Precise)
        assertNull(noTrack.t3Sample)
        assertEquals("audio_track_timestamp_unavailable", noTrack.reason)

        val badRoute = ReplyTimingMapper.mapToCaptureTimeline(
            t2Sample = 9_000,
            sampleRate = 48_000,
            playback = PlaybackAudioTimestamp(100, 1_010_000_000L),
            capture = capture,
            inputSampleRate = 48_000,
            outputSampleRate = 48_000,
            routeStable = false,
            inputRouteTrusted = true,
            outputRouteTrusted = true
        )
        assertFalse(badRoute.t3Precise)
        assertEquals("audio_route_changed_during_measurement", badRoute.reason)
    }

    @Test
    fun eventJournal_recoversCompleteJsonLinesAfterTruncatedTail() {
        val text = "{\"type\":\"a\"}\n{\"type\":\"b\"}\n{\"type\":"
        val recovered = SessionJournal.recoverCompleteObjects(text)
        assertEquals(2, recovered.size)
    }

    @Test
    fun persistedTreePermission_requiresMatchingReadAndWriteGrant() {
        assertTrue(PersistedTreePermissionPolicy.restorable(true, true, true))
        assertFalse(PersistedTreePermissionPolicy.restorable(true, true, false))
        assertFalse(PersistedTreePermissionPolicy.restorable(false, true, true))
    }

    @Test
    fun sha256_isStableAndSensitiveToProbeBytes() {
        val a = Sha256.hex(byteArrayOf(1, 2, 3))
        val b = Sha256.hex(byteArrayOf(1, 2, 3))
        val c = Sha256.hex(byteArrayOf(1, 2, 4))
        assertEquals(a, b)
        assertFalse(a == c)
        assertEquals(64, a.length)
    }
}
