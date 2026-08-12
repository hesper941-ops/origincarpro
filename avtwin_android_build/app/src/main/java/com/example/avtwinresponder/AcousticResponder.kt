package com.example.avtwinresponder

import android.content.Context
import android.media.AudioAttributes
import android.media.AudioFocusRequest
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioRecord
import android.media.AudioRouting
import android.media.AudioTimestamp
import android.media.MediaRecorder
import android.media.audiofx.AcousticEchoCanceler
import android.media.audiofx.AutomaticGainControl
import android.media.audiofx.NoiseSuppressor
import android.net.Uri
import android.os.Process
import android.os.SystemClock
import java.util.Locale
import java.util.UUID
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong
import kotlin.concurrent.thread

class AcousticResponder(
    private val context: Context,
    private val c1Signal: ProbeSignal,
    private val c2Signal: ProbeSignal,
    private val onStatus: (String) -> Unit,
    private val onSnapshot: (SessionSnapshot) -> Unit
) {
    data class SessionConfig(
        val linuxHost: String,
        val controlPort: Int,
        val resultPort: Int,
        val resultTreeUri: Uri,
        val saveDebugAudio: Boolean,
        val cooldownMs: Int = 300
    )

    data class SessionSnapshot(
        val state: String,
        val sessionId: String?,
        val measurementId: Long?,
        val pendingArmMeasurementId: Long?,
        val successResponses: Long,
        val c1Rejected: Long,
        val c2Failures: Long,
        val udpFailures: Long,
        val lastReplyDelaySamples: Long?,
        val lastT3Precise: Boolean,
        val inputRoute: String,
        val outputRoute: String,
        val note: String = ""
    )

    companion object {
        const val SAMPLE_RATE = ProbeSignal.SAMPLE_RATE
        const val HOP_SAMPLES = 240
        const val C1_THRESHOLD = 0.28
        const val C1_PRETRIGGER = 0.18
        const val C1_MIN_BAND_RATIO = 0.80
    }

    private val running = AtomicBoolean(false)
    private val testRunning = AtomicBoolean(false)
    private val totalFramesRead = AtomicLong(0L)
    private val routeGeneration = AtomicLong(0L)

    private var captureThread: Thread? = null
    private var responseThread: Thread? = null
    private var record: AudioRecord? = null
    private var recordRouteListener: AudioRouting.OnRoutingChangedListener? = null
    private var player: PersistentC2Player? = null
    private var udpControl: UdpControlServer? = null
    private var storage: SafSessionStorage? = null
    private var stateMachine: ContinuousResponderStateMachine? = null
    private var pairing: ArmPairingManager? = null
    private var detector: StreamingC1Detector? = null

    private var aec: AcousticEchoCanceler? = null
    private var ns: NoiseSuppressor? = null
    private var agc: AutomaticGainControl? = null
    private var audioManager: AudioManager? = null
    private var focusRequest: AudioFocusRequest? = null
    @Volatile private var audioFocusLost = false
    @Volatile private var latestCaptureTimestamp: CaptureAudioTimestamp? = null
    @Volatile private var inputRoute = "unavailable"
    @Volatile private var outputRoute = "unavailable"

    private var currentConfig: SessionConfig? = null
    private var sessionId: String? = null
    @Volatile private var currentMeasurementId: Long? = null
    private var successResponses = 0L
    private var c1Rejected = 0L
    private var c2Failures = 0L
    private var udpFailures = 0L
    private var lastReplyDelaySamples: Long? = null
    private var lastT3Precise = false

    fun startSession(config: SessionConfig) {
        if (!running.compareAndSet(false, true)) return
        currentConfig = config
        captureThread = thread(name = "AVTwin-Continuous-Capture") {
            Process.setThreadPriority(Process.THREAD_PRIORITY_AUDIO)
            try {
                runContinuousSession(config)
            } catch (t: Throwable) {
                log("SESSION ERROR: ${t.javaClass.simpleName}: ${t.message}")
                storage?.appendEvent(JsonWire.obj("type" to "session_error", "error" to "${t.javaClass.simpleName}: ${t.message}"))
            } finally {
                running.set(false)
                try { responseThread?.join(1500) } catch (_: Throwable) {}
                cleanupSessionAudio()
                udpControl?.stop()
                udpControl = null
                stateMachine?.stop()
                updateSessionFile("stopped")
                snapshot("session stopped")
            }
        }
    }

    fun pauseListening() {
        stateMachine?.requestPause()
        pairing?.clearPending()
        log("listening paused; pending ARM cleared")
        snapshot("paused")
    }

    fun resumeListening() {
        detector?.reset(totalFramesRead.get())
        stateMachine?.resume()
        log("listening resumed")
        snapshot("resumed")
    }

    fun stopAndSave() {
        running.set(false)
        try { record?.stop() } catch (_: Throwable) {}
        udpControl?.stop()
        log("safe stop requested")
    }

    fun isRunning(): Boolean = running.get()
    fun isTestRunning(): Boolean = testRunning.get()

    fun startRepeatedPlaybackTest() {
        if (running.get() || !testRunning.compareAndSet(false, true)) return
        thread(name = "AVTwin-C2-x20") {
            Process.setThreadPriority(Process.THREAD_PRIORITY_AUDIO)
            var p: PersistentC2Player? = null
            try {
                val manager = context.getSystemService(Context.AUDIO_SERVICE) as AudioManager
                p = PersistentC2Player(
                    samples = c2Signal.samples,
                    preferredOutput = AudioRouteUtil.preferredBuiltinSpeaker(manager),
                    onLog = { onStatus("C2 x20 TEST\n$it") }
                )
                val prep = p.prepare()
                val lines = ArrayList<String>()
                lines += "C2 x20 persistent AudioTrack test"
                lines += "state=${prep.trackState} performanceMode=${prep.performanceMode} frames=${prep.writtenFrames}"
                for (i in 1..20) {
                    val v = p.playAndVerify()
                    val pass = v.playbackVerified
                    lines += "[$i/20] ${if (pass) "PASS" else "FAIL"} head=${v.playbackHeadBefore}->${v.playbackHeadAfter} ts=${v.audioTimestampValid} route=${v.actualOutputRoute}"
                    onStatus(lines.joinToString("\n"))
                    if (i < 20) p.rearmForNextPlayback()
                }
                val passCount = lines.count { it.contains("] PASS") }
                lines += "RESULT: $passCount/20 hardware-verified"
                onStatus(lines.joinToString("\n"))
            } catch (t: Throwable) {
                onStatus("C2 x20 TEST ERROR: ${t.javaClass.simpleName}: ${t.message}")
            } finally {
                p?.release()
                testRunning.set(false)
            }
        }
    }

    fun testUdp(host: String, port: Int) {
        thread(name = "AVTwin-UDP-Test") {
            val eventId = UUID.randomUUID().toString()
            val json = JsonWire.obj(
                "type" to "udp_test",
                "protocol_version" to 1,
                "android_event_id" to eventId,
                "plays_audio" to false
            )
            val report = UdpReporter.sendRepeated(host, port, json, repeats = 1)
            onStatus(
                "UDP TEST ${if (report.success) "PASS" else "FAIL"}\n" +
                    "target=$host:$port\nandroid_event_id=$eventId\nNo audio was played."
            )
        }
    }

    private fun runContinuousSession(config: SessionConfig) {
        resetCounters()
        val sid = UUID.randomUUID().toString()
        sessionId = sid
        pairing = ArmPairingManager(sid)
        stateMachine = ContinuousResponderStateMachine(config.cooldownMs.toLong() * SAMPLE_RATE / 1000L)
        detector = StreamingC1Detector(
            fullTemplate = c1Signal.samples,
            threshold = C1_THRESHOLD,
            pretrigger = C1_PRETRIGGER,
            useHighFrequencyGate = c1Signal.isBuiltIn,
            minBandRatio = C1_MIN_BAND_RATIO,
            detectionMs = 60
        )
        detector!!.reset(0L)

        storage = SafSessionStorage(context, config.resultTreeUri, sid, c1Signal, c2Signal, config.saveDebugAudio)
        storage!!.start(sessionJson("starting"))
        log("session_id=$sid")
        log("C1 source SHA256=${c1Signal.sourceSha256}; internal=${c1Signal.internalPcmSha256}")
        log("C2 source SHA256=${c2Signal.sourceSha256}; internal=${c2Signal.internalPcmSha256}")

        preparePersistentAudio()
        startUdpControl(config)

        val r = record ?: error("AudioRecord not prepared")
        r.startRecording()
        require(r.recordingState == AudioRecord.RECORDSTATE_RECORDING) { "AudioRecord did not enter RECORDING state" }
        totalFramesRead.set(0L)
        latestCaptureTimestamp = null
        stateMachine!!.start()
        log("LISTENING started; AudioRecord remains open for the whole session")
        snapshot("waiting for ARM/C1")

        val hop = ShortArray(HOP_SAMPLES)
        var timestampPollCounter = 0
        while (running.get()) {
            val startFrame = totalFramesRead.get()
            val n = try { r.read(hop, 0, hop.size, AudioRecord.READ_BLOCKING) } catch (_: Throwable) { -1 }
            if (n <= 0) {
                if (!running.get()) break
                continue
            }
            val endFrame = totalFramesRead.addAndGet(n.toLong())

            timestampPollCounter++
            if (timestampPollCounter >= 5) {
                timestampPollCounter = 0
                updateCaptureTimestamp(r, endFrame)
                inputRoute = AudioRouteUtil.describe(try { r.routedDevice } catch (_: Throwable) { null })
            }

            if (stateMachine!!.updateCaptureSample(endFrame)) {
                detector!!.reset(endFrame)
                log("COOLDOWN complete -> LISTENING")
                snapshot("cooldown complete")
            }

            if (stateMachine!!.isListening()) {
                val d = detector!!.process(hop, n, startFrame)
                if (d != null) {
                    if (d.detected) {
                        val t2 = d.t2Sample ?: continue
                        if (stateMachine!!.acceptC1()) {
                            val claim = pairing!!.claimNext(SystemClock.elapsedRealtime())
                            currentMeasurementId = claim.measurementId
                            val routeGenAtDetect = routeGeneration.get()
                            log("C1 detected: t2_sample=$t2 score=${fmt(d.score)} measurement=${claim.measurementId} pairing=${claim.pairingMode}")
                            storage?.appendEvent(
                                JsonWire.obj(
                                    "type" to "c1_detected",
                                    "session_id" to claim.sessionId,
                                    "measurement_id" to claim.measurementId,
                                    "pairing_mode" to claim.pairingMode,
                                    "t2_sample" to t2,
                                    "candidate_peak_sample" to d.candidateSample,
                                    "c1_score" to d.score,
                                    "detection_completed_at_sample" to d.detectionCompletedAtSample,
                                    "detection_latency_samples" to (d.detectionCompletedAtSample - t2),
                                    "band_ratio" to d.bandRatio
                                )
                            )
                            launchResponse(d, claim, routeGenAtDetect)
                        }
                    } else {
                        c1Rejected++
                        storage?.appendEvent(
                            JsonWire.obj(
                                "type" to "c1_rejected",
                                "candidate_peak_sample" to d.candidateSample,
                                "score" to d.score,
                                "reason" to d.rejectionReason,
                                "band_ratio" to d.bandRatio
                            )
                        )
                        snapshot("C1 candidate rejected: ${d.rejectionReason}")
                    }
                }
            } else {
                detector!!.appendOnly(hop, n, startFrame)
            }
        }
    }

    private fun launchResponse(
        detection: StreamingC1Detector.Detection,
        claim: PairingClaim,
        routeGenAtDetect: Long
    ) {
        responseThread = thread(name = "AVTwin-C2-Response") {
            Process.setThreadPriority(Process.THREAD_PRIORITY_AUDIO)
            val t2 = detection.t2Sample ?: return@thread
            var verification: PersistentC2Player.PlaybackVerification? = null
            var playbackError: String? = null
            val decisionTimeNs = System.nanoTime() // diagnostic scheduler time, not an audio timestamp
            try {
                stateMachine?.c2Scheduled()
                log("C2_SCHEDULED measurement=${claim.measurementId}")
                verification = player!!.playAndVerify { playNs ->
                    stateMachine?.c2Playing()
                    log("C2_PLAYING play_call_time_ns=$playNs (diagnostic only)")
                }
            } catch (t: Throwable) {
                playbackError = "${t.javaClass.simpleName}: ${t.message}"
                c2Failures++
                log("C2 playback failure: $playbackError")
            }

            try { stateMachine?.reporting() } catch (_: Throwable) {}
            val v = verification
            if (v != null) {
                outputRoute = v.actualOutputRoute
                if (v.playbackVerified) successResponses++ else c2Failures++
            }

            val captureTs = latestCaptureTimestamp
            val playbackTs = if (v?.audioTimestampValid == true && v.firstValidAudioTimestampNs != null && v.firstValidAudioFramePosition != null) {
                PlaybackAudioTimestamp(v.firstValidAudioFramePosition, v.firstValidAudioTimestampNs)
            } else null
            val r = record
            val inputDevice = try { r?.routedDevice } catch (_: Throwable) { null }
            val routeStable = routeGeneration.get() == routeGenAtDetect && !audioFocusLost
            val timing = ReplyTimingMapper.mapToCaptureTimeline(
                t2Sample = t2,
                sampleRate = SAMPLE_RATE,
                playback = playbackTs,
                capture = captureTs,
                inputSampleRate = r?.sampleRate ?: -1,
                outputSampleRate = v?.outputSampleRate ?: player?.outputSampleRate() ?: -1,
                routeStable = routeStable,
                inputRouteTrusted = AudioRouteUtil.trustedInput(inputDevice),
                outputRouteTrusted = PersistentC2Player.isTrustedBuiltinSpeaker(v?.actualOutputRouteType)
            )
            lastT3Precise = timing.t3Precise
            lastReplyDelaySamples = timing.replyDelaySamples

            val eventId = ReplyEventId().value
            val playCallNs = v?.playCallTimeNs
            val replyJson = JsonWire.obj(
                "type" to "reply_timing",
                "protocol_version" to 1,
                "session_id" to claim.sessionId,
                "measurement_id" to claim.measurementId,
                "android_event_id" to eventId,
                "pairing_mode" to claim.pairingMode,
                "t3_precise" to timing.t3Precise,
                "t2_sample" to t2,
                "t3_sample" to timing.t3Sample,
                "reply_delay_samples" to timing.replyDelaySamples,
                "sample_rate" to SAMPLE_RATE,
                "c1_score" to detection.score,
                "c1_detected" to true,
                "c2_started" to (v != null),
                "playback_verified" to (v?.playbackVerified ?: false),
                "error" to (playbackError ?: timing.reason),
                "decision_time_ns" to decisionTimeNs,
                "play_call_time_ns" to playCallNs,
                "first_valid_audio_timestamp_ns" to v?.firstValidAudioTimestampNs,
                "first_valid_audio_frame_position" to v?.firstValidAudioFramePosition,
                "playback_frame_zero_nano_time" to timing.playbackFrameZeroNanoTime,
                "detection_completed_at_sample" to detection.detectionCompletedAtSample,
                "detection_latency_samples" to (detection.detectionCompletedAtSample - t2),
                "audio_track_head_before" to v?.playbackHeadBefore,
                "audio_track_head_after" to v?.playbackHeadAfter,
                "audio_track_timestamp_valid" to v?.audioTimestampValid,
                "audio_track_underruns" to v?.underrunCount,
                "input_route" to inputRoute,
                "output_route" to outputRoute,
                "route_stable" to routeStable,
                "t3_method" to if (timing.t3Precise) "AudioTrack+AudioRecord_MONOTONIC_frame_projection" else "unavailable"
            )
            storage?.appendEvent(replyJson)
            log(
                "REPORT measurement=${claim.measurementId} event=$eventId t3_precise=${timing.t3Precise} " +
                    "reply_delay_samples=${timing.replyDelaySamples ?: "null"} reason=${timing.reason ?: "none"}"
            )

            val config = currentConfig
            if (config != null) {
                val sendReport = UdpReporter.sendRepeated(config.linuxHost, config.resultPort, replyJson, repeats = 3)
                if (!sendReport.success) udpFailures++
                for (attempt in sendReport.attempts) {
                    storage?.appendEvent(
                        JsonWire.obj(
                            "type" to "udp_send",
                            "android_event_id" to eventId,
                            "target_ip" to config.linuxHost,
                            "target_port" to config.resultPort,
                            "attempt" to attempt.attempt,
                            "diagnostic_wall_time" to attempt.diagnosticWallTime,
                            "diagnostic_nano_time" to attempt.diagnosticNanoTime,
                            "success" to attempt.success,
                            "error" to attempt.error
                        )
                    )
                }
            }

            if (config?.saveDebugAudio == true) {
                detector?.debugWindow(t2, SAMPLE_RATE / 10, SAMPLE_RATE * 35 / 100)?.let {
                    storage?.saveDebugWindow("m${claim.measurementId}_c1_window.wav", it)
                }
                storage?.saveDebugWindow("m${claim.measurementId}_c2_reference.wav", c2Signal.samples)
            }

            try {
                player?.rearmForNextPlayback()
            } catch (t: Throwable) {
                c2Failures++
                log("C2 rearm failure: ${t.javaClass.simpleName}: ${t.message}; stopping session for safety")
                running.set(false)
            }

            val currentCapture = totalFramesRead.get()
            try { stateMachine?.enterCooldown(currentCapture) } catch (_: Throwable) {}
            updateSessionFile("running")
            snapshot(if (timing.t3Precise) "reply timing ready" else "t3_precise=false: ${timing.reason}")
        }
    }

    private fun preparePersistentAudio() {
        val manager = context.getSystemService(Context.AUDIO_SERVICE) as AudioManager
        audioManager = manager
        requestAudioFocus(manager)

        val minBuffer = AudioRecord.getMinBufferSize(SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
        require(minBuffer > 0) { "48 kHz mono PCM16 AudioRecord unsupported" }
        val source = if (manager.getProperty(AudioManager.PROPERTY_SUPPORT_AUDIO_SOURCE_UNPROCESSED) == "true") {
            MediaRecorder.AudioSource.UNPROCESSED
        } else MediaRecorder.AudioSource.VOICE_RECOGNITION
        val r = AudioRecord.Builder()
            .setAudioSource(source)
            .setAudioFormat(
                AudioFormat.Builder()
                    .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                    .setSampleRate(SAMPLE_RATE)
                    .setChannelMask(AudioFormat.CHANNEL_IN_MONO)
                    .build()
            )
            .setBufferSizeInBytes(maxOf(minBuffer * 8, HOP_SAMPLES * 32))
            .build()
        require(r.state == AudioRecord.STATE_INITIALIZED) { "AudioRecord initialization failed" }
        val mic = AudioRouteUtil.preferredBuiltinMic(manager)
        if (mic != null) log("preferred built-in mic accepted=${r.setPreferredDevice(mic)} ${AudioRouteUtil.describe(mic)}")
        disablePreprocessing(r.audioSessionId)
        recordRouteListener = AudioRouting.OnRoutingChangedListener { router ->
            routeGeneration.incrementAndGet()
            inputRoute = AudioRouteUtil.describe(router.routedDevice)
            log("input route changed: $inputRoute")
        }.also { r.addOnRoutingChangedListener(it, null) }
        record = r

        val speaker = AudioRouteUtil.preferredBuiltinSpeaker(manager)
        player = PersistentC2Player(
            samples = c2Signal.samples,
            preferredOutput = speaker,
            onLog = { log(it) },
            onRouteChanged = {
                routeGeneration.incrementAndGet()
                outputRoute = it
                log(it)
            }
        )
        val prep = player!!.prepare()
        outputRoute = prep.preferredRoute
        log(
            "C2 persistent track ready: state=${prep.trackState} performanceMode=${prep.performanceMode} " +
                "frames=${prep.writtenFrames} bytes=${prep.writtenBytes} preferred=${prep.preferredRoute}"
        )
    }

    private fun startUdpControl(config: SessionConfig) {
        udpControl = UdpControlServer(
            port = config.controlPort,
            onArm = { command, source ->
                val result = pairing?.accept(command, SystemClock.elapsedRealtime())
                    ?: ArmPairingManager.AcceptResult(false, "session_not_ready")
                log("ARM from $source measurement=${command.measurementId} accepted=${result.accepted} reason=${result.reason}")
                storage?.appendEvent(
                    JsonWire.obj(
                        "type" to "arm_received",
                        "protocol_version" to command.protocolVersion,
                        "session_id" to command.sessionId,
                        "measurement_id" to command.measurementId,
                        "source" to source,
                        "accepted" to result.accepted,
                        "reason" to result.reason
                    )
                )
                snapshot("ARM ${if (result.accepted) "accepted" else "rejected"}: ${result.reason}")
            },
            onMalformed = { raw, source ->
                log("non-ARM UDP from $source ignored")
                storage?.appendEvent(JsonWire.obj("type" to "udp_control_ignored", "source" to source, "raw" to raw.take(2048)))
            },
            onError = { error -> log(error); udpFailures++; snapshot(error) }
        ).also { it.start() }
        log("UDP ARM listener started on 0.0.0.0:${config.controlPort}")
    }

    private fun updateCaptureTimestamp(r: AudioRecord, observedReadFrames: Long) {
        val ts = AudioTimestamp()
        val rc = try { r.getTimestamp(ts, AudioTimestamp.TIMEBASE_MONOTONIC) } catch (_: Throwable) { -1 }
        if (rc == 0 && ts.nanoTime > 0L && ts.framePosition >= 0L) {
            latestCaptureTimestamp = CaptureAudioTimestamp(ts.framePosition, ts.nanoTime, observedReadFrames)
        }
    }

    private fun requestAudioFocus(manager: AudioManager) {
        val attributes = AudioAttributes.Builder()
            .setUsage(AudioAttributes.USAGE_MEDIA)
            .setContentType(AudioAttributes.CONTENT_TYPE_MUSIC)
            .build()
        focusRequest = AudioFocusRequest.Builder(AudioManager.AUDIOFOCUS_GAIN_TRANSIENT_EXCLUSIVE)
            .setAudioAttributes(attributes)
            .setOnAudioFocusChangeListener { change ->
                audioFocusLost = change != AudioManager.AUDIOFOCUS_GAIN
                log("audio focus change=$change lost=$audioFocusLost")
                if (audioFocusLost) routeGeneration.incrementAndGet()
            }
            .build()
        val result = manager.requestAudioFocus(focusRequest!!)
        audioFocusLost = result != AudioManager.AUDIOFOCUS_REQUEST_GRANTED
        log("audio focus request result=$result")
    }

    private fun disablePreprocessing(sessionId: Int) {
        try { if (AcousticEchoCanceler.isAvailable()) aec = AcousticEchoCanceler.create(sessionId)?.also { it.enabled = false } } catch (_: Throwable) {}
        try { if (NoiseSuppressor.isAvailable()) ns = NoiseSuppressor.create(sessionId)?.also { it.enabled = false } } catch (_: Throwable) {}
        try { if (AutomaticGainControl.isAvailable()) agc = AutomaticGainControl.create(sessionId)?.also { it.enabled = false } } catch (_: Throwable) {}
    }

    private fun cleanupSessionAudio() {
        val r = record
        if (r != null) {
            try { recordRouteListener?.let { r.removeOnRoutingChangedListener(it) } } catch (_: Throwable) {}
            try { r.stop() } catch (_: Throwable) {}
            try { r.release() } catch (_: Throwable) {}
        }
        recordRouteListener = null
        record = null
        player?.release()
        player = null
        try { aec?.release() } catch (_: Throwable) {}
        try { ns?.release() } catch (_: Throwable) {}
        try { agc?.release() } catch (_: Throwable) {}
        aec = null; ns = null; agc = null
        val manager = audioManager
        val focus = focusRequest
        if (manager != null && focus != null) try { manager.abandonAudioFocusRequest(focus) } catch (_: Throwable) {}
        focusRequest = null
        audioManager = null
    }

    private fun resetCounters() {
        successResponses = 0; c1Rejected = 0; c2Failures = 0; udpFailures = 0
        lastReplyDelaySamples = null; lastT3Precise = false; currentMeasurementId = null
        inputRoute = "unavailable"; outputRoute = "unavailable"; routeGeneration.set(0L)
    }

    private fun snapshot(note: String = "") {
        val sm = stateMachine
        onSnapshot(
            SessionSnapshot(
                state = sm?.state?.name ?: if (running.get()) "STARTING" else "STOPPED",
                sessionId = sessionId,
                measurementId = currentMeasurementId,
                pendingArmMeasurementId = pairing?.pendingMeasurementId(),
                successResponses = successResponses,
                c1Rejected = c1Rejected,
                c2Failures = c2Failures,
                udpFailures = udpFailures,
                lastReplyDelaySamples = lastReplyDelaySamples,
                lastT3Precise = lastT3Precise,
                inputRoute = inputRoute,
                outputRoute = outputRoute,
                note = note
            )
        )
    }

    private fun sessionJson(status: String): String = JsonWire.obj(
        "protocol_version" to 1,
        "session_id" to sessionId,
        "status" to status,
        "sample_rate" to SAMPLE_RATE,
        "c1_name" to c1Signal.name,
        "c1_source_sha256" to c1Signal.sourceSha256,
        "c1_internal_pcm_sha256" to c1Signal.internalPcmSha256,
        "c2_name" to c2Signal.name,
        "c2_source_sha256" to c2Signal.sourceSha256,
        "c2_internal_pcm_sha256" to c2Signal.internalPcmSha256,
        "success_responses" to successResponses,
        "c1_rejected" to c1Rejected,
        "c2_failures" to c2Failures,
        "udp_failures" to udpFailures,
        "last_reply_delay_samples" to lastReplyDelaySamples,
        "last_t3_precise" to lastT3Precise,
        "input_route" to inputRoute,
        "output_route" to outputRoute
    )

    private fun updateSessionFile(status: String) {
        try { storage?.updateSessionJson(sessionJson(status)) } catch (t: Throwable) { onStatus("SAVE ERROR: ${t.message}") }
    }

    private fun log(message: String) {
        onStatus(message)
        try { storage?.appendLog(message) } catch (_: Throwable) {}
    }

    private fun fmt(v: Double, digits: Int = 3): String = "% .${digits}f".format(Locale.US, v).trim()
}
