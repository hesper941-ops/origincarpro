package com.example.avtwinresponder

class ContinuousResponderStateMachine(
    private val cooldownSamples: Long,
    private val minimumRearmSamples: Long = DEFAULT_MINIMUM_REARM_SAMPLES
) {
    enum class State {
        STOPPED,
        LISTENING,
        C1_DETECTED,
        C2_SCHEDULED,
        C2_PLAYING,
        REPORTING,
        COOLDOWN,
        PAUSED
    }

    companion object {
        // This AV-Twin responder runs a fixed 48 kHz capture timeline.
        // 38,400 samples = 800 ms. This is deliberately longer than the 200 ms C1 probe
        // plus a normal reverberation tail, so one physical C1 cannot re-trigger C2 when the
        // detector is re-armed. Normal measurement rounds are much farther apart than this.
        const val DEFAULT_MINIMUM_REARM_SAMPLES = 38_400L
    }

    var state: State = State.STOPPED
        private set

    var cooldownUntilSample: Long = Long.MIN_VALUE
        private set

    private var pauseRequested = false
    private var pausedRequiresCooldown = false
    private var latestCaptureSample = 0L

    @Synchronized
    fun start() {
        require(state == State.STOPPED) { "start from $state" }
        pauseRequested = false
        pausedRequiresCooldown = false
        state = State.LISTENING
    }

    @Synchronized
    fun requestPause() {
        pauseRequested = true
        when (state) {
            State.LISTENING -> {
                pausedRequiresCooldown = false
                state = State.PAUSED
            }
            State.COOLDOWN -> {
                pausedRequiresCooldown = true
                state = State.PAUSED
            }
            else -> {
                // If a measurement is in flight, remain in that state. enterCooldown() will
                // transition to PAUSED while retaining the cooldown deadline.
            }
        }
    }

    @Synchronized
    fun resume() {
        pauseRequested = false
        if (state == State.PAUSED) {
            state = if (pausedRequiresCooldown && latestCaptureSample < cooldownUntilSample) {
                State.COOLDOWN
            } else {
                pausedRequiresCooldown = false
                State.LISTENING
            }
        }
    }

    @Synchronized
    fun acceptC1(): Boolean {
        if (state != State.LISTENING) return false
        state = State.C1_DETECTED
        return true
    }

    @Synchronized
    fun c2Scheduled() {
        require(state == State.C1_DETECTED) { "c2Scheduled from $state" }
        state = State.C2_SCHEDULED
    }

    @Synchronized
    fun c2Playing() {
        require(state == State.C2_SCHEDULED) { "c2Playing from $state" }
        state = State.C2_PLAYING
    }

    @Synchronized
    fun reporting() {
        require(state == State.C2_PLAYING || state == State.C2_SCHEDULED) { "reporting from $state" }
        state = State.REPORTING
    }

    @Synchronized
    fun enterCooldown(currentCaptureSample: Long) {
        require(state == State.REPORTING) { "enterCooldown from $state" }
        latestCaptureSample = currentCaptureSample
        val effectiveCooldown = maxOf(cooldownSamples, minimumRearmSamples)
        cooldownUntilSample = currentCaptureSample + effectiveCooldown
        if (pauseRequested) {
            pausedRequiresCooldown = true
            state = State.PAUSED
        } else {
            pausedRequiresCooldown = false
            state = State.COOLDOWN
        }
    }

    @Synchronized
    fun updateCaptureSample(currentCaptureSample: Long): Boolean {
        latestCaptureSample = currentCaptureSample
        if (state == State.COOLDOWN && currentCaptureSample >= cooldownUntilSample) {
            pausedRequiresCooldown = false
            state = if (pauseRequested) State.PAUSED else State.LISTENING
            return state == State.LISTENING
        }
        return false
    }

    @Synchronized
    fun stop() {
        pauseRequested = false
        pausedRequiresCooldown = false
        state = State.STOPPED
    }

    @Synchronized
    fun isListening(): Boolean = state == State.LISTENING
}
