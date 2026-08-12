package com.example.avtwinresponder

class ContinuousResponderStateMachine(
    private val cooldownSamples: Long
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

    var state: State = State.STOPPED
        private set

    var cooldownUntilSample: Long = Long.MIN_VALUE
        private set

    private var pauseRequested = false

    @Synchronized
    fun start() {
        require(state == State.STOPPED) { "start from $state" }
        pauseRequested = false
        state = State.LISTENING
    }

    @Synchronized
    fun requestPause() {
        pauseRequested = true
        if (state == State.LISTENING || state == State.COOLDOWN) state = State.PAUSED
    }

    @Synchronized
    fun resume() {
        pauseRequested = false
        if (state == State.PAUSED) state = State.LISTENING
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
        cooldownUntilSample = currentCaptureSample + cooldownSamples
        state = if (pauseRequested) State.PAUSED else State.COOLDOWN
    }

    @Synchronized
    fun updateCaptureSample(currentCaptureSample: Long): Boolean {
        if (state == State.COOLDOWN && currentCaptureSample >= cooldownUntilSample) {
            state = if (pauseRequested) State.PAUSED else State.LISTENING
            return state == State.LISTENING
        }
        return false
    }

    @Synchronized
    fun stop() {
        pauseRequested = false
        state = State.STOPPED
    }

    @Synchronized
    fun isListening(): Boolean = state == State.LISTENING
}
