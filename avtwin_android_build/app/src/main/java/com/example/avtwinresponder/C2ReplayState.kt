package com.example.avtwinresponder

internal class C2ReplayStateMachine {
    enum class State {
        NEW,
        TRACK_INITIALIZED,
        BUFFER_LOADED,
        PLAY_ISSUED,
        PLAYBACK_VERIFIED,
        PLAYBACK_UNVERIFIED,
        RELEASED
    }

    var state: State = State.NEW
        private set

    var completedCycles: Int = 0
        private set

    fun trackInitialized() {
        require(state == State.NEW) { "trackInitialized from $state" }
        state = State.TRACK_INITIALIZED
    }

    fun bufferLoaded() {
        require(
            state == State.TRACK_INITIALIZED ||
                state == State.PLAYBACK_VERIFIED ||
                state == State.PLAYBACK_UNVERIFIED
        ) { "bufferLoaded from $state" }
        state = State.BUFFER_LOADED
    }

    fun playIssued() {
        require(state == State.BUFFER_LOADED) { "playIssued from $state" }
        state = State.PLAY_ISSUED
    }

    fun verificationFinished(verified: Boolean) {
        require(state == State.PLAY_ISSUED) { "verificationFinished from $state" }
        completedCycles++
        state = if (verified) State.PLAYBACK_VERIFIED else State.PLAYBACK_UNVERIFIED
    }

    fun release() {
        state = State.RELEASED
    }
}
