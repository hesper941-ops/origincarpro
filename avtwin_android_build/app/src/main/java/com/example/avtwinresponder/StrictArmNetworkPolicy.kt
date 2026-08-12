package com.example.avtwinresponder

object StrictArmNetworkPolicy {
    @Volatile private var expectedLinuxHost: String = ""

    fun setExpectedLinuxHost(host: String) {
        expectedLinuxHost = host.trim()
    }

    fun expectedHost(): String = expectedLinuxHost

    fun sourceAllowed(sourceHost: String): Boolean {
        val expected = expectedLinuxHost
        if (expected.isBlank()) return false
        return sourceHost.trim() == expected
    }
}
