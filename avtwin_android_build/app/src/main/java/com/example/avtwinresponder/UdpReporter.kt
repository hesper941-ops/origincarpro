package com.example.avtwinresponder

import java.net.DatagramPacket
import java.net.DatagramSocket
import java.net.InetAddress
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

object UdpReporter {
    data class Attempt(
        val attempt: Int,
        val diagnosticWallTime: String,
        val diagnosticNanoTime: Long,
        val success: Boolean,
        val error: String?
    )

    data class SendReport(
        val targetHost: String,
        val targetPort: Int,
        val attempts: List<Attempt>
    ) {
        val success: Boolean get() = attempts.any { it.success }
    }

    fun interface Transport {
        fun send(host: String, port: Int, json: String)
    }

    private val datagramTransport = Transport { host, port, json -> sendOnce(host, port, json) }

    fun send(host: String, port: Int, json: String) {
        if (host.isBlank()) return
        datagramTransport.send(host, port, json)
    }

    internal fun productionAttemptCount(requestedRepeats: Int): Int = 1

    /**
     * Production result reporting is deliberately single-shot.
     *
     * Acoustic C1 -> C2 is the handshake. UDP only carries timing metadata after the
     * acoustic response has already happened, so duplicating the same reply_timing packet
     * can look like multiple responder events on a Linux controller that has not yet added
     * event-id de-duplication. The repeats argument is kept for source/protocol compatibility,
     * but the production path sends exactly one datagram.
     *
     * If reliable delivery is needed later, use an explicit ACK protocol rather than blind
     * duplicate sends.
     */
    fun sendRepeated(
        host: String,
        port: Int,
        json: String,
        repeats: Int = 1,
        spacingMs: Long = 0L
    ): SendReport = sendRepeatedWithTransport(
        host = host,
        port = port,
        json = json,
        repeats = productionAttemptCount(repeats),
        spacingMs = 0L,
        transport = datagramTransport
    )

    /** Testable retry primitive retained for future ACK/retry experiments. */
    internal fun sendRepeatedWithTransport(
        host: String,
        port: Int,
        json: String,
        repeats: Int,
        spacingMs: Long,
        transport: Transport,
        sleeper: (Long) -> Unit = { Thread.sleep(it) }
    ): SendReport {
        val attempts = ArrayList<Attempt>()
        if (host.isBlank()) {
            attempts += Attempt(1, wallNow(), System.nanoTime(), false, "blank_host")
            return SendReport(host, port, attempts)
        }
        val count = repeats.coerceAtLeast(1)
        for (i in 1..count) {
            val diagNs = System.nanoTime() // diagnostic only, not acoustic timing
            try {
                transport.send(host, port, json)
                attempts += Attempt(i, wallNow(), diagNs, true, null)
            } catch (t: Throwable) {
                attempts += Attempt(i, wallNow(), diagNs, false, "${t.javaClass.simpleName}: ${t.message}")
            }
            if (i < count && spacingMs > 0L) sleeper(spacingMs)
        }
        return SendReport(host, port, attempts)
    }

    private fun sendOnce(host: String, port: Int, json: String) {
        DatagramSocket().use { socket ->
            val bytes = json.toByteArray(Charsets.UTF_8)
            val packet = DatagramPacket(bytes, bytes.size, InetAddress.getByName(host), port)
            socket.send(packet)
        }
    }

    private fun wallNow(): String =
        SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSSZ", Locale.US).format(Date())
}
