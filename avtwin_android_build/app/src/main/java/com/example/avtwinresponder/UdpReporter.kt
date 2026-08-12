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

    fun send(host: String, port: Int, json: String) {
        if (host.isBlank()) return
        sendOnce(host, port, json)
    }

    fun sendRepeated(
        host: String,
        port: Int,
        json: String,
        repeats: Int = 3,
        spacingMs: Long = 35L
    ): SendReport {
        val attempts = ArrayList<Attempt>()
        if (host.isBlank()) {
            attempts += Attempt(1, wallNow(), System.nanoTime(), false, "blank_host")
            return SendReport(host, port, attempts)
        }
        for (i in 1..repeats.coerceAtLeast(1)) {
            val diagNs = System.nanoTime()
            try {
                sendOnce(host, port, json)
                attempts += Attempt(i, wallNow(), diagNs, true, null)
            } catch (t: Throwable) {
                attempts += Attempt(i, wallNow(), diagNs, false, "${t.javaClass.simpleName}: ${t.message}")
            }
            if (i < repeats) Thread.sleep(spacingMs)
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
