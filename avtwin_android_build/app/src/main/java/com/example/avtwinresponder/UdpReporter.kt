package com.example.avtwinresponder

import java.net.DatagramPacket
import java.net.DatagramSocket
import java.net.InetAddress

object UdpReporter {
    fun send(host: String, port: Int, json: String) {
        if (host.isBlank()) return
        DatagramSocket().use { socket ->
            val bytes = json.toByteArray(Charsets.UTF_8)
            val packet = DatagramPacket(bytes, bytes.size, InetAddress.getByName(host), port)
            socket.send(packet)
        }
    }
}
