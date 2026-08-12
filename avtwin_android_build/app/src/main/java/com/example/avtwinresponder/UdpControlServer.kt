package com.example.avtwinresponder

import java.net.DatagramPacket
import java.net.DatagramSocket
import java.net.SocketTimeoutException
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.concurrent.thread

class UdpControlServer(
    private val port: Int,
    private val onArm: (ArmCommand, String) -> Unit,
    private val onMalformed: (String, String) -> Unit,
    private val onError: (String) -> Unit
) {
    private val running = AtomicBoolean(false)
    private var socket: DatagramSocket? = null
    private var worker: Thread? = null

    fun start() {
        if (!running.compareAndSet(false, true)) return
        worker = thread(name = "AVTwin-UDP-Control", isDaemon = true) {
            try {
                DatagramSocket(port).use { s ->
                    socket = s
                    s.soTimeout = 500
                    val buf = ByteArray(8192)
                    while (running.get()) {
                        try {
                            val packet = DatagramPacket(buf, buf.size)
                            s.receive(packet)
                            val raw = String(packet.data, packet.offset, packet.length, Charsets.UTF_8)
                            val sourceHost = packet.address.hostAddress ?: ""
                            val source = "$sourceHost:${packet.port}"

                            // Strict formal-experiment mode: only the configured Linux host may ARM
                            // the responder. This policy is orchestration only; it is never used as
                            // an acoustic timing source.
                            if (!StrictArmNetworkPolicy.sourceAllowed(sourceHost)) {
                                onMalformed(raw, "$source [STRICT_ARM_SOURCE_REJECTED expected=${StrictArmNetworkPolicy.expectedHost()}]")
                                continue
                            }

                            val arm = ArmCommand.parse(raw)
                            if (arm != null) onArm(arm, source) else onMalformed(raw, source)
                        } catch (_: SocketTimeoutException) {
                            // periodic wakeup for safe stop
                        }
                    }
                }
            } catch (t: Throwable) {
                if (running.get()) onError("UDP control listener error: ${t.javaClass.simpleName}: ${t.message}")
            } finally {
                socket = null
                running.set(false)
            }
        }
    }

    fun stop() {
        running.set(false)
        try { socket?.close() } catch (_: Throwable) {}
        try { worker?.join(700) } catch (_: Throwable) {}
        worker = null
    }
}
