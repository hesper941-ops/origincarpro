package com.example.avtwinresponder

import java.net.Inet4Address
import java.net.NetworkInterface
import java.util.Collections

object LocalNetworkInfo {
    data class Address(val interfaceName: String, val hostAddress: String)

    fun ipv4Addresses(): List<Address> {
        return try {
            Collections.list(NetworkInterface.getNetworkInterfaces())
                .flatMap { iface ->
                    Collections.list(iface.inetAddresses)
                        .filterIsInstance<Inet4Address>()
                        .filter { !it.isLoopbackAddress && !it.isLinkLocalAddress }
                        .map { Address(iface.name ?: "?", it.hostAddress ?: "?") }
                }
                .sortedWith(compareBy<Address> { if (it.interfaceName.startsWith("wlan")) 0 else 1 }.thenBy { it.interfaceName })
        } catch (_: Throwable) {
            emptyList()
        }
    }

    fun preferredLanIpv4(): String? = ipv4Addresses().firstOrNull()?.hostAddress

    fun display(): String {
        val addresses = ipv4Addresses()
        if (addresses.isEmpty()) return "unavailable"
        return addresses.joinToString(", ") { "${it.interfaceName}=${it.hostAddress}" }
    }
}
