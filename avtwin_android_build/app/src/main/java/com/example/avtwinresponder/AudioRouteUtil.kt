package com.example.avtwinresponder

import android.media.AudioDeviceInfo
import android.media.AudioManager

object AudioRouteUtil {
    fun preferredBuiltinMic(manager: AudioManager): AudioDeviceInfo? =
        manager.getDevices(AudioManager.GET_DEVICES_INPUTS).firstOrNull { it.type == AudioDeviceInfo.TYPE_BUILTIN_MIC }

    fun preferredBuiltinSpeaker(manager: AudioManager): AudioDeviceInfo? =
        manager.getDevices(AudioManager.GET_DEVICES_OUTPUTS).firstOrNull { it.type == AudioDeviceInfo.TYPE_BUILTIN_SPEAKER }

    fun describe(device: AudioDeviceInfo?): String {
        if (device == null) return "unavailable"
        val rates = device.sampleRates.joinToString(",").ifBlank { "unspecified" }
        return "${typeName(device.type)} id=${device.id} ${device.productName} rates=$rates"
    }

    fun trustedInput(device: AudioDeviceInfo?): Boolean = device?.type == AudioDeviceInfo.TYPE_BUILTIN_MIC
    fun trustedOutput(device: AudioDeviceInfo?): Boolean = device?.type == AudioDeviceInfo.TYPE_BUILTIN_SPEAKER

    fun typeName(type: Int): String = when (type) {
        AudioDeviceInfo.TYPE_BUILTIN_MIC -> "BUILTIN_MIC"
        AudioDeviceInfo.TYPE_BUILTIN_SPEAKER -> "BUILTIN_SPEAKER"
        AudioDeviceInfo.TYPE_BUILTIN_EARPIECE -> "BUILTIN_EARPIECE"
        AudioDeviceInfo.TYPE_BLUETOOTH_A2DP -> "BLUETOOTH_A2DP"
        AudioDeviceInfo.TYPE_BLUETOOTH_SCO -> "BLUETOOTH_SCO"
        AudioDeviceInfo.TYPE_USB_DEVICE -> "USB_DEVICE"
        AudioDeviceInfo.TYPE_USB_HEADSET -> "USB_HEADSET"
        AudioDeviceInfo.TYPE_WIRED_HEADSET -> "WIRED_HEADSET"
        AudioDeviceInfo.TYPE_WIRED_HEADPHONES -> "WIRED_HEADPHONES"
        else -> "TYPE_$type"
    }
}
