package com.example.avtwinresponder

object SessionJournal {
    fun encodeLine(jsonObject: String): String = jsonObject
        .replace("\r", "")
        .replace("\n", "")
        .trim() + "\n"

    fun recoverCompleteObjects(text: String): List<String> = text
        .lineSequence()
        .map { it.trim() }
        .filter { it.startsWith("{") && it.endsWith("}") }
        .toList()
}

object PersistedTreePermissionPolicy {
    fun restorable(uriMatches: Boolean, hasRead: Boolean, hasWrite: Boolean): Boolean =
        uriMatches && hasRead && hasWrite
}
