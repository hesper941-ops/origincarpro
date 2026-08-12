package com.example.avtwinresponder

import android.content.ContentResolver
import android.content.Context
import android.net.Uri
import android.provider.DocumentsContract
import android.provider.OpenableColumns
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.UUID

class SafSessionStorage(
    private val context: Context,
    private val treeUri: Uri,
    private val sessionId: String,
    private val c1: ProbeSignal,
    private val c2: ProbeSignal,
    private val saveDebugAudio: Boolean
) {
    data class Handles(
        val sessionDir: Uri,
        val events: Uri,
        val logs: Uri,
        val sessionJson: Uri,
        val audioDir: Uri?,
        val probesDir: Uri
    )

    private val resolver: ContentResolver = context.contentResolver
    private var handles: Handles? = null
    private val eventsCache = StringBuilder()
    private val logsCache = StringBuilder()

    fun start(initialSessionJson: String): Handles {
        val validation = validateTree(context, treeUri)
        require(validation.first) { "Selected result directory is not writable: ${validation.second}; select it again" }
        val root = documentUriForTree(treeUri)
        val stamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        val safeSession = sessionId.replace(Regex("[^A-Za-z0-9._-]"), "_")
        val sessionDir = createDir(root, "${stamp}_${safeSession}")
        val probesDir = createDir(sessionDir, "probes")
        val audioDir = if (saveDebugAudio) createDir(sessionDir, "audio") else null
        val events = createFile(sessionDir, "application/json", "events.jsonl")
        val logs = createFile(sessionDir, "text/plain", "logs.txt")
        val sessionJson = createFile(sessionDir, "application/json", "session.json")
        handles = Handles(sessionDir, events, logs, sessionJson, audioDir, probesDir)

        writeText(sessionJson, initialSessionJson)
        writeProbe(probesDir, "c1_used.wav", c1.samples)
        writeProbe(probesDir, "c2_used.wav", c2.samples)
        writeText(
            createFile(probesDir, "application/json", "probe_metadata.json"),
            JsonWire.obj(
                "c1_name" to c1.name,
                "c1_source_sha256" to c1.sourceSha256,
                "c1_internal_pcm_sha256" to c1.internalPcmSha256,
                "c1_source_channel" to c1.sourceChannel,
                "c2_name" to c2.name,
                "c2_source_sha256" to c2.sourceSha256,
                "c2_internal_pcm_sha256" to c2.internalPcmSha256,
                "c2_source_channel" to c2.sourceChannel,
                "sample_rate" to ProbeSignal.SAMPLE_RATE,
                "note" to "c1_used.wav/c2_used.wav are the exact internal 48 kHz mono PCM templates; source SHA256 refers to the user-selected source WAV bytes"
            )
        )
        appendLog("session storage created: $sessionDir")
        return handles!!
    }

    @Synchronized
    fun appendEvent(json: String) {
        val h = handles ?: return
        appendText(h.events, eventsCache, SessionJournal.encodeLine(json))
    }

    @Synchronized
    fun appendLog(line: String) {
        val h = handles ?: return
        val stamp = SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS", Locale.US).format(Date())
        appendText(h.logs, logsCache, "$stamp $line\n")
    }

    @Synchronized
    fun updateSessionJson(json: String) {
        val h = handles ?: return
        writeText(h.sessionJson, json)
    }

    @Synchronized
    fun saveDebugWindow(fileName: String, samples: ShortArray) {
        if (!saveDebugAudio || samples.isEmpty()) return
        val dir = handles?.audioDir ?: return
        val uri = createFile(dir, "audio/wav", sanitizeName(fileName))
        resolver.openOutputStream(uri, "wt")?.use { WavWriter.writeMonoPcm16(it, samples, ProbeSignal.SAMPLE_RATE) }
            ?: error("Cannot open debug WAV output")
    }

    fun selectedTreeLabel(): String = displayName(context, treeUri) ?: treeUri.toString()

    private fun writeProbe(parent: Uri, name: String, samples: ShortArray) {
        val uri = createFile(parent, "audio/wav", name)
        resolver.openOutputStream(uri, "wt")?.use { WavWriter.writeMonoPcm16(it, samples, ProbeSignal.SAMPLE_RATE) }
            ?: error("Cannot write $name")
    }

    private fun createDir(parent: Uri, name: String): Uri =
        DocumentsContract.createDocument(
            resolver,
            parent,
            DocumentsContract.Document.MIME_TYPE_DIR,
            sanitizeName(name)
        ) ?: error("Cannot create directory $name")

    private fun createFile(parent: Uri, mime: String, name: String): Uri =
        DocumentsContract.createDocument(resolver, parent, mime, sanitizeName(name))
            ?: error("Cannot create file $name")

    private fun writeText(uri: Uri, text: String) {
        resolver.openOutputStream(uri, "wt")?.use { out ->
            out.write(text.toByteArray(Charsets.UTF_8))
            out.flush()
        } ?: error("Cannot open $uri for writing")
    }

    private fun appendText(uri: Uri, cache: StringBuilder, text: String) {
        cache.append(text)
        try {
            resolver.openOutputStream(uri, "wa")?.use { out ->
                out.write(text.toByteArray(Charsets.UTF_8))
                out.flush()
            } ?: error("append stream unavailable")
        } catch (_: Throwable) {
            // Some document providers do not implement append mode. Rewriting the cached JSONL/log
            // preserves already completed records and keeps a truncated last line recoverable.
            writeText(uri, cache.toString())
        }
    }

    companion object {
        fun hasPersistedWritePermission(context: Context, treeUri: Uri): Boolean =
            context.contentResolver.persistedUriPermissions.any {
                PersistedTreePermissionPolicy.restorable(
                    uriMatches = it.uri == treeUri,
                    hasRead = it.isReadPermission,
                    hasWrite = it.isWritePermission
                )
            }

        fun validateTree(context: Context, treeUri: Uri): Pair<Boolean, String> {
            if (!hasPersistedWritePermission(context, treeUri)) {
                return false to "Persistent read/write permission is missing"
            }
            return try {
                val resolver = context.contentResolver
                val root = documentUriForTree(treeUri)
                val readable = resolver.query(
                    root,
                    arrayOf(DocumentsContract.Document.COLUMN_DOCUMENT_ID),
                    null,
                    null,
                    null
                )?.use { it.moveToFirst() } == true
                if (!readable) return false to "Selected directory is not readable"

                // Requirement: verify actual create/write/delete ability before a session starts.
                val probeName = ".avtwin_write_test_${UUID.randomUUID()}.tmp"
                val probe = DocumentsContract.createDocument(resolver, root, "application/octet-stream", probeName)
                    ?: return false to "Cannot create a test file in selected directory"
                try {
                    resolver.openOutputStream(probe, "wt")?.use { out ->
                        out.write(byteArrayOf(0x41))
                        out.flush()
                    } ?: return false to "Cannot write a test file in selected directory"
                } finally {
                    try { DocumentsContract.deleteDocument(resolver, probe) } catch (_: Throwable) {}
                }
                true to "OK (read/write/create verified)"
            } catch (t: Throwable) {
                false to (t.message ?: t.javaClass.simpleName)
            }
        }

        fun displayName(context: Context, uri: Uri): String? {
            return try {
                context.contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use { c ->
                    if (c.moveToFirst()) {
                        val i = c.getColumnIndex(OpenableColumns.DISPLAY_NAME)
                        if (i >= 0) c.getString(i) else null
                    } else null
                }
            } catch (_: Throwable) {
                null
            }
        }

        private fun documentUriForTree(treeUri: Uri): Uri =
            DocumentsContract.buildDocumentUriUsingTree(treeUri, DocumentsContract.getTreeDocumentId(treeUri))

        private fun sanitizeName(name: String): String =
            name.replace(Regex("[\\/:*?\"<>|]"), "_").take(120)
    }
}
