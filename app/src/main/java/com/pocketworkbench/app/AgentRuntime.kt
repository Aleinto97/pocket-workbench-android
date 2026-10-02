package com.pocketworkbench.app

import org.json.JSONArray
import org.json.JSONObject

/**
 * Thin Kotlin side of the Rust agent runtime.
 *
 * Every call returns JSON text. A `{"ok":false,"error":...}` payload becomes an
 * exception here rather than a silent default, so a failed operation can never
 * look like a successful empty one.
 *
 * This object is not thread-safe by itself: the service owns one instance and
 * calls it from a single worker thread.
 */
class AgentRuntime private constructor() {

    interface EventListener {
        /** A JSONL batch of session events. */
        fun onEvents(jsonl: String)

        /** Runtime-level state change. */
        fun onServiceState(code: Int, detail: JSONObject)
    }

    @Volatile
    var listener: EventListener? = null

    @Volatile
    var lastFailure: String? = null
        private set

    val protocolVersion: Int
        get() {
            ensureLoaded()
            return nativeProtocolVersion()
        }

    fun capabilities(): JSONObject = call("capabilities") { nativeCapabilitiesJson() }

    fun openSession(sessionId: String, logPath: String, workspaceRoot: String, config: JSONObject): JSONObject =
        call("openSession") { nativeOpenSession(sessionId, logPath, workspaceRoot, config.toString(), nativeLibraryDir, cacheDir) }

    fun closeSession(sessionId: String): JSONObject =
        call("closeSession") { nativeCloseSession(sessionId) }

    fun sessionIds(): JSONArray = call("sessionIds") { nativeSessionIds() }.getJSONArray("ids")

    fun submit(sessionId: String, requestId: String, text: String): JSONObject =
        call("submit") { nativeSubmit(sessionId, requestId, text) }

    fun pump(sessionId: String, maxTurns: Int = 1): JSONObject =
        call("pump") { nativePump(sessionId, maxTurns) }

    fun cancel(sessionId: String): JSONObject =
        call("cancel") { nativeCancel(sessionId) }

    fun loadModel(sessionId: String, modelPath: String, backend: String, contextTokens: Int, threads: Int, useGpu: Boolean): JSONObject =
        call("loadModel") { nativeLoadModel(sessionId, modelPath, backend, contextTokens, threads, useGpu) }

    fun unloadModel(sessionId: String): JSONObject =
        call("unloadModel") { nativeUnloadModel(sessionId) }

    fun sessionState(sessionId: String): JSONObject =
        call("sessionState") { nativeSessionState(sessionId) }

    fun drainEvents(maxEvents: Int = 256): JSONObject =
        call("drainEvents") { nativeDrainEvents(maxEvents) }

    fun transcript(logPath: String): JSONObject =
        call("transcript") { nativeTranscript(logPath) }

    private fun call(name: String, block: () -> String): JSONObject {
        val failure = ensureLoaded()
        if (failure != null) throw AgentRuntimeException("The native agent runtime is unavailable: $failure")
        val raw = block()
        val value = try {
            JSONObject(raw)
        } catch (e: Exception) {
            throw AgentRuntimeException("$name returned a malformed payload")
        }
        if (!value.optBoolean("ok", false)) {
            val message = value.optString("error").ifBlank { "$name failed" }
            lastFailure = message
            throw AgentRuntimeException(message)
        }
        return value
    }

    private val nativeLibraryDir: String get() = nativeDir
    private val cacheDir: String get() = cachePath

    companion object {
        @Volatile private var instance: AgentRuntime? = null
        @Volatile private var loadError: String? = null
        @Volatile private var nativeDir: String = ""
        @Volatile private var cachePath: String = ""

        /**
         * One runtime per process. The UI process does not use it: the agent runs
         * in ":inference" so a native crash there cannot take the chat down.
         */
        fun obtain(nativeLibraryDir: String, cacheDir: String): AgentRuntime {
            nativeDir = nativeLibraryDir
            cachePath = cacheDir
            return instance ?: synchronized(this) {
                instance ?: AgentRuntime().also { instance = it }
            }
        }

        /** Loads the library once and reports the failure instead of crashing. */
        fun ensureLoaded(): String? {
            loadError?.let { return it }
            return try {
                System.loadLibrary("pocketinfer")
                null
            } catch (t: Throwable) {
                loadError = t.toString()
                loadError
            }
        }
    }

    private external fun nativeProtocolVersion(): Int
    private external fun nativeCapabilitiesJson(): String
    private external fun nativeOpenSession(sessionId: String, logPath: String, workspaceRoot: String, configJson: String, nativeDir: String, cacheDir: String): String
    private external fun nativeCloseSession(sessionId: String): String
    private external fun nativeSessionIds(): String
    private external fun nativeSubmit(sessionId: String, requestId: String, text: String): String
    private external fun nativePump(sessionId: String, maxTurns: Int): String
    private external fun nativeCancel(sessionId: String): String
    private external fun nativeLoadModel(sessionId: String, modelPath: String, backend: String, contextTokens: Int, threads: Int, useGpu: Boolean): String
    private external fun nativeUnloadModel(sessionId: String): String
    private external fun nativeSessionState(sessionId: String): String
    private external fun nativeDrainEvents(maxEvents: Int): String
    private external fun nativeTranscript(logPath: String): String
}

class AgentRuntimeException(message: String) : Exception(message)
