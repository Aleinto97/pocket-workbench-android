package com.pocketworkbench.app

class NativeEngine {
    companion object {
        @Volatile private var state = 0
        @Volatile var loadError: String? = null
            private set

        /** Loads libpocketinfer.so once and reports the failure instead of crashing the app. */
        @Synchronized fun ensureLoaded(): String? {
            if (state == 1) return null
            if (state == 2) return loadError
            return try {
                System.loadLibrary("pocketinfer")
                loadError = null
                state = 1
                null
            } catch (t: Throwable) {
                loadError = t.toString()
                state = 2
                loadError
            }
        }
    }

    interface TokenCallback {
        fun onToken(piece: String)
        fun onStats(json: String)
    }

    external fun generate(path: String, roles: Array<String>, contents: Array<String>, callback: TokenCallback, logDir: String, threads: Int, contextTokens: Int, useGpu: Boolean, directAnswer: Boolean)
    external fun stop()
    external fun engineInfo(): String
    external fun diagnose(path: String, threads: Int, contextTokens: Int): String

    fun infoJson(): String {
        val err = ensureLoaded()
        if (err != null) return "{\"loaded\":false,\"error\":${org.json.JSONObject.quote(err)}}"
        return try {
            engineInfo()
        } catch (t: Throwable) {
            "{\"loaded\":false,\"error\":${org.json.JSONObject.quote(t.toString())}}"
        }
    }

    fun diagnoseJson(path: String, threads: Int, contextTokens: Int): String {
        val err = ensureLoaded()
        if (err != null) {
            return "{\"ok\":false,\"steps\":[{\"name\":\"native_load\",\"ok\":false,\"ms\":0,\"error\":${org.json.JSONObject.quote(err)}}]}"
        }
        return try {
            diagnose(path, threads, contextTokens)
        } catch (t: Throwable) {
            "{\"ok\":false,\"steps\":[{\"name\":\"native_call\",\"ok\":false,\"ms\":0,\"error\":${org.json.JSONObject.quote(t.toString())}}]}"
        }
    }
}
