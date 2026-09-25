package com.pocketworkbench.app

class NativeEngine {
    init { System.loadLibrary("pocketnative") }
    interface TokenCallback {
        fun onToken(piece: String)
        fun onStats(json: String)
    }
    // logDir: app files/logs directory — native code writes native_state.txt
    // (last generation phase) and, on a signal fault, native_crash.txt there so
    // process deaths during inference can be diagnosed after the fact.
    external fun generate(path: String, roles: Array<String>, contents: Array<String>, callback: TokenCallback, logDir: String)
    external fun stop()
    external fun transcribe(path: String, samples: FloatArray, logDir: String): String
}
