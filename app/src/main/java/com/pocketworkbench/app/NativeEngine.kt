package com.pocketworkbench.app

class NativeEngine {
    init { System.loadLibrary("pocketinfer") }
    interface TokenCallback {
        fun onToken(piece: String)
        fun onStats(json: String)
    }
    // logDir: app files/logs directory — native code writes native_state.txt
    // (last generation phase) and, on a signal fault, native_crash.txt there so
    // process deaths during inference can be diagnosed after the fact.
    // threads: 1..6; best measured throughput on Snapdragon is 4.
    external fun generate(path: String, roles: Array<String>, contents: Array<String>, callback: TokenCallback, logDir: String, threads: Int, contextTokens: Int, useGpu: Boolean, directAnswer: Boolean)
    external fun stop()
}
