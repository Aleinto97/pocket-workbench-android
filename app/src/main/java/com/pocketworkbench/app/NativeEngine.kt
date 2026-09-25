package com.pocketworkbench.app

class NativeEngine {
    init { System.loadLibrary("pocketnative") }
    interface TokenCallback { fun onToken(piece: String) }
    external fun generate(path: String, roles: Array<String>, contents: Array<String>, callback: TokenCallback)
    external fun stop()
    external fun transcribe(path: String, samples: FloatArray): String
}
