package com.pocketworkbench.app

class SpeechEngine {
    init { System.loadLibrary("pocketnative") }
    // whisper.cpp transcription stays in the C++ bridge; only the LLM engine
    // was replaced by the Rust implementation.
    external fun transcribe(path: String, samples: FloatArray, logDir: String): String
}
