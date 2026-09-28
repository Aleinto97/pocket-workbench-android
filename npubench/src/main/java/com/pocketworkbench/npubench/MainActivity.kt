package com.pocketworkbench.npubench

import android.app.Activity
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import java.io.File

/**
 * Measures whether the Hexagon NPU beats the CPU on this device, using the
 * GenieX benchmark tool.
 *
 * The NPU is only reachable from inside an app: the backend asks RPCCode for
 * /libggml-htp-v81.so, and RPCCode resolves that relative to the app's native
 * library directory. From adb shell there is no such directory, and without
 * root the skel cannot be placed where it would be found, so the session fails
 * with 0x80000406. That is why this needs to be an APK.
 */
class MainActivity : Activity() {

    private val ui = Handler(Looper.getMainLooper())
    private lateinit var log: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val pad = (16 * resources.displayMetrics.density).toInt()
        log = TextView(this).apply {
            textSize = 11f
            setTextIsSelectable(true)
        }
        val scroll = ScrollView(this)
        scroll.addView(log)
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(pad, pad, pad, pad)
            addView(scroll, LinearLayout.LayoutParams(-1, -1))
        }
        setContentView(root)
        Thread { run() }.start()
    }

    private fun say(line: String) {
        Log.i(TAG, line)
        ui.post { log.append(line + "\n") }
    }

    private fun run() {
        try {
            val nativeDir = File(applicationInfo.nativeLibraryDir)
            say("nativeLibraryDir = ${nativeDir.absolutePath}")
            say("skel in jniLibs: " + File(nativeDir, SKEL).exists())
            say("htp host lib:    " + File(nativeDir, HOST).exists())
            val bench = extractBench() ?: return
            say("bench: ${bench.absolutePath} executable=${bench.canExecute()}")
            for (device in arrayOf("npu", "cpu")) {
                say("")
                say("=== device=$device ===")
                val out = runBench(nativeDir, bench, device) ?: continue
                for (line in out.lines()) {
                    if (line.contains("pp") || line.contains("tg") ||
                        line.contains("model size") || line.contains("error")
                    ) say(line.trim())
                }
            }
        } catch (t: Throwable) {
            say("FAILED: ${t.message}")
            t.printStackTrace()
        }
    }

    /**
     * The benchmark is an ELF executable, not a shared library, so Android
     * will not extract it from jniLibs. Assets are copied verbatim instead.
     */
    private fun extractBench(): File? {
        val out = File(externalCacheDir ?: cacheDir, BENCH)
        if (!out.exists() || out.length() == 0L) {
            assets.open(BENCH).use { input ->
                out.outputStream().use { input.copyTo(it) }
            }
        }
        out.setExecutable(true, true)
        return out
    }

    /** Run the benchmark for one compute unit, with the app's libs on the path. */
    private fun runBench(nativeDir: File, bench: File, device: String): String? {
        val model = File(filesDir, MODEL)
        if (!model.exists()) {
            say("ABORT: model missing at ${model.absolutePath}")
            return null
        }
        val cmd = arrayOf(
            bench.absolutePath,
            "--plugin", "llama_cpp",
            "--device", device,
            "-m", model.absolutePath,
            "-p", "512",
            "-n", "32",
            "-r", "1",
        )
        val pb = ProcessBuilder(*cmd)
        // RPCCode finds the HTP skel beside the app's native libraries
        val env: MutableMap<String, String> = pb.environment()
        env["LD_LIBRARY_PATH"] = nativeDir.absolutePath
        env["ADSP_LIBRARY_PATH"] = nativeDir.absolutePath + ";/vendor/lib/rfsa/adsp;/dsp"
        say("running --device $device")
        return try {
            val p = pb.start()
            val text: String = p.inputStream.bufferedReader().use { it.readText() }
            p.waitFor()
            say("exit=${p.exitValue()}")
            text
        } catch (t: Throwable) {
            say("exec failed: ${t.message}")
            null
        }
    }

    companion object {
        private const val TAG = "NpuBench"
        private const val SKEL = "libggml-htp-v81.so"
        private const val HOST = "libggml-hexagon.so"
        private const val BENCH = "geniex-bench"
        private const val MODEL = "bench.gguf"
    }
}
