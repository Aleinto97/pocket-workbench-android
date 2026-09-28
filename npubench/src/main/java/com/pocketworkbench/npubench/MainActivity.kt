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
 * Measures whether the Hexagon NPU is worth using for Pocket Workbench, by
 * comparing it against the CPU on the same device with the same model.
 *
 * This must be an app, not a terminal command: the ggml-hexagon backend asks
 * RPCCode for an absolute skel URI, and RPCCode resolves it against the
 * calling process's native library directory. From adb shell there is none,
 * and the device has no root, so the session fails with error 0x80000406.
 * The native libraries are installed by the package manager, which only
 * happens for an app.
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
        setContentView(
            LinearLayout(this).apply {
                orientation = LinearLayout.VERTICAL
                setPadding(pad, pad, pad, pad)
                addView(scroll, LinearLayout.LayoutParams(-1, -1))
            },
        )
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
            say("htp skel:  " + File(nativeDir, SKEL).exists())
            say("htp host:  " + File(nativeDir, HOST).exists())
            say("llama:     " + File(nativeDir, LLAMA).exists())

            val bench = extractBench()
            if (bench == null) {
                say("ABORT: benchmark executable missing from assets")
                return
            }

            val model = findModel() ?: return

            for (device in DEVICES) {
                say("")
                say("=== device=$device ===")
                val out = runBench(nativeDir, bench, model, device) ?: continue
                for (raw in out.lines()) {
                    val line = raw.trim()
                    if (line.isEmpty()) continue
                    if (KEEP.any { line.contains(it, ignoreCase = true) }) say(line)
                }
            }
        } catch (t: Throwable) {
            say("FAILED: ${t.message}")
            t.printStackTrace()
        }
    }

    /**
     * The benchmark is an ELF program, not a shared library, so Android will
     * not extract it from jniLibs; copy it out of the assets instead.
     *
     * It must land in the internal cache: externalCacheDir lives on the
     * /storage/emulated FUSE mount, which is noexec, so exec fails with
     * EACCES whatever mode the file carries.
     */
    private fun extractBench(): File? {
        val out = File(cacheDir, BENCH)
        if (!out.exists() || out.length() == 0L) {
            assets.open(BENCH).use { input -> out.outputStream().use { input.copyTo(it) } }
        }
        out.setExecutable(true, true)
        say("bench: ${out.absolutePath} executable=${out.canExecute()}")
        return out
    }

    /**
     * The model is staged by the harness in a shell-readable directory, so
     * this works on a release build and needs no run-as.
     */
    private fun findModel(): File? {
        val candidates = listOf(
            File(STAGED_DIR, MODEL),
            File(externalCacheDir ?: filesDir, MODEL),
            File(filesDir, MODEL),
        )
        val hit = candidates.firstOrNull { it.isFile && it.length() > 1024L * 1024L }
        if (hit == null) {
            say("ABORT: no model found. Looked in:")
            candidates.forEach {
                say("   ${it.absolutePath} " +
                    (if (it.exists()) "${it.length()} bytes" else "missing"))
            }
        } else {
            say("model: ${hit.absolutePath} (${hit.length() / 1048576} MB)")
        }
        return hit
    }

    /** Run the benchmark for one compute unit, with the app's libs on the path. */
    private fun runBench(
        nativeDir: File,
        bench: File,
        model: File,
        device: String,
    ): String? {
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
        // RPCCode resolves the HTP skel beside the app's native libraries
        val env: MutableMap<String, String> = pb.environment()
        env["LD_LIBRARY_PATH"] = nativeDir.absolutePath
        env["ADSP_LIBRARY_PATH"] =
            nativeDir.absolutePath + ";/vendor/lib/rfsa/adsp;/dsp"
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
        private const val LLAMA = "libllama.so"
        private const val BENCH = "geniex-bench"
        private const val STAGED_DIR = "/data/local/tmp/npb"
        private const val MODEL = "bench.gguf"
        private val DEVICES = arrayOf("npu", "cpu")
        private val KEEP = arrayOf("pp", "tg", "model size", "error", "failed", "htp", "hexagon")
    }
}
