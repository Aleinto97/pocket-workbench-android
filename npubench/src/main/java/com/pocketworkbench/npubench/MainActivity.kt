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
import java.nio.file.Files
import java.util.Locale
import java.util.concurrent.TimeUnit
import org.json.JSONObject

/**
 * Measures whether the Hexagon NPU is worth using for Pocket Workbench, by
 * comparing it against the CPU on the same device with the same model.
 *
 * This must be an app, not a terminal command: the ggml-hexagon backend asks
 * RPCCode for an absolute skel URI. The plugin sets ADSP_LIBRARY_PATH to its
 * plugin root; the skel must be linked there from an executable, installed
 * native library directory. From adb shell there is no such app directory,
 * and the device has no root, so the session fails with error 0x80000406.
 */
class MainActivity : Activity() {

    private val ui = Handler(Looper.getMainLooper())
    private lateinit var log: TextView
    private val logFile by lazy { File(filesDir, "npubench.log") }

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
        runCatching { logFile.appendText(line + "\n") }
        ui.post { log.append(line + "\n") }
    }

    private fun run() {
        try {
            logFile.writeText("")
            val nativeDir = File(applicationInfo.nativeLibraryDir)
            say("nativeLibraryDir = ${nativeDir.absolutePath}")
            say("htp skel:  " + File(nativeDir, SKEL).exists())
            say("htp host:  " + File(nativeDir, HOST).exists())
            say("llama:     " + File(nativeDir, LLAMA).exists())

            val bench = File(nativeDir, BENCH)
            say("bench: ${bench.absolutePath} executable=${bench.canExecute()}")
            if (!bench.isFile || !bench.canExecute()) {
                say("ABORT: benchmark executable missing or not executable in nativeLibraryDir")
                return
            }
            val exitShim = File(nativeDir, EXIT_SHIM)
            if (!exitShim.isFile) {
                say("ABORT: benchmark exit shim missing from nativeLibraryDir")
                return
            }

            // GenieX only looks for libgeniex_plugin.so inside child directories.
            // jniLibs is installed flat and the app data directory is noexec, so
            // point a child directory at the executable native library directory.
            val pluginRoot = File(cacheDir, "geniex-plugins")
            check(pluginRoot.isDirectory || pluginRoot.mkdirs()) { "Cannot create $pluginRoot" }
            val pluginLink = File(pluginRoot, "llama_cpp").toPath()
            Files.deleteIfExists(pluginLink)
            Files.createSymbolicLink(pluginLink, nativeDir.toPath())
            // LlamaPlugin replaces ADSP_LIBRARY_PATH with GENIEX_PLUGIN_PATH.
            // Keep the skel at that root as well for FastRPC's absolute URI.
            val skelLink = File(pluginRoot, SKEL).toPath()
            Files.deleteIfExists(skelLink)
            Files.createSymbolicLink(skelLink, File(nativeDir, SKEL).toPath())
            say("plugin: $pluginLink -> $nativeDir")

            val modelName = intent.getStringExtra("model") ?: MODEL
            if (!modelName.matches(Regex("[A-Za-z0-9._-]+\\.gguf"))) {
                say("ABORT: invalid model filename: $modelName")
                return
            }
            val model = findModel(modelName) ?: return
            val requestedDevice = intent.getStringExtra("device")
            if (requestedDevice != null && requestedDevice !in DEVICES) {
                say("ABORT: invalid device: $requestedDevice")
                return
            }
            val devices = if (requestedDevice == null) DEVICES else arrayOf(requestedDevice)
            val accuracy = intent.getBooleanExtra("accuracy", false)
            val maxTokens = intent.getIntExtra("tokens", 96).coerceIn(1, 512)
            val prompt = if (accuracy) File(filesDir, "bench-prompt.txt").apply {
                writeText((intent.getStringExtra("prompt") ?: DEFAULT_PROMPT).trim() + "\n")
                say("accuracy prompt: ${readText().trim()}")
            } else null
            DEVICES.forEach { File(filesDir, "bench-$it.json").delete() }

            for (device in devices) {
                say("")
                say("=== device=$device ===")
                val out = runBench(nativeDir, pluginRoot, bench, exitShim, model, device, prompt, maxTokens) ?: continue
                for (raw in out.lines()) {
                    val line = raw.replace(ANSI, "").trim()
                    if (line.isEmpty()) continue
                    if (line.contains("ErrorCode[0](Success)") ||
                        line.contains("failed to get plugin version for qairt")) continue
                    if ((accuracy && line.startsWith("[gen ]")) ||
                        KEEP.any { line.contains(it, ignoreCase = true) }) say(line)
                }
            }
            val npu = File(filesDir, "bench-npu.json")
            val cpu = File(filesDir, "bench-cpu.json")
            if (npu.isFile && cpu.isFile) {
                val npuAgg = JSONObject(npu.readText()).getJSONObject("agg")
                val cpuAgg = JSONObject(cpu.readText()).getJSONObject("agg")
                val cpuPrefill = cpuAgg.getJSONObject("prefill_tps").getDouble("median")
                val cpuDecode = cpuAgg.getJSONObject("decode_tps").getDouble("median")
                if (cpuPrefill > 0 && cpuDecode > 0) {
                    say(String.format(Locale.US, "NPU/CPU: prefill %.1fx, decode %.2fx",
                        npuAgg.getJSONObject("prefill_tps").getDouble("median") / cpuPrefill,
                        npuAgg.getJSONObject("decode_tps").getDouble("median") / cpuDecode))
                }
            }
        } catch (t: Throwable) {
            say("FAILED: ${t.message}")
            t.printStackTrace()
        }
    }

    /**
     * The model is staged by the harness in a shell-readable directory, so
     * this works on a release build and needs no run-as.
     */
    private fun findModel(filename: String): File? {
        val candidates = listOf(
            File(STAGED_DIR, filename),
            File(externalCacheDir ?: filesDir, filename),
            File(filesDir, filename),
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
        pluginRoot: File,
        bench: File,
        exitShim: File,
        model: File,
        device: String,
        prompt: File?,
        maxTokens: Int,
    ): String? {
        val report = File(filesDir, "bench-$device.json")
        report.delete()
        val outputFile = File(filesDir, "bench-$device.log")
        val cmd = mutableListOf(
            bench.absolutePath,
            "--plugin", "llama_cpp",
            "--device", device,
            "-m", model.absolutePath,
        )
        if (prompt != null) {
            cmd += listOf("--accuracy", "--no-think", "--prompt-file", prompt.absolutePath,
                "-n", maxTokens.toString())
        } else {
            cmd += listOf("-p", "512", "-n", "32", "-r", "1")
        }
        cmd += listOf("--output-json", report.absolutePath)
        val pb = ProcessBuilder(cmd)
        pb.redirectErrorStream(true)
        // Save output as it arrives: a memory-bound NPU run can be killed before
        // readText() reaches EOF, and the partial backend diagnostics matter.
        pb.redirectOutput(outputFile)
        // The plugin sets ADSP_LIBRARY_PATH to GENIEX_PLUGIN_PATH; both
        // the plugin directory and the DSP skel are linked from that root.
        val env: MutableMap<String, String> = pb.environment()
        env["LD_LIBRARY_PATH"] = nativeDir.absolutePath + ":/vendor/lib64"
        env["LD_PRELOAD"] = exitShim.absolutePath
        env["GENIEX_PLUGIN_PATH"] = pluginRoot.absolutePath
        say("running --device $device")
        return try {
            val p = pb.start()
            if (!p.waitFor(120, TimeUnit.SECONDS)) {
                say("timeout after 120s on $device; stopping benchmark")
                p.destroyForcibly()
                p.waitFor(10, TimeUnit.SECONDS)
            }
            say("exit=${if (p.isAlive) "still running" else p.exitValue()}")
            val text = outputFile.readText()
            if (report.isFile) {
                val agg = JSONObject(report.readText()).getJSONObject("agg")
                say("$device: prefill=${agg.getJSONObject("prefill_tps").getDouble("median")} tok/s " +
                    "decode=${agg.getJSONObject("decode_tps").getDouble("median")} tok/s " +
                    "ttft=${agg.getJSONObject("ttft_ms").getDouble("median")} ms")
                if (prompt != null) {
                    say("$device: generated=${agg.getJSONObject("gen_tokens").getDouble("median")} tokens")
                }
            }
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
        private const val BENCH = "libgeniexbench.so"
        private const val EXIT_SHIM = "libgeniexbench_exit.so"
        private const val STAGED_DIR = "/data/local/tmp/npb"
        private const val MODEL = "bench.gguf"
        private const val DEFAULT_PROMPT = "Rispondi in italiano, senza spiegazioni: quanto fa 17 moltiplicato per 23?"
        private val ANSI = Regex("\u001B\\[[0-9;]*[A-Za-z]")
        private val DEVICES = arrayOf("npu", "cpu")
        private val KEEP = arrayOf(
            "[ok  ]", "prefill=", "decode=", "model size", "error", "failed",
            "ggml-hex:", "Registered plugin", "Setting ADSP_LIBRARY_PATH",
        )
    }
}
