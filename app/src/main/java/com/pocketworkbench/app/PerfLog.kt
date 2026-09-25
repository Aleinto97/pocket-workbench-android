package com.pocketworkbench.app

import android.app.ActivityManager
import android.content.Context
import android.os.Build
import androidx.compose.runtime.mutableStateListOf
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

data class PerfEntry(
    val timestamp: Long,
    val model: String,
    val backend: String,
    val threads: Int,
    val loadMs: Double,
    val cached: Boolean,
    val gpuFallback: Boolean,
    val prefillTokens: Int,
    val prefillMs: Double,
    val genTokens: Int,
    val genMs: Double,
    val stop: String,
    val deviceModel: String,
    val soc: String,
    val android: String,
    val totalRamMb: Long,
    val freeRamMb: Long,
    val problem: String,
    val ttftMs: Double = 0.0,
    val pssMb: Int = 0,
    val thermal: String = "",
    val foreground: Boolean = true
) {
    val genTps: Double get() = if (genMs > 0) genTokens / (genMs / 1000.0) else 0.0
    val prefillTps: Double get() = if (prefillMs > 0) prefillTokens / (prefillMs / 1000.0) else 0.0
    fun chip(): String {
        val parts = mutableListOf<String>()
        if (genTps > 0) parts.add(String.format(Locale.US, "%.1f tok/s", genTps))
        if (genTokens > 0) parts.add("$genTokens tok")
        if (ttftMs > 0) parts.add(String.format(Locale.US, "%.1fs to 1st", ttftMs / 1000.0))
        if (!cached && loadMs > 0) parts.add(String.format(Locale.US, "load %.1fs", loadMs / 1000.0))
        if (parts.isEmpty()) return backend
        return parts.joinToString(" · ") + " · $backend"
    }
    fun row(): String {
        val time = SimpleDateFormat("MM-dd HH:mm", Locale.US).format(Date(timestamp))
        val load = if (cached) "cached" else String.format(Locale.US, "%.1fs", loadMs / 1000.0)
        val ttft = if (ttftMs > 0) String.format(Locale.US, "%.1fs", ttftMs / 1000.0) else "-"
        val flags = buildList {
            if (gpuFallback) add("GPU-fallback")
            if (cached) add("model-cached")
            if (!foreground) add("backgrounded")
            if (thermal.isNotBlank() && thermal != "none") add("thermal=$thermal")
        }.joinToString(",")
        return String.format(Locale.US, "| %s | %s | %s | %s | %.0f | %.1f | %d | %s | %s | %s | %s |",
            time, model.take(28), backend, load, prefillTps, genTps, genTokens, ttft, threads, stop, flags.ifEmpty { "-" })
    }
}

data class DeviceSnapshot(
    val model: String, val manufacturer: String, val soc: String, val cores: Int,
    val android: String, val totalRamMb: Long, val freeRamMb: Long
) {
    fun summary(): String = "$manufacturer $model · SoC: $soc · $cores cores · Android $android · RAM ${totalRamMb / 1024} GB (${freeRamMb} MB free)"
}

object DeviceInfo {
    fun capture(context: Context): DeviceSnapshot {
        val am = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val mem = ActivityManager.MemoryInfo()
        am.getMemoryInfo(mem)
        val soc = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) Build.SOC_MODEL else Build.HARDWARE
        return DeviceSnapshot(
            model = Build.MODEL, manufacturer = Build.MANUFACTURER, soc = soc,
            cores = Runtime.getRuntime().availableProcessors(),
            android = Build.VERSION.RELEASE, totalRamMb = mem.totalMem / (1024 * 1024),
            freeRamMb = mem.availMem / (1024 * 1024)
        )
    }
}

class PerfLog(private val context: Context) {
    private val file = File(context.filesDir, "perf_log.json")
    val entries = mutableStateListOf<PerfEntry>()
    val device: DeviceSnapshot = DeviceInfo.capture(context)

    init { load() }

    fun add(entry: PerfEntry) {
        entries.add(0, entry)
        while (entries.size > 200) entries.removeAt(entries.lastIndex)
        persist()
    }

    fun clear() { entries.clear(); file.delete() }

    private fun detectProblem(e: PerfEntry): String {
        val issues = mutableListOf<String>()
        if (e.gpuFallback) issues.add("GPU load failed: fell back to CPU")
        when (e.stop) {
            "context_full" -> issues.add("Context overflow (4096-token window)")
            "error" -> issues.add("Inference error during generation")
            "max_tokens" -> issues.add("Reply hit the 512-token cap")
            else -> {}
        }
        if (e.genTps in 0.01..1.0) issues.add(String.format(Locale.US, "Very low generation speed (%.2f tok/s)", e.genTps))
        if (e.ttftMs > 8000) issues.add(String.format(Locale.US, "Slow first token (TTFT %.1fs incl. model load)", e.ttftMs / 1000.0))
        if (!e.foreground) issues.add("Run ended while the app was in the background")
        if (e.thermal in listOf("severe", "critical", "emergency", "shutdown")) issues.add("Thermal throttling (${e.thermal})")
        if (e.pssMb > 3500) issues.add("High app memory footprint (${e.pssMb} MB PSS)")
        if (e.freeRamMb < 800) issues.add("Low free memory at run time (${e.freeRamMb} MB)")
        return issues.joinToString("; ")
    }

    fun record(model: String, stats: JSONObject): PerfEntry {
        val snap = DeviceInfo.capture(context)
        val entry = PerfEntry(
            timestamp = System.currentTimeMillis(), model = model,
            backend = stats.optString("backend", "CPU"), threads = stats.optInt("threads", 0),
            loadMs = stats.optDouble("load_ms", 0.0), cached = stats.optInt("model_cached", 0) == 1,
            gpuFallback = stats.optInt("gpu_fallback", 0) == 1,
            prefillTokens = stats.optInt("prefill_tokens", 0), prefillMs = stats.optDouble("prefill_ms", 0.0),
            genTokens = stats.optInt("gen_tokens", 0), genMs = stats.optDouble("gen_ms", 0.0),
            stop = stats.optString("stop", "unknown"),
            deviceModel = snap.model, soc = snap.soc, android = snap.android,
            totalRamMb = snap.totalRamMb, freeRamMb = snap.freeRamMb,
            problem = "",
            ttftMs = if (stats.has("ttft_ms")) stats.optDouble("ttft_ms", 0.0) else 0.0,
            pssMb = stats.optInt("pss_mb", 0),
            thermal = stats.optString("thermal", ""),
            foreground = stats.optBoolean("fg", true)
        )
        val withProblem = entry.copy(problem = detectProblem(entry))
        add(withProblem)
        return withProblem
    }

    private fun persist() {
        try {
            val array = JSONArray()
            entries.forEach { e ->
                array.put(JSONObject()
                    .put("ts", e.timestamp).put("model", e.model).put("backend", e.backend)
                    .put("threads", e.threads).put("load_ms", e.loadMs).put("cached", e.cached)
                    .put("gpu_fallback", e.gpuFallback).put("p_tokens", e.prefillTokens).put("p_ms", e.prefillMs)
                    .put("g_tokens", e.genTokens).put("g_ms", e.genMs).put("stop", e.stop)
                    .put("device", e.deviceModel).put("soc", e.soc).put("android", e.android)
                    .put("ram_total", e.totalRamMb).put("ram_free", e.freeRamMb).put("problem", e.problem)
                    .put("ttft", e.ttftMs).put("pss", e.pssMb).put("thermal", e.thermal).put("fg", e.foreground))
            }
            val temp = File(file.parentFile, "perf_log.tmp")
            temp.writeText(array.toString())
            if (!temp.renameTo(file)) temp.delete()
        } catch (_: Exception) {}
    }

    private fun load() {
        try {
            if (!file.exists()) return
            val array = JSONArray(file.readText())
            (0 until array.length()).map { i ->
                val o = array.getJSONObject(i)
                PerfEntry(
                    timestamp = o.getLong("ts"), model = o.getString("model"), backend = o.getString("backend"),
                    threads = o.optInt("threads"), loadMs = o.optDouble("load_ms", 0.0), cached = o.optBoolean("cached"),
                    gpuFallback = o.optBoolean("gpu_fallback"),
                    prefillTokens = o.optInt("p_tokens"), prefillMs = o.optDouble("p_ms", 0.0),
                    genTokens = o.optInt("g_tokens"), genMs = o.optDouble("g_ms", 0.0),
                    stop = o.optString("stop", "unknown"), deviceModel = o.optString("device"),
                    soc = o.optString("soc"), android = o.optString("android"),
                    totalRamMb = o.optLong("ram_total"), freeRamMb = o.optLong("ram_free"),
                    problem = o.optString("problem"),
                    ttftMs = o.optDouble("ttft", 0.0), pssMb = o.optInt("pss", 0),
                    thermal = o.optString("thermal"), foreground = o.optBoolean("fg", true)
                )
            }.forEach { entries.add(it) }
        } catch (_: Exception) {}
    }

    fun exportText(): String {
        val sb = StringBuilder()
        sb.appendLine("Pocket Workbench — technical performance log")
        sb.appendLine("Device: ${device.summary()}")
        sb.appendLine("Inference: llama.cpp (CPU + optional Vulkan GPU; NPU not used by this build)")
        sb.appendLine("Exported: ${SimpleDateFormat("yyyy-MM-dd HH:mm", Locale.US).format(Date())}")
        sb.appendLine("Runs recorded: ${entries.size}")
        sb.appendLine()
        sb.appendLine("| time | model | backend | load | prefill tok/s | gen tok/s | gen tok | ttft | threads | stop | flags |")
        sb.appendLine("|---|---|---|---|---|---|---|---|---|---|---|")
        entries.forEach { sb.appendLine(it.row()) }
        val ttfts = entries.map { it.ttftMs }.filter { it > 0 }
        if (ttfts.isNotEmpty()) sb.appendLine(String.format(Locale.US, "Average time to first token: %.1fs", ttfts.average() / 1000.0))
        val problems = entries.filter { it.problem.isNotBlank() }
        if (problems.isNotEmpty()) {
            sb.appendLine()
            sb.appendLine("Detected problems:")
            problems.forEach { sb.appendLine("- ${SimpleDateFormat("MM-dd HH:mm", Locale.US).format(Date(it.timestamp))} [${it.model}] ${it.problem}") }
        }
        val cpu = entries.filter { it.backend.startsWith("CPU") }
        val gpu = entries.filter { it.backend.startsWith("Vulkan") }
        sb.appendLine()
        fun avg(list: List<PerfEntry>) = if (list.isEmpty()) 0.0 else list.map { it.genTps }.average()
        sb.appendLine(String.format(Locale.US, "Summary: %d CPU runs (avg %.1f tok/s), %d Vulkan GPU runs (avg %.1f tok/s)", cpu.size, avg(cpu), gpu.size, avg(gpu)))
        return sb.toString()
    }
}
