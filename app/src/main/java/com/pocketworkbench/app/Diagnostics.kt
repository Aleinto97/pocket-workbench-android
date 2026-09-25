package com.pocketworkbench.app

import android.app.Activity
import android.app.Application
import android.content.Context
import android.os.Build
import android.os.Bundle
import android.os.Debug
import android.os.PowerManager
import android.os.Process
import org.json.JSONObject
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Diag — technical diagnostic log that survives process death.
 *
 * Why: users reported the app "jumps to another app" at the end of a generation,
 * and afterwards stats/history are gone. That behaviour is the signature of a
 * process death (crash or kill): Android returns to the previous task. To catch
 * the culprit we keep, on disk in filesDir/logs:
 *   - diag.log            : rolling event log (lifecycle, generation milestones, memory)
 *   - session_state.txt   : last known phase of the previous process (pid, phase, tokens)
 *   - crash_latest.txt    : Java crash report written by the uncaught-exception handler
 *   - native_crash.txt    : signal report written by the native handler in bridge.cpp
 * On the next launch the previous state is consumed and surfaced in Stats > Diagnostics.
 */
object Diag {
    private const val MAX_RING = 800
    private const val MAX_LOG_BYTES = 400_000L

    private var appContext: Context? = null
    private lateinit var logsDir: File
    private lateinit var logFile: File
    private val ring = ArrayDeque<String>()
    private val ts = SimpleDateFormat("MM-dd HH:mm:ss.SSS", Locale.US)

    var foreground = false; private set
    var sessionStart = 0L; private set
    var restarts = 0; private set
    var previousEndSummary: String? = null; private set
    var lastCrashReport: String? = null; private set
    private var previousHandler: Thread.UncaughtExceptionHandler? = null

    fun install(context: Context) {
        appContext = context.applicationContext
        logsDir = File(appContext!!.filesDir, "logs").apply { mkdirs() }
        logFile = File(logsDir, "diag.log")
        sessionStart = System.currentTimeMillis()
        val pid = Process.myPid()

        // 1) Consume the previous process state BEFORE writing our own.
        //    The native side (bridge.cpp) writes native_state.txt with the exact
        //    generation phase + token count; the JVM side writes session_state.txt.
        //    The fresher of the two describes how the previous process ended.
        val jvmState = readStateFile(File(logsDir, "session_state.txt"))
        val nativeState = readStateFile(File(logsDir, "native_state.txt"))
        val state = when {
            jvmState == null -> nativeState
            nativeState == null -> jvmState
            nativeState.optLong("ts") >= jvmState.optLong("ts") -> nativeState
            else -> jvmState
        }
        if (state != null && state.optInt("pid", -1) != Process.myPid()) {
            restarts = jvmState?.optInt("restarts", 0)?.plus(1) ?: 1
            val phase = state.optString("phase", "unknown")
            val detail = state.optString("detail", "")
            val whenMs = state.optLong("ts", 0L)
            val whenTxt = SimpleDateFormat("MM-dd HH:mm", Locale.US).format(Date(whenMs))
            previousEndSummary = when (phase) {
                "done", "startup" ->
                    "Previous process (pid ${state.optInt("pid")}) ended after phase '$phase' at $whenTxt " +
                        "(killed in background or swiped away, not a crash)"
                else -> "Previous process (pid ${state.optInt("pid")}) DIED during phase '$phase'" +
                    (if (detail.isNotBlank()) " ($detail)" else "") + " at $whenTxt — crash or system kill"
            }
        } else if (jvmState != null) {
            restarts = jvmState.optInt("restarts", 0)
        }
        // consume the native phase snapshot: it describes only this one death and
        // will be rewritten by the next generation anyway
        try { File(logsDir, "native_state.txt").delete() } catch (_: Exception) {}

        // 2) Consume native crash report (written by the signal handler in bridge.cpp).
        val nativeCrash = File(logsDir, "native_crash.txt")
        if (nativeCrash.exists()) {
            try {
                val body = nativeCrash.readText().trim()
                if (body.isNotBlank()) {
                    lastCrashReport = "NATIVE SIGNAL\n" + body.lines().joinToString("\n")
                }
                nativeCrash.copyTo(File(logsDir, "native_crash_seen.txt"), overwrite = true)
            } catch (_: Exception) {}
            nativeCrash.delete()
        }

        // 3) Consume Java crash report (written by the uncaught-exception handler).
        val crashFile = File(logsDir, "crash_latest.txt")
        if (crashFile.exists()) {
            try {
                val body = crashFile.readText()
                if (body.isNotBlank()) lastCrashReport = body.trim().lines().take(14).joinToString("\n")
                val history = File(logsDir, "crash_history").apply { mkdirs() }
                crashFile.copyTo(File(history, "crash_${System.currentTimeMillis()}.txt"), overwrite = true)
                // keep only the newest 5 archived reports
                history.listFiles()?.sortedByDescending { it.name }?.drop(5)?.forEach { it.delete() }
            } catch (_: Exception) {}
            crashFile.delete()
        }

        // 4) Rotate the log if it grew too large.
        try {
            if (logFile.exists() && logFile.length() > MAX_LOG_BYTES) {
                val old = File(logsDir, "diag.old.log")
                old.delete()
                logFile.renameTo(old)
            }
        } catch (_: Exception) {}

        // 5) Install the Java crash handler (covers uncaught exceptions, incl. coroutines).
        previousHandler = Thread.getDefaultUncaughtExceptionHandler()
        Thread.setDefaultUncaughtExceptionHandler { thread, throwable ->
            try { writeCrashReport(thread, throwable) } catch (_: Throwable) {}
            previousHandler?.uncaughtException(thread, throwable)
        }

        updateState("startup", "app ${appVersion()}")

        val device = DeviceInfo.capture(context)
        log("app", "=== session start === pid=$pid restarts=$restarts app=${appVersion()}")
        log("app", "device: ${device.manufacturer} ${device.model} · ${device.soc} · ${device.cores} cores · Android ${device.android} · RAM ${device.totalRamMb / 1024} GB")
        previousEndSummary?.let { log("app", "NOTE: $it") }
        lastCrashReport?.let { log("app", "NOTE: last crash report present:\n$it") }
    }

    val sessionInfo: String
        get() {
            val start = SimpleDateFormat("MM-dd HH:mm:ss", Locale.US).format(Date(sessionStart))
            return "PID ${Process.myPid()} · session from $start · process restarts: $restarts"
        }

    fun appVersion(): String {
        val ctx = appContext ?: return "?"
        return try {
            val pm = ctx.packageManager
            val p = pm.getPackageInfo(ctx.packageName, 0)
            val code = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) p.longVersionCode else p.versionCode.toLong()
            "${p.versionName} ($code)"
        } catch (_: Exception) { "?" }
    }

    @Synchronized fun log(tag: String, message: String) {
        val line = "${ts.format(Date())} [${Thread.currentThread().name.take(14)}] $tag: $message"
        synchronized(ring) {
            ring.addLast(line)
            while (ring.size > MAX_RING) ring.removeFirst()
        }
        try {
            if (this::logFile.isInitialized) logFile.appendText(line + "\n")
        } catch (_: Throwable) {}
    }

    fun recent(n: Int): List<String> = synchronized(ring) { ring.toList().takeLast(n) }

    /** Foreground bookkeeping — set from WorkbenchApp lifecycle callbacks. */
    fun setForeground(v: Boolean) { foreground = v }

    /** Last known phase bookkeeping, consumed on the next launch to diagnose process death. */
    fun updateState(phase: String, detail: String = "") {
        if (!this::logsDir.isInitialized) return
        try {
            File(logsDir, "session_state.txt").writeText(
                "pid=${Process.myPid()}\nphase=$phase\ndetail=$detail\nrestarts=$restarts\nts=${System.currentTimeMillis()}\n")
        } catch (_: Exception) {}
    }

    private fun readStateFile(f: File): JSONObject? = try {
        if (!f.exists()) null
        else {
            val map = mutableMapOf<String, String>()
            f.readText().lines().forEach { l ->
                val i = l.indexOf('=')
                if (i > 0) map[l.substring(0, i)] = l.substring(i + 1)
            }
            JSONObject(map)
        }
    } catch (_: Exception) { null }

    private fun writeCrashReport(thread: Thread, throwable: Throwable) {
        if (!this::logsDir.isInitialized) return
        val sb = StringBuilder()
        sb.appendLine("UNCAUGHT EXCEPTION on thread '${thread.name}' at ${ts.format(Date())}")
        sb.appendLine("app ${appVersion()} · phase context below")
        sb.appendLine(throwable.javaClass.name + ": " + (throwable.message ?: ""))
        throwable.stackTrace.take(40).forEach { sb.appendLine("  at $it") }
        throwable.cause?.let { cause ->
            sb.appendLine("Caused by ${cause.javaClass.name}: ${cause.message}")
            cause.stackTrace.take(20).forEach { sb.appendLine("  at $it") }
        }
        sb.appendLine()
        sb.appendLine("Memory: pss=${pssMb()} MB · freeRam=${freeRamMb()} MB · thermal=${thermalName()}")
        sb.appendLine("Foreground: $foreground")
        sb.appendLine("Last 30 events:")
        synchronized(ring) { ring.toList().takeLast(30) }.forEach { sb.appendLine("  $it") }
        File(logsDir, "crash_latest.txt").writeText(sb.toString())
        log("crash", "UNCAUGHT ${throwable.javaClass.simpleName}: ${throwable.message}")
        updateState("crashed", "${throwable.javaClass.simpleName}: ${throwable.message?.take(80)}")
    }

    fun freeRamMb(): Long = try {
        DeviceInfo.capture(appContext!!).freeRamMb
    } catch (_: Exception) { 0L }

    // Debug.getPss() returns KiB — convert to MB so exports read sensibly
    // (before this fix the diagnostics export showed "pss=176583 MB").
    fun pssMb(): Long = try { Debug.getPss() / 1024L } catch (_: Exception) { 0L }

    fun thermalName(): String = try {
        val pm = appContext!!.getSystemService(Context.POWER_SERVICE) as PowerManager
        when (pm.currentThermalStatus) {
            PowerManager.THERMAL_STATUS_NONE -> "none"
            PowerManager.THERMAL_STATUS_LIGHT -> "light"
            PowerManager.THERMAL_STATUS_MODERATE -> "moderate"
            PowerManager.THERMAL_STATUS_SEVERE -> "severe"
            PowerManager.THERMAL_STATUS_CRITICAL -> "critical"
            PowerManager.THERMAL_STATUS_EMERGENCY -> "emergency"
            PowerManager.THERMAL_STATUS_SHUTDOWN -> "shutdown"
            else -> "unknown"
        }
    } catch (_: Exception) { "unknown" }

    fun snapshot(): String {
        val sb = StringBuilder()
        sb.appendLine("Pocket Workbench — diagnostics export")
        sb.appendLine("App: ${appVersion()} · ${sessionInfo}")
        sb.appendLine("Device: ${appContext?.let { DeviceInfo.capture(it) }?.summary() ?: "?"}")
        sb.appendLine("Memory now: pss=${pssMb()} MB · freeRam=${freeRamMb()} MB · thermal=${thermalName()}")
        sb.appendLine("Exported: ${SimpleDateFormat("yyyy-MM-dd HH:mm", Locale.US).format(Date())}")
        previousEndSummary?.let { sb.appendLine(); sb.appendLine("Previous process: $it") } ?: sb.appendLine("Previous process: no anomaly recorded")
        sb.appendLine()
        sb.appendLine(lastCrashReport ?: "No crash report present.")
        sb.appendLine()
        sb.appendLine("--- recent events (${recent(MAX_RING).size} lines kept) ---")
        recent(MAX_RING).forEach { sb.appendLine(it) }
        sb.appendLine()
        sb.appendLine("--- on-disk log tail (covers previous sessions, incl. the dead process) ---")
        try {
            val old = File(logsDir, "diag.old.log")
            if (old.exists()) old.readLines().takeLast(50).forEach { sb.appendLine(it) }
            if (this::logFile.isInitialized) logFile.readLines().takeLast(120).forEach { sb.appendLine(it) }
        } catch (_: Exception) {}
        sb.appendLine()
        sb.appendLine("--- old rotation (if present) ---")
        try {
            val old = File(logsDir, "diag.old.log")
            if (old.exists()) old.readLines().takeLast(120).forEach { sb.appendLine(it) }
        } catch (_: Exception) {}
        return sb.toString()
    }

    fun clearCrashMarkers() {
        try {
            File(logsDir, "native_crash_seen.txt").delete()
            File(logsDir, "crash_history").listFiles()?.forEach { it.delete() }
            File(logsDir, "session_state.txt").delete()
        } catch (_: Exception) {}
        lastCrashReport = null
        previousEndSummary = null
        restarts = 0
        log("app", "diagnostics crash markers cleared by user")
        updateState("startup", "markers cleared")
    }
}

/**
 * Application class: installs Diag as early as possible and records every
 * activity lifecycle transition so we can tell whether the app left the
 * foreground on its own (intent) or died (crash/kill).
 */
class WorkbenchApp : Application() {
    override fun onCreate() {
        super.onCreate()
        Diag.install(this)
        registerActivityLifecycleCallbacks(object : Application.ActivityLifecycleCallbacks {
            private fun name(a: Activity) = a.javaClass.simpleName
            override fun onActivityCreated(a: Activity, b: Bundle?) { Diag.log("lifecycle", "onCreate ${name(a)}") }
            override fun onActivityStarted(a: Activity) { Diag.log("lifecycle", "onStart ${name(a)}") }
            override fun onActivityResumed(a: Activity) {
                Diag.setForeground(true)
                Diag.log("lifecycle", "onResume ${name(a)} (app comes to foreground)")
            }
            override fun onActivityPaused(a: Activity) {
                Diag.setForeground(false)
                Diag.log("lifecycle", "onPause ${name(a)} (app leaves foreground)")
            }
            override fun onActivityStopped(a: Activity) { Diag.log("lifecycle", "onStop ${name(a)}") }
            override fun onActivitySaveInstanceState(a: Activity, b: Bundle) {}
            override fun onActivityDestroyed(a: Activity) { Diag.log("lifecycle", "onDestroy ${name(a)}") }
        })
    }

    override fun onTrimMemory(level: Int) {
        super.onTrimMemory(level)
        Diag.log("memory", "onTrimMemory level=$level pss=${Diag.pssMb()} MB freeRam=${Diag.freeRamMb()} MB")
    }
}
