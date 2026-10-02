package com.pocketworkbench.app

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.IBinder
import android.util.Log
import java.io.File
import java.util.concurrent.CopyOnWriteArrayList
import java.util.concurrent.atomic.AtomicBoolean

/**
 * The agent runtime and its resident model, in their own process.
 *
 * Same UID, so `filesDir` is readable from here and from the UI: the runtime owns
 * the session log, the UI derives the transcript from it, and neither keeps a
 * second copy that could disagree. A native crash inside llama.cpp or GenieX takes
 * down this process only, and the UI reports the death instead of pretending the
 * answer is still coming.
 *
 * One worker thread runs turns. `pump` blocks on it, so the Binder call that
 * admits a message returns immediately and Stop can be delivered while a turn is
 * generating.
 */
class AgentService : Service() {

    /**
     * The binder *is* the API object. It extends the generated Stub rather than
     * the Service so there is exactly one implementation of the interface.
     */
    inner class LocalBinder : IPocketAgent.Stub() {
        override fun protocolVersion(): Int = runtime.protocolVersion

        override fun capabilities(): String = respond("capabilities") { runtime.capabilities().toString() }

        override fun openSession(
            sessionId: String,
            logPath: String,
            workspaceRoot: String,
            configJson: String,
        ): String = respond("openSession") {
            val id = sessionId.requireSafeId()
            val log = File(logPath).requireUnder(store.sessions)
            val root = File(workspaceRoot).requireUnder(store.workspaces)
            val config = runCatching { org.json.JSONObject(configJson) }.getOrElse { org.json.JSONObject() }
            runtime.openSession(id, log.absolutePath, root.absolutePath, config).toString().also {
                opened.add(id)
            }
        }

        override fun closeSession(sessionId: String): String = respond("closeSession") {
            opened.remove(sessionId)
            runtime.closeSession(sessionId).toString()
        }

        override fun sessionIds(): String = respond("sessionIds") {
            org.json.JSONObject().put("ok", true).put("ids", runtime.sessionIds()).toString()
        }

        override fun submit(sessionId: String, requestId: String, text: String): String =
            respond("submit") { runtime.submit(sessionId, requestId, text).toString() }

        override fun pump(sessionId: String, maxTurns: Int): String {
            if (!pumping.compareAndSet(false, true)) {
                // A turn is already running on the worker: report that rather
                // than starting a second one for the same session.
                return org.json.JSONObject()
                    .put("ok", true)
                    .put("turns", 0)
                    .put("busy", true)
                    .toString()
            }
            // Drain live while the turn runs on this Binder thread: deltas are
            // pushed every ~150ms so the chat streams instead of appearing only
            // at the end. Ordering still matches the log because both come from
            // the same runtime event queue.
            val drainer = Thread({
                while (pumping.get()) {
                    publishEvents()
                    try { Thread.sleep(150) } catch (_: InterruptedException) { break }
                }
            }, "agent-live-drain").apply { isDaemon = true; start() }
            return try {
                // Blocking on purpose: this *is* the turn.
                val result = runtime.pump(sessionId, maxTurns.coerceIn(1, 8)).toString()
                publishEvents()
                result
            } catch (e: Exception) {
                publishEvents()
                fail(e)
            } finally {
                pumping.set(false)
                drainer.interrupt()
                runCatching { drainer.join(1000) }
                // Final drain after the turn ended, in case the last batch
                // arrived between the last poll and the flag flip.
                publishEvents()
            }
        }

        override fun cancel(sessionId: String): String =
            respond("cancel") { runtime.cancel(sessionId).toString() }

        override fun loadModel(
            sessionId: String,
            modelPath: String,
            backend: String,
            contextTokens: Int,
            threads: Int,
            useGpu: Boolean,
        ): String = respond("loadModel") {
            val file = File(modelPath).requireUnder(store.models)
            require(file.length() > 0) { "Il file del modello è vuoto" }
            notifyState(STATE_MODEL_LOADING, org.json.JSONObject().put("model", file.name).put("backend", backend))
            val result = runtime.loadModel(sessionId, file.absolutePath, backend, contextTokens, threads, useGpu)
            val info = result.optJSONObject("info")
            notifyState(
                STATE_MODEL_READY,
                org.json.JSONObject()
                    .put("model", file.name)
                    .put("backend", info?.optString("backend").orEmpty())
                    .put("fallback", info?.optBoolean("fallback") ?: false),
            )
            result.toString()
        }

        override fun unloadModel(sessionId: String): String = respond("unloadModel") {
            val payload = runtime.unloadModel(sessionId).toString()
            notifyState(STATE_MODEL_UNLOADED, org.json.JSONObject())
            payload
        }

        override fun sessionState(sessionId: String): String =
            respond("sessionState") { runtime.sessionState(sessionId).toString() }

        override fun drainEvents(maxEvents: Int): String =
            respond("drainEvents") { runtime.drainEvents(maxEvents).toString() }

        override fun transcript(logPath: String): String = respond("transcript") {
            runtime.transcript(File(logPath).requireUnder(store.sessions).absolutePath).toString()
        }

        override fun registerCallback(cb: IPocketAgentCallback?) {
            if (cb == null) return
            // Dedupe by underlying binder: re-binding after rotation must not
            // stack duplicate deliveries.
            callbacks.removeIf { existing ->
                runCatching { existing.asBinder() == cb.asBinder() }.getOrDefault(false)
            }
            callbacks.add(cb)
            runCatching {
                cb.asBinder().linkToDeath({
                    callbacks.removeIf { existing ->
                        runCatching { existing.asBinder() == cb.asBinder() }.getOrDefault(false)
                    }
                }, 0)
            }
            runCatching {
                cb.onServiceState(
                    STATE_READY,
                    org.json.JSONObject().put("protocol_version", runtime.protocolVersion).toString(),
                )
            }
        }

        override fun unregisterCallback(cb: IPocketAgentCallback?) {
            if (cb == null) return
            callbacks.removeIf { existing ->
                runCatching { existing.asBinder() == cb.asBinder() }.getOrDefault(false)
            }
            // The linkToDeath recipient from register stays armed; it only
            // removes the same binder again, which is idempotent and harmless.
        }
    }

    private val binder = LocalBinder()
    private val pumping = AtomicBoolean(false)
    private val opened = CopyOnWriteArrayList<String>()
    private val callbacks = CopyOnWriteArrayList<IPocketAgentCallback>()
    private lateinit var runtime: AgentRuntime
    private lateinit var store: WorkbenchStore

    override fun onCreate() {
        super.onCreate()
        runtime = AgentRuntime.obtain(applicationInfo.nativeLibraryDir, cacheDir.absolutePath)
        store = WorkbenchStore(this)
        store.rotateLogs()
        // The curl symlink for Android-shell mode. Independent of the Debian
        // tree: harmless when the native lib is absent (e.g. stripped builds).
        runCatching { LinuxModule(this).ensureTools(applicationInfo.nativeLibraryDir) }
        val failure = AgentRuntime.ensureLoaded()
        if (failure != null) {
            Log.e(TAG, "The native agent runtime did not load: $failure")
            notifyState(STATE_ERROR, org.json.JSONObject().put("stage", "load_library").put("error", failure))
        } else {
            notifyState(
                STATE_READY,
                org.json.JSONObject()
                    .put("protocol_version", runtime.protocolVersion)
                    .put("storage", store.root.absolutePath),
            )
        }
        // Foreground so Android does not kill the process between turns. The
        // notification states plainly what is running and why. The channel must
        // exist before startForeground, otherwise the process dies with
        // CannotPostForegroundServiceNotificationException on first bind.
        runCatching {
            val manager = getSystemService(NotificationManager::class.java)
            manager?.createNotificationChannel(
                NotificationChannel(CHANNEL, getString(R.string.service_agent_title), NotificationManager.IMPORTANCE_LOW)
            )
            startForeground(
                NOTIFICATION_ID,
                notification(getString(R.string.service_agent_ready)),
                ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC,
            )
        }
    }

    override fun onBind(intent: Intent?): IBinder = binder

    override fun onDestroy() {
        // Release the weights explicitly: the teardown of the prebuilt runtime has
        // crashed before, and doing it here keeps that away from the UI process.
        for (id in opened) runCatching { runtime.unloadModel(id) }
        runCatching { opened.forEach { runtime.closeSession(it) } }
        callbacks.clear()
        super.onDestroy()
    }

    private fun notifyState(code: Int, detail: org.json.JSONObject) {
        val payload = detail.toString()
        callbacks.removeIf { cb -> runCatching { cb.onServiceState(code, payload) }.isFailure }
    }

    private fun publishEvents() {
        val batch = runCatching { runtime.drainEvents(DRAIN_BATCH) }.getOrNull() ?: return
        val jsonl = batch.optString("jsonl")
        if (jsonl.isBlank()) return
        // Drop dead UI processes instead of accumulating them forever.
        callbacks.removeIf { cb -> runCatching { cb.onEvents(jsonl) }.isFailure }
    }

    private fun respond(label: String, block: () -> String): String =
        runCatching(block).getOrElse { fail(it) }

    private fun fail(error: Throwable): String {
        Log.e(TAG, "Agent runtime call failed", error)
        return org.json.JSONObject()
            .put("ok", false)
            .put("error", error.message ?: error.javaClass.simpleName)
            .toString()
    }

    private fun notification(text: String): Notification =
        Notification.Builder(this, CHANNEL)
            .setSmallIcon(android.R.drawable.stat_notify_sync)
            .setContentTitle(getString(R.string.service_agent_title))
            .setContentText(text)
            .setOngoing(true)
            .build()

    private fun String.requireSafeId(): String {
        require(matches(SAFE_ID)) { "Identificativo non valido" }
        return this
    }

    /** Keeps a caller-supplied path inside a directory the app owns. */
    private fun File.requireUnder(parent: File): File {
        val canonical = canonicalFile
        val parentPath = parent.canonicalFile
        require(canonical == parentPath || canonical.path.startsWith(parentPath.path + File.separator)) {
            "Percorso fuori da ${parentPath.name}"
        }
        return canonical
    }

    companion object {
        private const val TAG = "AgentService"
        private const val CHANNEL = "agent-runtime"
        private const val NOTIFICATION_ID = 20_491
        private const val DRAIN_BATCH = 256
        private val SAFE_ID = Regex("[A-Za-z0-9._-]{1,96}")

        const val STATE_READY = 1
        const val STATE_MODEL_LOADING = 2
        const val STATE_MODEL_READY = 3
        const val STATE_MODEL_UNLOADED = 4
        const val STATE_ERROR = 5
    }
}