package com.pocketworkbench.app

import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.os.IBinder
import android.util.Log
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.withContext
import java.io.File

/**
 * Binds to the agent runtime in the ":inference" process and gives the UI a
 * suspending API over it.
 *
 * Every runtime call that can block (pump, model load) goes through [scope] on
 * an IO dispatcher, so the chat thread is never the thread that waits for a
 * generation. Events arrive through the callback, not by polling.
 */
class AgentClient(private val context: Context) {

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private var remote: IPocketAgent? = null
    private var pending: CompletableDeferred<IPocketAgent>? = null
    private var closed = false

    var lastError: String? = null
        private set

    var modelBackend: String? = null
        private set

    var modelFallback: Boolean = false
        private set

    private val listener = object : IPocketAgentCallback.Stub() {
        override fun onEvents(jsonl: String?) {
            // oneway: never blocks the caller's thread, and a dead UI is harmless.
            onBatch(jsonl.orEmpty())
        }

        override fun onServiceState(code: Int, detailJson: String?) {
            val detail = runCatching { org.json.JSONObject(detailJson.orEmpty()) }.getOrElse { org.json.JSONObject() }
            when (code) {
                AgentService.STATE_MODEL_READY -> {
                    modelBackend = detail.optString("backend").ifBlank { null }
                    modelFallback = detail.optBoolean("fallback")
                }
                AgentService.STATE_MODEL_UNLOADED -> {
                    modelBackend = null
                    modelFallback = false
                }
                AgentService.STATE_ERROR -> lastError = detail.optString("error").ifBlank { "runtime error" }
            }
            onState(code, detail)
        }
    }

    @Volatile var onBatch: (String) -> Unit = {}
    @Volatile var onState: (Int, org.json.JSONObject) -> Unit = { _, _ -> }

    private val connection = object : ServiceConnection {
        override fun onServiceConnected(name: ComponentName?, binder: IBinder?) {
            // Cross-process: only the AIDL interface is valid here. Never cast
            // to the service's concrete binder class; that only works in-process
            // and silently skips registration on a real device.
            val api = IPocketAgent.Stub.asInterface(binder)
            if (api == null) {
                pending?.completeExceptionally(IllegalStateException("Unexpected binder"))
                pending = null
                return
            }
            remote = api
            runCatching { api.registerCallback(listener) }
            pending?.complete(api)
            pending = null
        }

        override fun onServiceDisconnected(name: ComponentName?) {
            remote = null
            lastError = "The agent runtime stopped"
        }

        override fun onBindingDied(name: ComponentName?) {
            remote = null
            lastError = "The agent runtime process died"
            onState(AgentService.STATE_ERROR, org.json.JSONObject().put("error", "process_died"))
        }
    }

    fun bind() {
        if (closed || remote != null) return
        val intent = Intent(context, AgentService::class.java)
        if (!context.bindService(intent, connection, Context.BIND_AUTO_CREATE)) {
            lastError = "Cannot bind the agent service"
            return
        }
    }

    fun unbind() {
        if (closed) return
        remote?.let { api -> runCatching { api.unregisterCallback(listener) } }
        runCatching { context.unbindService(connection) }
        remote = null
    }

    private suspend fun agent(): IPocketAgent {
        remote?.let { return it }
        val waiter = CompletableDeferred<IPocketAgent>()
        pending = waiter
        bind()
        return withContext(Dispatchers.IO) { waiter.await() }
    }

    /**
     * Wraps a Binder call, turning failures into a readable message.
     * Binder transactions block by design (pump runs a whole turn), so they
     * always run on Dispatchers.IO — never on the caller's (often Main) thread.
     */
    private suspend fun <T> call(label: String, block: suspend (IPocketAgent) -> T): T = try {
        val api = agent()
        withContext(Dispatchers.IO) { block(api) }
    } catch (e: Exception) {
        val message = when (e) {
            is AgentRuntimeException -> e.message ?: label
            else -> "$label failed: ${e.message ?: e.javaClass.simpleName}"
        }
        lastError = message
        Log.w(TAG, "$label failed", e)
        throw AgentRuntimeException(message)
    }

    suspend fun protocolVersion(): Int = call("protocolVersion") { it.protocolVersion() }

    suspend fun capabilities(): org.json.JSONObject = call("capabilities") { org.json.JSONObject(it.capabilities()) }

    suspend fun openSession(session: WorkbenchStore.Session, workspace: File, config: org.json.JSONObject) {
        call("openSession") {
            org.json.JSONObject(it.openSession(session.id, session.log.absolutePath, workspace.absolutePath, config.toString()))
        }
    }

    suspend fun closeSession(sessionId: String) {
        call("closeSession") { org.json.JSONObject(it.closeSession(sessionId)) }
    }

    suspend fun submit(sessionId: String, requestId: String, text: String): org.json.JSONObject =
        call("submit") { org.json.JSONObject(it.submit(sessionId, requestId, text)) }

    suspend fun pump(sessionId: String, maxTurns: Int = 1): org.json.JSONObject =
        call("pump") { org.json.JSONObject(it.pump(sessionId, maxTurns)) }

    suspend fun cancel(sessionId: String) {
        call("cancel") { org.json.JSONObject(it.cancel(sessionId)) }
    }

    suspend fun loadModel(sessionId: String, modelPath: String, backend: String, contextTokens: Int, threads: Int, useGpu: Boolean): org.json.JSONObject =
        call("loadModel") {
            org.json.JSONObject(it.loadModel(sessionId, modelPath, backend, contextTokens, threads, useGpu))
        }

    suspend fun unloadModel(sessionId: String) {
        call("unloadModel") { org.json.JSONObject(it.unloadModel(sessionId)) }
    }

    suspend fun sessionState(sessionId: String): org.json.JSONObject =
        call("sessionState") { org.json.JSONObject(it.sessionState(sessionId)) }

    suspend fun transcript(logPath: String): org.json.JSONObject =
        call("transcript") { org.json.JSONObject(it.transcript(logPath)) }

    fun close() {
        closed = true
        unbind()
    }

    private companion object {
        const val TAG = "AgentClient"
    }
}
