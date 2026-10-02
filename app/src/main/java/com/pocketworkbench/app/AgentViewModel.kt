package com.pocketworkbench.app

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import org.json.JSONObject
import java.io.File
import java.util.UUID

/** The five states the orb can show. Derived from real runtime events only. */
enum class AgentMood { IDLE, LOADING, THINKING, WORKING, ERROR }

data class ToolActivity(
    val callId: String,
    val name: String,
    val arguments: String,
    val result: String,
    val ok: Boolean,
    val running: Boolean,
    val durationMs: Long,
    val truncated: Boolean,
)

sealed interface ChatEntry {
    val key: String

    data class User(val id: String, val text: String, val queued: Boolean) : ChatEntry {
        override val key = "user:$id"
    }

    data class Assistant(val id: String, val text: String, val streaming: Boolean) : ChatEntry {
        override val key = "assistant:$id"
    }

    data class Reasoning(val id: String, val text: String) : ChatEntry {
        override val key = "reasoning:$id"
    }

    data class Tool(val id: String, val activity: ToolActivity) : ChatEntry {
        override val key = "tool:$id"
    }

    data class Notice(val id: String, val level: String, val message: String) : ChatEntry {
        override val key = "notice:$id"
    }

    data class Checkpoint(val id: String, val summary: String, val dropped: Int) : ChatEntry {
        override val key = "checkpoint:$id"
    }
}

data class AgentUiState(
    val sessionId: String? = null,
    val projectId: String = "",
    val entries: List<ChatEntry> = emptyList(),
    val mood: AgentMood = AgentMood.IDLE,
    val status: String = "",
    val busy: Boolean = false,
    val pending: Int = 0,
    val backend: String? = null,
    val backendFallback: Boolean = false,
    val modelName: String? = null,
    val lastUsage: String = "",
    val lastStop: String? = null,
    val lastPrompt: String = "",
    val error: String? = null,
    val interrupted: Boolean = false,
    /** Human-readable context breakdown from sessionState ("2.1k/8k · prefix a1b2…"). */
    val contextInfo: String = "",
)

/**
 * Drives the chat from the runtime's events.
 *
 * The transcript is not accumulated here from scratch: durable entries come from
 * the projection the runtime reads out of the session log, and only token deltas
 * are appended locally, until the durable record catches up. That keeps the UI
 * and the log from ever showing two different histories.
 */
class AgentViewModel(application: Application) : AndroidViewModel(application) {

    private val store = WorkbenchStore(application)
    private val client = AgentClient(application)
    private var pumpJob: Job? = null
    private var deltaBuffer = StringBuilder()
    private var reasoningBuffer = StringBuilder()
    private var liveTurn: String? = null

    private val _state = MutableStateFlow(AgentUiState())
    val state: StateFlow<AgentUiState> = _state.asStateFlow()

    val projects: List<WorkbenchStore.Project> get() = store.projects()
    val sessions: List<WorkbenchStore.Session> get() = store.sessions()
    val models: List<WorkbenchStore.ModelEntry> get() = store.models()
    val storeRef: WorkbenchStore get() = store

    init {
        client.onBatch = ::onEvents
        client.onState = { code, detail ->
            when (code) {
                AgentService.STATE_MODEL_LOADING -> _state.update { it.copy(mood = AgentMood.LOADING, status = "Loading the model…") }
                AgentService.STATE_MODEL_READY -> _state.update {
                    it.copy(
                        mood = if (it.busy) it.mood else AgentMood.IDLE,
                        backend = detail.optString("backend").ifBlank { null },
                        backendFallback = detail.optBoolean("fallback"),
                        modelName = detail.optString("model").ifBlank { null },
                    )
                }
                AgentService.STATE_MODEL_UNLOADED -> _state.update { it.copy(backend = null, modelName = null) }
                AgentService.STATE_ERROR -> _state.update { it.copy(mood = AgentMood.ERROR, error = detail.optString("error").ifBlank { "runtime error" }) }
            }
        }
        viewModelScope.launch {
            runCatching { client.bind() }
        }
    }

    override fun onCleared() {
        pumpJob?.cancel()
        client.close()
        super.onCleared()
    }

    // ------------------------------------------------------------------ session

    /**
     * Switching project changes the folder the agent works on. Sessions already
     * exist per project, so this only retargets the workspace.
     */
    fun openProject(projectId: String) {
        val sessionId = _state.value.sessionId
        if (sessionId == null) {
            _state.update { it.copy(projectId = projectId, error = null) }
            return
        }
        val session = store.session(sessionId)
        viewModelScope.launch {
            runCatching { client.openSession(session, store.project(projectId).workspace, sessionConfig()) }
                .onSuccess { _state.update { it.copy(projectId = projectId, error = null) }; refreshTranscript() }
                .onFailure { error -> fail(error) }
        }
    }

    /** Opens a session from the UI thread; the work happens in the runtime. */
    fun selectSession(session: WorkbenchStore.Session) {
        viewModelScope.launch { openSession(session) }
    }

    suspend fun openSession(session: WorkbenchStore.Session) {
        val project = store.project(if (session.projectId.isBlank()) "default" else session.projectId)
        val config = sessionConfig()
        runCatching { client.openSession(session, project.workspace, config) }
            .onFailure { error -> fail(error) }
            .onSuccess {
                _state.update { it.copy(sessionId = session.id, projectId = project.id, error = null) }
                refreshTranscript()
            }
    }

    fun newSession(projectId: String = _state.value.projectId.ifBlank { "default" }) {
        pumpJob?.cancel()
        val session = store.newSession(projectId)
        viewModelScope.launch {
            val project = store.project(projectId)
            runCatching { client.openSession(session, project.workspace, sessionConfig()) }
                .onSuccess {
                    _state.value = AgentUiState(sessionId = session.id, projectId = projectId, status = "Ready")
                }
                .onFailure { error -> fail(error) }
        }
    }

    // --------------------------------------------------------------------- send

    fun send(text: String) {
        val message = text.trim()
        val sessionId = _state.value.sessionId
        if (message.isEmpty() || sessionId == null) return
        pumpJob?.cancel()
        _state.update {
            it.copy(
                busy = true,
                mood = AgentMood.THINKING,
                status = "Thinking…",
                error = null,
                lastPrompt = message,
            )
        }
        pumpJob = viewModelScope.launch {
            val requestId = UUID.randomUUID().toString()
            try {
                // Seen on device: a message sent before any load dies
                // instantly with "No model is loaded for the NPU backend".
                // Load the selected model first instead of failing the turn.
                if (_state.value.backend == null && _state.value.modelName != null) {
                    if (!ensureLoaded(sessionId)) return@launch
                }
                client.submit(sessionId, requestId, message)
                refreshTranscript()
                // pump blocks in the runtime process until the turn ends. The
                // callback delivers events meanwhile, so the UI stays live.
                client.pump(sessionId, MAX_TURNS_PER_PUMP)
                refreshTranscript()
            } catch (e: Exception) {
                fail(e)
            }
        }
    }

    fun stop() {
        val sessionId = _state.value.sessionId ?: return
        viewModelScope.launch {
            runCatching { client.cancel(sessionId) }
            _state.update { it.copy(status = "Stopping…", mood = AgentMood.LOADING) }
        }
    }

    fun retry() {
        val prompt = _state.value.lastPrompt
        if (prompt.isNotBlank() && _state.value.sessionId != null) {
            _state.update { it.copy(error = null) }
            send(prompt)
        }
    }

    fun clearError() {
        _state.update { it.copy(error = null) }
    }

    fun dismissInterrupted() {
        _state.update { it.copy(interrupted = false) }
    }

    // --------------------------------------------------------------------- model

    fun loadSelectedModel() {
        val sessionId = _state.value.sessionId ?: return
        val entry = store.selectedModel() ?: store.models().firstOrNull() ?: return
        if (!entry.selected) store.selectModel(entry.id)
        _state.update { it.copy(mood = AgentMood.LOADING, status = "Loading ${entry.name}…", error = null) }
        viewModelScope.launch {
            ensureLoaded(sessionId, entry.name)
        }
    }

    /**
     * Loads [modelName] (or the stored selection) into the session and reports
     * what actually runs, including fallbacks. Returns false when nothing
     * could be loaded, so send() aborts the turn instead of recording a
     * doomed "no model" error. Shared by the manual button and auto-load.
     */
    private suspend fun ensureLoaded(sessionId: String, modelName: String? = null): Boolean {
        val entry = store.selectedModel() ?: store.models().firstOrNull() ?: return false
        if (!entry.selected) store.selectModel(entry.id)
        val name = modelName ?: entry.name
        _state.update { it.copy(mood = AgentMood.LOADING, status = "Loading $name…", error = null) }
        val settings = store.settingsJson()
        return try {
            val info = client.loadModel(
                sessionId,
                entry.file.absolutePath,
                settings.optString("backend", "rust-cpu"),
                settings.optInt("context_tokens", 8192),
                settings.optInt("threads", 4),
                settings.optBoolean("use_gpu", false),
            )
            val loaded = info.optJSONObject("info")
            _state.update {
                it.copy(
                    modelName = entry.name,
                    backend = loaded?.optString("backend"),
                    backendFallback = loaded?.optBoolean("fallback") ?: false,
                    status = describeBackend(loaded),
                )
            }
            true
        } catch (e: Exception) {
            fail(e)
            false
        }
    }

    fun unloadModel() {
        val sessionId = _state.value.sessionId ?: return
        viewModelScope.launch {
            runCatching { client.unloadModel(sessionId) }
            _state.update { it.copy(backend = null, modelName = null, status = "Model released") }
        }
    }

    fun selectModel(id: String) {
        runCatching { store.selectModel(id) }
        _state.update { it.copy(modelName = id, backend = null, status = "Selected $id — load it to start") }
    }

    fun updateSetting(key: String, value: Any) {
        val settings = store.settingsJson()
        when (value) {
            is Boolean -> settings.put(key, value)
            is Int -> settings.put(key, value)
            is Float -> settings.put(key, value.toDouble())
            else -> settings.put(key, value.toString())
        }
        runCatching { store.saveSettings(settings) }
    }

    // ----------------------------------------------------------------- transcript

    private fun refreshTranscript() {
        val sessionId = _state.value.sessionId ?: return
        val log = store.session(sessionId).log
        viewModelScope.launch {
            runCatching { client.transcript(log.absolutePath) }
                .onSuccess { payload -> applyTranscript(payload, log) }
                .onFailure { error -> fail(error) }
        }
    }

    private fun applyTranscript(payload: JSONObject, log: File) {
        val items = payload.optJSONArray("items") ?: return
        val entries = mutableListOf<ChatEntry>()
        for (index in 0 until items.length()) {
            val item = items.optJSONObject(index) ?: continue
            val id = item.optString("seq")
            when (item.optString("kind")) {
                "user" -> entries += ChatEntry.User(id, item.optString("text"), item.optBoolean("queued"))
                "assistant" -> entries += ChatEntry.Assistant(id, item.optString("text"), false)
                "reasoning" -> entries += ChatEntry.Reasoning(id, item.optString("text"))
                "tool" -> entries += ChatEntry.Tool(
                    id,
                    ToolActivity(
                        callId = item.optString("call_id"),
                        name = item.optString("name"),
                        arguments = item.optJSONObject("arguments")?.toString().orEmpty(),
                        result = item.optJSONObject("result")?.toString().orEmpty(),
                        ok = item.optBoolean("ok", true),
                        running = item.optString("phase") == "called",
                        durationMs = item.optLong("duration_ms"),
                        truncated = item.optBoolean("truncated"),
                    ),
                )
                "error" -> entries += ChatEntry.Notice(id, item.optString("class").ifBlank { "error" }, item.optString("message"))
                "cancelled" -> entries += ChatEntry.Notice(id, "cancelled", "Stopped before this turn finished.")
                "checkpoint" -> entries += ChatEntry.Checkpoint(id, item.optString("summary"), item.optInt("dropped_items"))
            }
        }
        // Live token deltas are appended only while their turn is still open.
        // Once the turn ended, the durable record carries the full text —
        // re-adding the tail would print the message twice.
        val stillOpen = (payload.optJSONArray("open_turns")?.length() ?: 0) > 0
        val liveId = liveTurn
        if (stillOpen && liveId != null) {
            _state.value.entries.filterIsInstance<ChatEntry.Assistant>()
                .lastOrNull { it.streaming && it.id == liveId }
                ?.let { entries += it }
            _state.value.entries.filterIsInstance<ChatEntry.Reasoning>()
                .lastOrNull { it.id == liveId }
                ?.let { entries += it }
        }
        val interrupted = payload.optJSONArray("interrupted_turns")?.length() ?: 0
        _state.update {
            it.copy(
                entries = entries,
                interrupted = interrupted > 0,
                busy = it.busy && stillOpen,
                mood = if (it.busy && stillOpen) it.mood else if (it.error != null) AgentMood.ERROR else AgentMood.IDLE,
                status = if (it.busy && stillOpen) it.status else "Ready",
            )
        }
    }

    private fun onEvents(jsonl: String) {
        val records = jsonl.lineSequence().mapNotNull { line ->
            runCatching { JSONObject(line) }.getOrNull()
        }.toList()
        if (records.isEmpty()) return
        var usage = _state.value.lastUsage
        var stop: String? = _state.value.lastStop
        var ended = false
        records.forEach { record ->
            val kind = record.optString("kind")
            val turnId = record.optString("turn_id")
            // Event payloads live under "data" (see AgentEvent.to_json); the top
            // level only carries kind/seq/turn_id. Reading them from the top
            // level silently yields blanks, which is how usage/stop reason were
            // lost before.
            val data = record.optJSONObject("data") ?: JSONObject()
            when (kind) {
                "turn.started" -> liveTurn = turnId
                "delta.text" -> appendDelta(data.optString("text"), reasoning = false)
                "delta.reasoning" -> appendDelta(data.optString("text"), reasoning = true)
                "usage" -> usage = data.optJSONObject("usage")?.let { formatUsage(it) }.orEmpty()
                "tool.called" -> _state.update {
                    it.copy(mood = AgentMood.WORKING, status = "Using ${data.optString("name").ifBlank { "a tool" }}…")
                }
                "error" -> _state.update {
                    it.copy(
                        error = data.optString("message").ifBlank { "The turn failed" },
                        mood = AgentMood.ERROR,
                    )
                }
                "cancelled" -> _state.update { it.copy(status = "Stopped") }
                "turn.ended" -> {
                    ended = true
                    stop = data.optString("reason").ifBlank { null }
                    usage = data.optJSONObject("usage")?.let { formatUsage(it, data.optLong("duration_ms")) } ?: usage
                    // Drop the live streaming entries of the finished turn now:
                    // the refresh below rebuilds from the durable record, which
                    // already holds the full text. Without this the message
                    // would appear twice (live tail + durable entry).
                    val finished = record.optString("turn_id")
                    if (finished.isNotBlank()) {
                        _state.update { current ->
                            current.copy(
                                entries = current.entries.filterNot { entry ->
                                    (entry is ChatEntry.Assistant && entry.streaming && entry.id == finished) ||
                                        (entry is ChatEntry.Reasoning && entry.id == finished)
                                },
                            )
                        }
                    }
                }
            }
        }
        _state.update {
            it.copy(
                lastUsage = usage,
                lastStop = stop,
                mood = when {
                    ended -> if (it.error != null) AgentMood.ERROR else AgentMood.IDLE
                    it.busy && _state.value.entries.any { entry -> entry is ChatEntry.Tool && entry.activity.running } -> AgentMood.WORKING
                    it.busy -> AgentMood.THINKING
                    else -> it.mood
                },
                status = if (ended) describeStop(stop) else it.status,
            )
        }
        if (ended) {
            deltaBuffer = StringBuilder()
            reasoningBuffer = StringBuilder()
            liveTurn = null
            refreshTranscript()
            refreshContextInfo()
        }
    }

    /** Pulls the context breakdown (tokens per category, prefix hash) for the Stats page. */
    private fun refreshContextInfo() {
        val sessionId = _state.value.sessionId ?: return
        viewModelScope.launch {
            val info = runCatching { client.sessionState(sessionId) }.getOrNull() ?: return@launch
            val context = info.optJSONObject("context") ?: return@launch
            val total = context.optInt("total")
            val budget = context.optInt("budget")
            val hash = context.optString("prefix_hash").take(8)
            val parts = listOf("system", "user", "assistant", "tools", "summary")
                .mapNotNull { key ->
                    val value = context.optInt(key)
                    if (value > 0) "$key ${formatShort(value)}" else null
                }
                .joinToString(" · ")
            _state.update {
                it.copy(
                    contextInfo = if (total > 0 && budget > 0) {
                        "${formatShort(total)}/${formatShort(budget)} · prefix $hash · $parts"
                    } else {
                        ""
                    },
                )
            }
        }
    }

    private fun formatShort(tokens: Int): String = when {
        tokens >= 1000 -> "%.1fk".format(tokens / 1000.0)
        else -> tokens.toString()
    }

    /** Model family for the prompt overlay: minicpm/llama need their own temperament lines. */
    private fun modelFamilyOf(modelId: String): String {
        val id = modelId.lowercase()
        return when {
            "minicpm" in id -> "minicpm"
            "llama" in id -> "llama"
            else -> "qwen"
        }
    }

    /**
     * Buffers token deltas into the live assistant entry. Buffering matters: a
     * Compose recomposition per token would be unusable on this device, and the
     * durable record is what finally wins.
     */
    private fun appendDelta(piece: String, reasoning: Boolean) {
        if (piece.isEmpty()) return
        if (reasoning) reasoningBuffer.append(piece) else deltaBuffer.append(piece)
        val id = liveTurn ?: return
        if (reasoning) {
            val text = reasoningBuffer.toString()
            _state.update { current -> current.copy(entries = upsert(current.entries, ChatEntry.Reasoning(id, text))) }
            return
        }
        val text = deltaBuffer.toString()
        _state.update { current ->
            current.copy(
                entries = upsert(current.entries, ChatEntry.Assistant(id, text, true)),
                mood = if (current.mood == AgentMood.WORKING) AgentMood.WORKING else AgentMood.THINKING,
            )
        }
    }

    private fun upsert(entries: List<ChatEntry>, entry: ChatEntry): List<ChatEntry> {
        val index = entries.indexOfFirst { it.key == entry.key }
        if (index < 0) return entries + entry
        val copy = entries.toMutableList()
        copy[index] = entry
        return copy
    }

    // ------------------------------------------------------------------ helpers

    private fun nativeLibraryDir(): String =
        getApplication<android.app.Application>().applicationInfo.nativeLibraryDir

    private fun sessionConfig(): JSONObject = JSONObject()
        .put("workspace_id", _state.value.projectId.ifBlank { "default" })
        .put("backend", store.settingsJson().optString("backend", "rust-cpu"))
        .put("model_family", modelFamilyOf(_state.value.modelName.orEmpty()))
        .put("use_model_summary", store.settingsJson().optBoolean("use_model_summary", false))
        // Debian paths are deterministic (filesDir-based) and the runtime
        // validates the tree on every linux call, so installing afterwards
        // needs no session reopen.
        .put("linux", LinuxModule(getApplication()).configJson(nativeLibraryDir()))
        .put(
            "limits",
            JSONObject()
                .put("max_steps", store.settingsJson().optInt("max_steps", 12))
                .put("turn_timeout_ms", store.settingsJson().optLong("turn_timeout_ms", 600_000L))
                .put("max_tool_output_bytes", store.settingsJson().optInt("max_tool_output_bytes", 16 * 1024))
                .put("max_context_tokens", store.settingsJson().optInt("context_tokens", 8192))
                .put("max_repeated_tool_errors", store.settingsJson().optInt("max_tool_errors", 3))
                .put("max_queued_messages", store.settingsJson().optInt("max_queued", 8)),
        )
        .put(
            "generation",
            JSONObject()
                .put("max_tokens", store.settingsJson().optInt("max_tokens", 512))
                .put("temperature", store.settingsJson().optDouble("temperature", 0.7))
                .put("top_p", store.settingsJson().optDouble("top_p", 0.95))
                .put("threads", store.settingsJson().optInt("threads", 4)),
        )

    private fun formatUsage(usage: JSONObject, durationMs: Long = 0L): String {
        val prompt = usage.optInt("prompt_tokens")
        val cached = usage.optInt("cached_prompt_tokens")
        val completion = usage.optInt("completion_tokens")
        val prefill = usage.optDouble("prefill_ms")
        val decode = usage.optDouble("decode_ms")
        val parts = mutableListOf<String>()
        if (prompt > 0) parts += "prompt $prompt" + if (cached > 0) " (${cached} cached)" else ""
        if (completion > 0) parts += "reply $completion"
        if (prefill > 0.0) parts += "prefill ${prefill.toInt()}ms"
        if (decode > 0.0 && completion > 0) {
            parts += "decode ${(completion / (decode / 1000.0)).toInt()} tok/s"
        } else if (completion > 0 && durationMs > 0) {
            // Backends without decode timing (e.g. the NPU runner, which only
            // reports token counts): derive tok/s from the whole turn duration.
            parts += "≈${(completion / (durationMs / 1000.0)).toInt()} tok/s"
        }
        return parts.joinToString(" · ")
    }

    private fun describeBackend(info: JSONObject?): String {
        if (info == null) return "Ready"
        val backend = info.optString("backend", "unknown")
        val fallback = info.optBoolean("fallback")
        return if (fallback) "Running on $backend (requested ${info.optString("requested_backend")})" else "Running on $backend"
    }

    private fun describeStop(reason: String?): String = when (reason) {
        null, "", "eos" -> "Ready"
        "cancelled" -> "Stopped"
        "token_limit" -> "Stopped at the reply limit"
        "context_full" -> "Stopped: the context window is full"
        "step_limit" -> "Stopped: too many steps in one turn"
        "turn_timeout" -> "Stopped: the turn timed out"
        "tool_output_limit" -> "Stopped: a tool produced too much output"
        "repeated_tool_error" -> "Stopped: repeated tool errors"
        "no_model" -> "No model loaded"
        else -> "Stopped: $reason"
    }

    private fun fail(error: Throwable) {
        _state.update {
            it.copy(
                busy = false,
                mood = AgentMood.ERROR,
                error = error.message ?: "Unexpected error",
                status = "Error",
            )
        }
    }

    private companion object {
        const val MAX_TURNS_PER_PUMP = 1
    }
}