package com.pocketworkbench.app

import android.app.Application
import android.content.Context
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.SystemClock
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import androidx.compose.runtime.getValue
import androidx.compose.runtime.setValue
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.io.File

class WorkbenchViewModel(app: Application): AndroidViewModel(app) {
    private val store = PrivateStore(app)
    private val hub = HubClient()
    private val native = NativeEngine()
    private val gh = GitHubClient(app)
    private val perf = PerfLog(app)
    private val config = app.getSharedPreferences("workbench_config", Context.MODE_PRIVATE)
    val workspace: File get() = store.workspace
    var conversations = androidx.compose.runtime.mutableStateListOf<Conversation>(); private set
    var activeId by androidx.compose.runtime.mutableStateOf(""); private set
    var installed = androidx.compose.runtime.mutableStateListOf<LocalModel>(); private set
    var selectedModel by androidx.compose.runtime.mutableStateOf<LocalModel?>(null); private set
    var results = androidx.compose.runtime.mutableStateListOf<RemoteModel>(); private set
    var transfers = androidx.compose.runtime.mutableStateListOf<Transfer>(); private set
    var busy by androidx.compose.runtime.mutableStateOf(false); private set
    var listening by androidx.compose.runtime.mutableStateOf(false); private set
    var status by androidx.compose.runtime.mutableStateOf("Ready"); private set
    var transcript by androidx.compose.runtime.mutableStateOf(""); private set
    var terminalOutput by androidx.compose.runtime.mutableStateOf("Workspace is private to this app. Commands use Android's shell.\n"); private set

    // GitHub account state
    var ghLoggedIn by androidx.compose.runtime.mutableStateOf(false); private set
    var ghLogin by androidx.compose.runtime.mutableStateOf(""); private set
    var ghScopes by androidx.compose.runtime.mutableStateOf(""); private set
    var ghStatus by androidx.compose.runtime.mutableStateOf("Not signed in"); private set
    var ghUserCode by androidx.compose.runtime.mutableStateOf(""); private set
    var ghPolling by androidx.compose.runtime.mutableStateOf(false); private set
    var ghClientId by androidx.compose.runtime.mutableStateOf(gh.clientId)
    var agentMode by androidx.compose.runtime.mutableStateOf(config.getBoolean("agent_mode", false)); private set
    var toolStatus by androidx.compose.runtime.mutableStateOf(""); private set

    // llama.cpp #28878 (SIGSEGV in CPU path with 6+ threads on Android/aarch64):
    // default 4 compute threads, user-tunable in Models > Compute threads.
    var genThreads by androidx.compose.runtime.mutableStateOf(config.getInt("gen_threads", 4)); private set
    fun applyGenThreads(n: Int) {
        genThreads = n.coerceIn(1, 8)
        config.edit().putInt("gen_threads", genThreads).apply()
        Diag.log("config", "gen_threads=$genThreads")
    }

    private var generation: Job? = null
    private var ghJob: Job? = null
    private var recording: Job? = null
    private var recorder: AudioRecord? = null
    private val pending = mutableSetOf<String>()

    // ---------- Diagnostics: shared generation tracking ----------
    private val logDir: String by lazy { File(getApplication<Application>().filesDir, "logs").apply { mkdirs() }.absolutePath }
    private var genStartMs = 0L
    private var firstTokenMs = 0L
    private var tokenCount = 0        // counts JNI batches received (v0.2.5+ batches ~100ms of text each)
    private var nativeGenTokens = 0   // authoritative token count from native onStats (gen_tokens)
    private var lastUiFlushMs = 0L

    init {
        conversations.addAll(sanitizeHistory(store.readHistory()))
        if (conversations.isEmpty()) conversations.add(store.newChat())
        activeId = conversations.first().id
        refresh()
        Diag.log("vm", "init: ${conversations.size} chats, ${installed.size} models, agent=$agentMode, app=${Diag.appVersion()}")
        gh.token()?.let {
            viewModelScope.launch {
                try {
                    val (login, scopes) = gh.fetchUser()
                    ghLogin = login; ghScopes = scopes; ghLoggedIn = true
                    ghStatus = "Signed in as $login"
                } catch (e: Exception) { ghStatus = "Stored token rejected: ${e.message}" }
            }
        }
    }
    val active: Conversation? get() = conversations.find { it.id == activeId }
    fun refresh() {
        installed.clear(); installed.addAll(store.localModels())
        if (selectedModel?.file?.exists() != true) selectedModel = installed.firstOrNull { !it.speech }
    }
    fun chooseModel(model: LocalModel) { if (busy) return; selectedModel = model; status = "Selected ${model.name}" }
    fun selectChat(id: String) { if (!busy) activeId = id }
    fun newChat() { if (!busy) { val c = store.newChat(); conversations.add(0, c); activeId = c.id; persist() } }
    fun deleteChat(id: String) { if (busy) return; conversations.removeAll { it.id == id }; if (conversations.isEmpty()) conversations.add(store.newChat()); if (activeId == id) activeId = conversations.first().id; persist() }
    fun deleteModel(model: LocalModel) {
        if (busy || pending.contains(model.file.name)) return
        if (!model.file.delete()) { status = "Could not delete model"; return }
        refresh(); status = "Deleted ${model.name}"
    }
    private fun persist() {
        // Trim trailing empty assistant placeholders before saving: a process death
        // mid-generation used to leave a permanent "Thinking…" bubble after restart.
        val cleaned = conversations.map { c ->
            val msgs = c.messages.toMutableList()
            while (msgs.isNotEmpty() && msgs.last().role == "assistant" && msgs.last().text.isBlank() && msgs.last().perf.isBlank()) msgs.removeAt(msgs.lastIndex)
            c.copy(messages = msgs)
        }
        try { store.saveHistory(cleaned) } catch (e: Exception) { Diag.log("persist", "saveHistory FAILED: ${e.message}") }
    }

    /** Drop trailing empty assistant placeholders from loaded history and mark old
     *  mid-history empties clearly (they were silent "Thinking…" zombies). */
    private fun sanitizeHistory(chats: List<Conversation>): List<Conversation> = chats.map { c ->
        val msgs = c.messages.toMutableList()
        while (msgs.isNotEmpty() && msgs.last().role == "assistant" && msgs.last().text.isBlank() && msgs.last().perf.isBlank()) msgs.removeAt(msgs.lastIndex)
        val marked = msgs.map { if (it.role == "assistant" && it.text.isBlank()) it.copy(text = "(generation interrupted)") else it }
        c.copy(messages = marked.toMutableList())
    }

    // ---------- GitHub sign-in (OAuth Device Flow) ----------
    fun saveGhClientId(value: String) {
        gh.clientId = value; ghClientId = gh.clientId
        ghStatus = if (gh.clientId.isBlank()) "Client ID cleared" else "Client ID saved"
    }
    fun startGhLogin() {
        if (ghPolling) return
        if (gh.clientId.isBlank()) { ghStatus = "Create an OAuth App with Device Flow enabled, then paste its Client ID here"; return }
        ghJob = viewModelScope.launch {
            try {
                ghPolling = true; ghUserCode = ""
                ghStatus = "Requesting device code from GitHub…"
                val device = gh.deviceCodeStart()
                ghUserCode = device.userCode
                ghStatus = "Enter this code on github.com/login/device"
                val token = gh.pollForToken(device) { note -> ghStatus = note }
                gh.saveToken(token)
                val (login, scopes) = gh.fetchUser()
                ghLogin = login; ghScopes = scopes; ghLoggedIn = true
                ghUserCode = ""
                ghStatus = "Signed in as $login (scopes: ${scopes.ifBlank { "repo workflow" }})"
            } catch (e: CancellationException) { ghStatus = "Sign-in cancelled" }
            catch (e: Exception) { ghStatus = e.message ?: "Sign-in failed"; ghUserCode = "" }
            finally { ghPolling = false }
        }
    }
    fun cancelGhLogin() { ghJob?.cancel(); ghPolling = false; ghUserCode = ""; ghStatus = "Sign-in cancelled" }
    fun signOutGh() {
        gh.clearToken(); ghLoggedIn = false; ghLogin = ""; ghScopes = ""
        ghStatus = "Signed out; token removed from this device"
    }
    fun testGh() = viewModelScope.launch {
        ghStatus = "Testing GitHub connection…"
        try { val (login, scopes) = gh.fetchUser(); ghLogin = login; ghScopes = scopes; ghLoggedIn = true; ghStatus = "Connection OK: $login (scopes: $scopes)" }
        catch (e: Exception) { ghStatus = e.message ?: "Connection failed" }
    }
    fun toggleAgentMode() {
        agentMode = !agentMode
        config.edit().putBoolean("agent_mode", agentMode).apply()
        if (agentMode && !ghLoggedIn) status = "Agent mode on: sign in on the GitHub page to enable tools"
    }

    // ---------- Diagnostics API (Stats page) ----------
    val diagRecent: List<String> get() = Diag.recent(40)
    fun diagSessionInfo(): String = Diag.sessionInfo
    fun diagLastCrash(): String? = Diag.lastCrashReport
    fun diagPreviousEnd(): String? = Diag.previousEndSummary
    fun appVersion(): String = Diag.appVersion()
    fun exportDiag(): String = Diag.snapshot()
    fun clearDiag() { Diag.clearCrashMarkers(); status = "Diagnostics cleared" }

    // ---------- Performance log ----------
    val perfEntries: List<PerfEntry> get() = perf.entries
    fun exportPerfLog(): String = perf.exportText()
    fun clearPerfLog() { perf.clear(); status = "Performance log cleared" }
    fun deviceSummary(): String = perf.device.summary()

    // ---------- Hub models ----------
    fun search(query: String) = viewModelScope.launch {
        if (query.isBlank()) return@launch
        status = "Searching Hugging Face…"
        try { val found = hub.search(query); results.clear(); results.addAll(found); status = "${found.size} GGUF files found" }
        catch (e: Exception) { status = e.message ?: "Search failed" }
    }
    fun download(remote: RemoteModel, speech: Boolean = false) {
        val destination = store.modelDestination(remote.repo, remote.filename, speech)
        if (!pending.add(destination.name)) return
        if (destination.exists()) { pending.remove(destination.name); status = "Already downloaded"; return }
        viewModelScope.launch {
            val id = destination.name
            transfers.add(Transfer(id, 0, remote.bytes, "Downloading"))
            var lastUpdate = 0L
            try {
                hub.download(remote, destination) { done, total ->
                    val now = System.currentTimeMillis()
                    if (now - lastUpdate > 200 || (total > 0 && done == total)) {
                        lastUpdate = now
                        viewModelScope.launch { val i = transfers.indexOfFirst { it.id == id }; if (i >= 0) transfers[i] = Transfer(id, done, total, "Downloading") }
                    }
                }
                status = "Saved ${remote.filename}"
                refresh()
                val i = transfers.indexOfFirst { it.id == id }; if (i >= 0) transfers[i] = Transfer(id, destination.length(), destination.length(), "Complete")
            } catch (e: Exception) {
                status = e.message ?: "Download failed"
                val i = transfers.indexOfFirst { it.id == id }; if (i >= 0) transfers[i] = Transfer(id, 0, remote.bytes, "Paused: ${e.message}")
            } finally { pending.remove(id) }
        }
    }
    fun installSpeechModel() = download(RemoteModel("ggerganov/whisper.cpp", "ggml-base.bin", -1, "main"), speech = true)
    fun importModel(uri: android.net.Uri) = viewModelScope.launch(Dispatchers.IO) {
        try {
            val resolver = getApplication<Application>().contentResolver
            val name = "imported_${System.currentTimeMillis()}.gguf"
            val dest = File(store.models, name)
            resolver.openInputStream(uri).use { source -> requireNotNull(source); dest.outputStream().use { source.copyTo(it) } }
            val signature = ByteArray(4); dest.inputStream().use { it.read(signature) }
            if (!signature.contentEquals(byteArrayOf(0x47, 0x47, 0x55, 0x46))) { dest.delete(); throw IllegalArgumentException("The selected file is not GGUF") }
            withContext(Dispatchers.Main) { refresh(); status = "Imported $name" }
        } catch (e: Exception) { withContext(Dispatchers.Main) { status = e.message ?: "Import failed" } }
    }

    // ---------- Chat: message bookkeeping ----------
    private fun updateChat(chatId: String, transform: (Conversation) -> Conversation) {
        val i = conversations.indexOfFirst { it.id == chatId }
        if (i >= 0) conversations[i] = transform(conversations[i])
    }
    private fun streamAbsolute(chatId: String, replyIndex: Int, buffer: StringBuilder) {
        viewModelScope.launch(Dispatchers.Main) {
            flushBuffer(chatId, replyIndex, buffer)
        }
    }
    // Throttled streaming: per-token Main-thread updates used to storm the UI
    // (one coroutine launch per token). Now at most ~11 updates/sec.
    private fun streamThrottled(chatId: String, replyIndex: Int, buffer: StringBuilder) {
        val now = SystemClock.elapsedRealtime()
        if (now - lastUiFlushMs < 90 && tokenCount % 8 != 0) return
        lastUiFlushMs = now
        streamAbsolute(chatId, replyIndex, buffer)
    }
    private fun flushBuffer(chatId: String, replyIndex: Int, buffer: StringBuilder) {
        updateChat(chatId) { chat ->
            if (chat.messages.size > replyIndex) {
                val messages = chat.messages.toMutableList()
                messages[replyIndex] = messages[replyIndex].copy(text = buffer.toString())
                chat.copy(messages = messages)
            } else chat
        }
    }
    private fun attachPerf(chatId: String, replyIndex: Int, chip: String) {
        updateChat(chatId) { chat ->
            if (chat.messages.size > replyIndex) {
                val messages = chat.messages.toMutableList()
                messages[replyIndex] = messages[replyIndex].copy(perf = chip)
                chat.copy(messages = messages)
            } else chat
        }
    }
    private fun addMessage(chatId: String, message: ChatMessage) {
        // Synchronous mutation on the (thread-safe) snapshot list so agent-loop
        // steps that follow immediately can read the message.
        updateChatBlocking(chatId) { chat -> chat.copy(messages = chat.messages.toMutableList().apply { add(message) }) }
    }

    // ---------- Chat: generation ----------
    fun send(text: String) {
        val model = selectedModel ?: run { status = "Download or import a GGUF model first"; return }
        val chat = active ?: return
        if (busy || text.isBlank()) return
        val useAgent = agentMode && ghLoggedIn
        if (agentMode && !ghLoggedIn) status = "Agent mode needs GitHub sign-in; replying locally for now"
        chat.messages.add(ChatMessage("user", text.trim()))
        if (chat.title == "New conversation") { val idx = conversations.indexOf(chat); conversations[idx] = chat.copy(title = text.take(45)); }
        val idx = conversations.indexOfFirst { it.id == chat.id }
        val started = conversations[idx]
        val replyIndex = started.messages.size
        started.messages.add(ChatMessage("assistant", ""))
        conversations[idx] = started.copy(messages = started.messages.toMutableList(), model = model.name)
        persist()
        genStartMs = SystemClock.elapsedRealtime(); firstTokenMs = 0L; tokenCount = 0; nativeGenTokens = 0; lastUiFlushMs = 0L
        Diag.log("chat", "send: model=${model.name} chars=${text.trim().length} agent=$useAgent replyIndex=$replyIndex")
        Diag.updateState("generating", "model=${model.name}")
        busy = true; toolStatus = ""
        status = if (useAgent) "Agent: thinking…" else "Generating locally…"
        generation = viewModelScope.launch(Dispatchers.IO) {
            val heartbeat = startHeartbeat(model.name)
            try {
                if (useAgent) runAgentTurn(chat.id, model) else runLocalTurn(chat.id, model)
                Diag.updateState("idle", "turn complete")
            } catch (e: Exception) {
                Diag.log("gen", "turn FAILED: ${e.javaClass.simpleName}: ${e.message}")
                withContext(Dispatchers.Main) { status = e.message ?: "Generation failed" }
            }
            finally {
                heartbeat.cancel()
                withContext(Dispatchers.Main) { busy = false; delay(120) }
                persist()
                Diag.log("chat", "turn finished: tokens=${if (nativeGenTokens > 0) nativeGenTokens else tokenCount} ttft=${firstTokenMs}ms")
            }
        }
    }

    // While generating: periodic liveness + memory evidence, plus a periodic
    // history save so even a hard crash keeps the partial response on disk.
    private fun startHeartbeat(modelName: String): Job = viewModelScope.launch(Dispatchers.IO) {
        while (true) {
            delay(3000)
            if (!busy) break
            Diag.log("gen", "heartbeat: model=$modelName chunks=$tokenCount ttft=${firstTokenMs}ms ram=${Diag.freeRamMb()}MB pss=${Diag.pssMb()}MB thermal=${Diag.thermalName()} fg=${Diag.foreground}")
            Diag.updateState("generating", "chunks=$tokenCount model=$modelName")
            persist()
        }
    }

    private fun modelSnapshot(chatId: String, uptoReplyIndex: Int, includeSystem: Boolean): Pair<Array<String>, Array<String>> {
        val chat = conversations.find { it.id == chatId } ?: return emptyArray<String>() to emptyArray<String>()
        val pre = mutableListOf<ChatMessage>()
        if (includeSystem) pre.add(ChatMessage("system", McpTools.systemPrompt()))
        chat.messages.take(uptoReplyIndex).forEach { m ->
            when {
                m.text.isBlank() -> {}
                m.role == "tool" -> {} // display-only marker
                m.role == "assistant" -> pre.add(m)
                else -> pre.add(m.copy(role = "user")) // user, tool_result, system from tools
            }
        }
        val window = pre.drop(if (includeSystem) 1 else 0).takeLast(19)
        val final = if (includeSystem) listOf(pre.first()) + window else window
        return final.map { it.role }.toTypedArray() to final.map { it.text }.toTypedArray()
    }

    private fun enriched(json: String): JSONObject = try {
        val o = JSONObject(json)
        if (firstTokenMs > 0) o.put("ttft_ms", firstTokenMs.toDouble())
        o.put("pss_mb", Diag.pssMb())
        o.put("thermal", Diag.thermalName())
        o.put("fg", Diag.foreground)
        o
    } catch (_: Exception) { JSONObject(json) }

    private suspend fun runLocalTurn(chatId: String, model: LocalModel) {
        val chat = conversations.find { it.id == chatId } ?: return
        val replyIndex = chat.messages.size - 1
        val (roles, texts) = modelSnapshot(chatId, replyIndex, includeSystem = false)
        val buffer = StringBuilder()
        Diag.log("gen", "local turn start: promptMsgs=${roles.size} threads=$genThreads replyIndex=$replyIndex")
        native.generate(model.file.absolutePath, roles, texts, object : NativeEngine.TokenCallback {
            override fun onToken(piece: String) {
                if (firstTokenMs == 0L) { firstTokenMs = SystemClock.elapsedRealtime() - genStartMs; Diag.log("gen", "first token after ${firstTokenMs}ms (incl. any model load)") }
                buffer.append(piece); tokenCount++
                streamThrottled(chatId, replyIndex, buffer)
            }
            override fun onStats(json: String) {
                nativeGenTokens += try { JSONObject(json).optInt("gen_tokens", 0) } catch (_: Exception) { 0 }
                Diag.log("stats", "received: $json")
                val entry = try { perf.record(model.name, enriched(json)) } catch (e: Exception) { Diag.log("stats", "record FAILED: ${e.message}"); null }
                entry?.let { viewModelScope.launch(Dispatchers.Main) { attachPerf(chatId, replyIndex, it.chip()) } }
            }
        }, logDir, genThreads)
        flushBuffer(chatId, replyIndex, buffer) // guarantee final text shows despite throttling
        persist()
        Diag.log("gen", "local turn done: tokens=${if (nativeGenTokens > 0) nativeGenTokens else tokenCount} (chunks=$tokenCount) ttft=${firstTokenMs}ms chars=${buffer.length}")
        withContext(Dispatchers.Main) { status = "Ready" }
    }

    private suspend fun runAgentTurn(chatId: String, model: LocalModel) {
        val chat = conversations.find { it.id == chatId } ?: return
        var replyIndex = chat.messages.size - 1
        var lastEntry: PerfEntry? = null
        for (step in 0 until 6) {
            val (roles, texts) = modelSnapshot(chatId, replyIndex, includeSystem = true)
            val buffer = StringBuilder()
            withContext(Dispatchers.Main) { status = if (step == 0) "Agent: thinking…" else "Agent: step ${step + 1}" }
            Diag.log("agent", "step $step start: replyIndex=$replyIndex promptMsgs=${roles.size}")
            native.generate(model.file.absolutePath, roles, texts, object : NativeEngine.TokenCallback {
                override fun onToken(piece: String) {
                    if (firstTokenMs == 0L) { firstTokenMs = SystemClock.elapsedRealtime() - genStartMs; Diag.log("gen", "first token after ${firstTokenMs}ms (incl. any model load)") }
                    buffer.append(piece); tokenCount++
                    streamThrottled(chatId, replyIndex, buffer)
                }
                override fun onStats(json: String) {
                    nativeGenTokens += try { JSONObject(json).optInt("gen_tokens", 0) } catch (_: Exception) { 0 }
                    Diag.log("stats", "step $step received: $json")
                    val entry = try { perf.record(model.name, enriched(json)) } catch (e: Exception) { Diag.log("stats", "record FAILED: ${e.message}"); null }
                    if (entry != null) { lastEntry = entry; viewModelScope.launch(Dispatchers.Main) { attachPerf(chatId, replyIndex, entry.chip()) } }
                }
            }, logDir, genThreads)
            flushBuffer(chatId, replyIndex, buffer) // full step text in UI before parse/replace
            val parsed = McpTools.parse(buffer.toString())
            if (parsed.call == null) {
                if (parsed.attempted) {
                    Diag.log("agent", "step $step: malformed tool call, retrying with correction prompt")
                    addMessage(chatId, ChatMessage("tool_result", "[TOOL RESULT] error: malformed tool call. Use <tool>{\"name\":\"…\",\"arguments\":{…}}</tool> or answer in plain text."))
                    updateChatBlocking(chatId) { it.copy(messages = it.messages.toMutableList().apply { add(ChatMessage("assistant", "")) }) }
                    replyIndex = (conversations.find { it.id == chatId }?.messages?.size ?: 1) - 1
                    continue
                }
                Diag.log("agent", "step $step: final plain answer (${buffer.length} chars)")
                break // plain final answer: turn complete
            }
            val call = parsed.call
            val summary = call.arguments.let { args ->
                args.keys().asSequence().take(4).joinToString(", ") { key -> "$key=${args.optString(key).take(48)}" }
            }
            Diag.log("agent", "step $step: tool call ${call.name}($summary)")
            withContext(Dispatchers.Main) {
                updateChat(chatId) { conversation ->
                    val messages = conversation.messages.toMutableList()
                    if (messages.size > replyIndex) messages[replyIndex] = messages[replyIndex].copy(text = "🔧 ${call.name}($summary)")
                    conversation.copy(messages = messages)
                }
                toolStatus = "Running ${call.name}…"
                status = "Agent: ${call.name}"
            }
            val result = McpTools.execute(call, gh)
            Diag.log("agent", "step $step: tool ${call.name} result: ${result.take(160).replace('\n', ' ')}")
            addMessage(chatId, ChatMessage("tool_result", "[TOOL RESULT name=${call.name}]\n$result"))
            updateChatBlocking(chatId) { it.copy(messages = it.messages.toMutableList().apply { add(ChatMessage("assistant", "")) }) }
            replyIndex = (conversations.find { it.id == chatId }?.messages?.size ?: 1) - 1
        }
        lastEntry?.let { entry -> withContext(Dispatchers.Main) { attachPerf(chatId, replyIndex, entry.chip()) } }
        withContext(Dispatchers.Main) { status = "Ready"; toolStatus = "" }
    }

    // Synchronous append used inside the agent loop (IO thread) so the next
    // generation sees the placeholder message even before Main dispatches.
    private fun updateChatBlocking(chatId: String, transform: (Conversation) -> Conversation) {
        val i = conversations.indexOfFirst { it.id == chatId }
        if (i >= 0) conversations[i] = transform(conversations[i])
    }

    fun stop() {
        Diag.log("gen", "user requested stop at chunks=$tokenCount (ttft=${firstTokenMs}ms)")
        native.stop(); status = "Stopping…"
    }

    // ---------- Voice ----------
    fun startRecording() {
        if (listening) { stopRecording(); return }
        if (installed.none { it.speech }) { status = "Download the offline speech model in Models first"; return }
        val min = AudioRecord.getMinBufferSize(16000, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
        if (min <= 0) { status = "Microphone is unavailable"; return }
        val audio = AudioRecord(MediaRecorder.AudioSource.MIC, 16000, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT, min * 2)
        if (audio.state != AudioRecord.STATE_INITIALIZED) { audio.release(); status = "Could not open microphone"; return }
        recorder = audio; listening = true; status = "Recording; tap microphone to transcribe"
        recording = viewModelScope.launch(Dispatchers.IO) {
            val samples = ArrayList<Float>()
            try {
                audio.startRecording()
                val buffer = ShortArray(2048)
                while (isActive && listening && samples.size < 16000 * 30) {
                    val n = audio.read(buffer, 0, buffer.size)
                    if (n > 0) for (i in 0 until n) samples.add(buffer[i] / 32768f)
                }
            } catch (e: Exception) { withContext(Dispatchers.Main) { status = e.message ?: "Recording failed" } }
            finally {
                try { audio.stop() } catch (_: Exception) {}
                audio.release(); recorder = null
                withContext(Dispatchers.Main) { listening = false }
            }
            if (samples.isNotEmpty()) {
                withContext(Dispatchers.Main) { status = "Transcribing offline…" }
                try {
                    val speech = installed.first { it.speech }
                    Diag.log("voice", "transcribe start (${samples.size} samples)")
                    val text = native.transcribe(speech.file.absolutePath, samples.toFloatArray(), logDir)
                    Diag.log("voice", "transcribe done: ${text.length} chars")
                    withContext(Dispatchers.Main) { transcript = text.trim(); status = "Review and edit the transcript before sending" }
                } catch (e: Exception) { withContext(Dispatchers.Main) { status = e.message ?: "Transcription failed" } }
            }
        }
    }
    fun stopRecording() { listening = false }
    fun clearTranscript() { transcript = "" }

    // ---------- Workspace shell ----------
    fun runShell(command: String) = viewModelScope.launch(Dispatchers.IO) {
        if (command.isBlank()) return@launch
        withContext(Dispatchers.Main) { terminalOutput += "\n$ $command\n" }
        try {
            val process = ProcessBuilder("/system/bin/sh", "-c", command)
                .directory(workspace).redirectErrorStream(true).start()
            val reader = process.inputStream.bufferedReader()
            var length = 0
            while (true) {
                val line = reader.readLine() ?: break
                length += line.length
                if (length > 128_000) { process.destroyForcibly(); throw IllegalStateException("Output limit exceeded") }
                withContext(Dispatchers.Main) { terminalOutput += "$line\n" }
            }
            val code = process.waitFor()
            withContext(Dispatchers.Main) { terminalOutput += "Exit code: $code\n" }
        } catch (e: Exception) { withContext(Dispatchers.Main) { terminalOutput += "Error: ${e.message}\n" } }
    }
    override fun onCleared() {
        Diag.log("vm", "onCleared (activity torn down) busy=$busy tokens=$tokenCount")
        native.stop(); listening = false; generation?.cancel(); ghJob?.cancel(); recording?.cancel(); super.onCleared()
    }
}
