package com.pocketworkbench.app

import android.app.Application
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
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
import java.io.File

class WorkbenchViewModel(app: Application): AndroidViewModel(app) {
    private val store = PrivateStore(app)
    private val hub = HubClient()
    private val native = NativeEngine()
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
    private var generation: Job? = null
    private var recording: Job? = null
    private var recorder: AudioRecord? = null
    private val pending = mutableSetOf<String>()
    init {
        conversations.addAll(store.readHistory())
        if (conversations.isEmpty()) conversations.add(store.newChat())
        activeId = conversations.first().id
        refresh()
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
    private fun persist() { store.saveHistory(conversations.toList()) }
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
    fun send(text: String) {
        val model = selectedModel ?: run { status = "Download or import a GGUF model first"; return }
        val chat = active ?: return
        if (busy || text.isBlank()) return
        chat.messages.add(ChatMessage("user", text.trim()))
        if (chat.title == "New conversation") { val idx = conversations.indexOf(chat); conversations[idx] = chat.copy(title = text.take(45)); }
        val idx = conversations.indexOfFirst { it.id == chat.id }
        val started = conversations[idx]
        val replyIndex = started.messages.size
        started.messages.add(ChatMessage("assistant", ""))
        conversations[idx] = started.copy(messages = started.messages.toMutableList(), model = model.name)
        persist()
        busy = true; status = "Generating locally…"
        val snapshot = conversations[idx].messages.dropLast(1).takeLast(18)
        generation = viewModelScope.launch(Dispatchers.IO) {
            try {
                native.generate(model.file.absolutePath, snapshot.map { it.role }.toTypedArray(), snapshot.map { it.text }.toTypedArray(), object : NativeEngine.TokenCallback {
                    override fun onToken(piece: String) {
                        viewModelScope.launch(Dispatchers.Main) {
                            val i = conversations.indexOfFirst { it.id == chat.id }
                            if (i >= 0 && conversations[i].messages.size > replyIndex) {
                                val updated = conversations[i].copy(messages = conversations[i].messages.toMutableList())
                                val previous = updated.messages[replyIndex]
                                updated.messages[replyIndex] = previous.copy(text = previous.text + piece)
                                conversations[i] = updated
                            }
                        }
                    }
                })
                withContext(Dispatchers.Main) { status = "Ready" }
            } catch (e: Exception) { withContext(Dispatchers.Main) { status = e.message ?: "Generation failed" } }
            finally { withContext(Dispatchers.Main) { busy = false; delay(120); persist() } }
        }
    }
    fun stop() { native.stop(); status = "Stopping…" }
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
                    val text = native.transcribe(speech.file.absolutePath, samples.toFloatArray())
                    withContext(Dispatchers.Main) { transcript = text.trim(); status = "Review and edit the transcript before sending" }
                } catch (e: Exception) { withContext(Dispatchers.Main) { status = e.message ?: "Transcription failed" } }
            }
        }
    }
    fun stopRecording() { listening = false }
    fun clearTranscript() { transcript = "" }
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
    override fun onCleared() { native.stop(); listening = false; generation?.cancel(); recording?.cancel(); super.onCleared() }
}
