package com.pocketworkbench.app

import android.content.Context
import android.os.StatFs
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.IOException
import java.net.URLEncoder
import java.util.UUID
import java.util.concurrent.TimeUnit

data class ChatMessage(val role: String, val text: String)
data class Conversation(val id: String, val title: String, val model: String, val messages: MutableList<ChatMessage>)
data class LocalModel(val repo: String, val name: String, val file: File, val speech: Boolean = false)
data class RemoteModel(val repo: String, val filename: String, val bytes: Long, val revision: String, val gated: Boolean = false)
data class Transfer(val id: String, val done: Long, val total: Long, val status: String)

class PrivateStore(private val context: Context) {
    val models = File(context.filesDir, "models").apply { mkdirs() }
    val workspace = File(context.filesDir, "workspace").apply { mkdirs() }
    private val history = File(context.filesDir, "history.json")
    fun availableBytes(): Long = StatFs(models.absolutePath).availableBytes
    fun localModels(): List<LocalModel> = models.listFiles()?.filter { it.isFile && it.extension.lowercase() in listOf("gguf", "bin") }
        ?.map { file -> LocalModel(file.name.substringBefore("--"), file.name, file, file.name.startsWith("speech--")) }?.sortedBy { it.name } ?: emptyList()
    fun readHistory(): List<Conversation> = try {
        val array = JSONArray(history.readText())
        (0 until array.length()).map { i ->
            val item = array.getJSONObject(i)
            val msgs = item.getJSONArray("messages")
            Conversation(item.getString("id"), item.getString("title"), item.optString("model"),
                (0 until msgs.length()).map { j -> val m = msgs.getJSONObject(j); ChatMessage(m.getString("role"), m.getString("text")) }.toMutableList())
        }
    } catch (_: Exception) { emptyList() }
    @Synchronized fun saveHistory(chats: List<Conversation>) {
        val array = JSONArray()
        chats.forEach { chat ->
            val msgs = JSONArray(); chat.messages.forEach { msgs.put(JSONObject().put("role", it.role).put("text", it.text)) }
            array.put(JSONObject().put("id", chat.id).put("title", chat.title).put("model", chat.model).put("messages", msgs))
        }
        val temp = File(history.parentFile, "history.tmp")
        temp.writeText(array.toString())
        if (!temp.renameTo(history)) { temp.delete(); throw IOException("Cannot save conversation history") }
    }
    fun newChat() = Conversation(UUID.randomUUID().toString(), "New conversation", "", mutableListOf())
    fun modelDestination(repo: String, filename: String, speech: Boolean): File {
        require(Regex("[\\w.-]+/[\\w.-]+").matches(repo) && Regex("[\\w. -]+\\.(gguf|bin)", RegexOption.IGNORE_CASE).matches(filename))
        return File(models, (if (speech) "speech--" else "") + repo.replace('/', '_') + "--" + filename)
    }
}

class HubClient {
    private val client = OkHttpClient.Builder().connectTimeout(25, TimeUnit.SECONDS).readTimeout(45, TimeUnit.SECONDS).build()
    private fun request(url: String): String {
        val response = client.newCall(Request.Builder().url(url).header("User-Agent", "PocketWorkbench/0.1").build()).execute()
        response.use { if (!it.isSuccessful) throw IOException("Hugging Face returned HTTP ${it.code}"); return it.body?.string() ?: throw IOException("Empty response") }
    }
    suspend fun search(query: String): List<RemoteModel> = withContext(Dispatchers.IO) {
        val q = URLEncoder.encode(query.trim(), "UTF-8")
        val array = JSONArray(request("https://huggingface.co/api/models?search=$q&filter=gguf&limit=24&full=false"))
        (0 until array.length()).map { array.getJSONObject(it).getString("id") }.flatMap { repo ->
            try { files(repo).take(12) } catch (_: Exception) { emptyList() }
        }
    }
    suspend fun files(repo: String): List<RemoteModel> = withContext(Dispatchers.IO) {
        require(Regex("[\\w.-]+/[\\w.-]+").matches(repo))
        val info = JSONObject(request("https://huggingface.co/api/models/$repo"))
        val commit = info.optString("sha", "main")
        val gated = info.optBoolean("gated", false) || info.optString("gated") == "auto"
        val siblings = info.getJSONArray("siblings")
        (0 until siblings.length()).mapNotNull { index ->
            val item = siblings.getJSONObject(index)
            val filename = item.getString("rfilename")
            if (filename.contains('/') || !filename.endsWith(".gguf", true)) null
            else RemoteModel(repo, filename, item.optLong("size", item.optJSONObject("lfs")?.optLong("size") ?: -1L), commit, gated)
        }.sortedBy { it.filename }
    }
    suspend fun download(model: RemoteModel, destination: File, onProgress: (Long, Long) -> Unit) = withContext(Dispatchers.IO) {
        if (model.gated) throw IOException("This repository needs Hugging Face access approval; import a permitted GGUF manually.")
        val partial = File(destination.path + ".part")
        val marker = File(destination.path + ".revision")
        if (marker.exists() && marker.readText() != model.revision) partial.delete()
        marker.writeText(model.revision)
        val url = "https://huggingface.co/${model.repo}/resolve/${model.revision}/${URLEncoder.encode(model.filename, "UTF-8").replace("+", "%20")}?download=true"
        val existing = partial.length()
        val builder = Request.Builder().url(url).header("User-Agent", "PocketWorkbench/0.1")
        if (existing > 0) builder.header("Range", "bytes=$existing-")
        client.newCall(builder.build()).execute().use { response ->
            if (response.code !in listOf(200, 206)) throw IOException("Download failed: HTTP ${response.code}")
            if (existing > 0 && response.code == 206 && !response.header("Content-Range", "").startsWith("bytes $existing-")) throw IOException("Server returned a mismatched byte range")
            val append = existing > 0 && response.code == 206
            val initial = if (append) existing else 0L
            val contentLength = response.body?.contentLength() ?: -1L
            val total = if (contentLength >= 0) initial + contentLength else model.bytes
            if (total > 0 && total + 256L * 1024 * 1024 > StatFs(destination.parentFile!!.absolutePath).availableBytes) throw IOException("Insufficient storage: need ${total / 1048576} MiB plus reserve")
            val body = response.body ?: throw IOException("Empty download")
            var done = initial
            body.byteStream().use { input -> partial.outputStream(append).buffered().use { output ->
                // A 200 response restarts the file instead of appending stale partial bytes.
                val buffer = ByteArray(256 * 1024)
                while (true) {
                    val n = input.read(buffer); if (n < 0) break
                    if (done + n > 0 && done + n > StatFs(destination.parentFile!!.absolutePath).availableBytes + partial.length()) throw IOException("Storage exhausted during download")
                    output.write(buffer, 0, n); done += n; onProgress(done, total)
                }
            } }
            if (total > 0 && done != total) throw IOException("Incomplete download: $done of $total bytes")
            if (model.bytes > 0 && done != model.bytes) throw IOException("Unexpected file size: $done bytes")
            if (partial.length() < 4) throw IOException("Downloaded file is empty")
            partial.inputStream().use { stream ->
                val head = ByteArray(4); stream.read(head)
                if (!destination.name.startsWith("speech--") && !head.contentEquals(byteArrayOf(0x47, 0x47, 0x55, 0x46))) throw IOException("The downloaded file is not a GGUF model")
            }
            if (!partial.renameTo(destination)) throw IOException("Cannot finish download")
            marker.delete()
        }
    }
}
