package com.pocketworkbench.app

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.IOException
import java.net.URLEncoder
import java.util.concurrent.TimeUnit

/**
 * Hugging Face: search, list and resumable download of GGUF weights.
 *
 * A download writes to `<name>.gguf.part` and is resumable with a `Range`
 * header. The revision the partial file belongs to is recorded next to it,
 * because resuming across a re-upload would silently produce a corrupt model.
 */
class ModelHub(private val models: File, private val freeBytes: () -> Long) {
    private val client = OkHttpClient.Builder()
        .connectTimeout(25, TimeUnit.SECONDS)
        .readTimeout(60, TimeUnit.SECONDS)
        .build()

    data class Remote(val repo: String, val filename: String, val bytes: Long, val revision: String, val gated: Boolean)

    suspend fun search(query: String): List<Remote> = withContext(Dispatchers.IO) {
        val encoded = URLEncoder.encode(query.trim(), "UTF-8")
        val array = JSONArray(get("https://huggingface.co/api/models?search=$encoded&filter=gguf&limit=20"))
        (0 until array.length())
            .map { array.getJSONObject(it).getString("id") }
            .distinct()
            .take(6)
            .flatMap { repo -> runCatching { files(repo).take(6) }.getOrDefault(emptyList()) }
    }

    suspend fun files(repo: String): List<Remote> = withContext(Dispatchers.IO) {
        require(Regex("[\\w.-]+/[\\w.-]+").matches(repo)) { "Invalid repository" }
        val info = JSONObject(get("https://huggingface.co/api/models/$repo"))
        val revision = info.optString("sha", "main")
        val gated = info.optBoolean("gated", false) || info.optString("gated") == "auto"
        val siblings = info.getJSONArray("siblings")
        (0 until siblings.length())
            .mapNotNull { index ->
                val item = siblings.getJSONObject(index)
                val filename = item.getString("rfilename")
                if (filename.contains('/') || !filename.endsWith(".gguf", true)) null
                else Remote(
                    repo,
                    filename,
                    item.optLong("size", item.optJSONObject("lfs")?.optLong("size") ?: -1L),
                    revision,
                    gated,
                )
            }
            .sortedBy { it.filename }
    }

    private fun get(url: String): String {
        val response = client.newCall(Request.Builder().url(url).header("User-Agent", AGENT).build()).execute()
        response.use {
            if (!it.isSuccessful) throw IOException("Hugging Face returned HTTP ${it.code}")
            return it.body?.string() ?: throw IOException("Empty response")
        }
    }

    /**
     * Downloads into `filesDir/models`, resumable, reporting progress. Returns the
     * destination file. Throws with a readable message; the caller decides what
     * to do about it.
     */
    suspend fun download(
        remote: Remote,
        onProgress: (Long, Long) -> Unit,
    ): File = withContext(Dispatchers.IO) {
        if (remote.gated) {
            throw IOException("Questo repository richiede l'approvazione di Hugging Face: importa un GGUF consentito.")
        }
        val name = remote.filename.removeSuffix(".gguf").take(120)
        val destination = File(models, "$name.gguf")
        val partial = File(models, "$name.gguf.part")
        val marker = File(models, "$name.gguf.revision")
        if (marker.isFile && marker.readText() != remote.revision) partial.delete()
        if (destination.isFile && destination.length() > 0 && remote.bytes > 0 && destination.length() == remote.bytes) {
            marker.delete()
            return@withContext destination
        }
        marker.writeText(remote.revision)
        val url = "https://huggingface.co/${remote.repo}/resolve/${remote.revision}/" +
            URLEncoder.encode(remote.filename, "UTF-8").replace("+", "%20") + "?download=true"
        val existing = if (partial.isFile) partial.length() else 0L
        val builder = Request.Builder().url(url).header("User-Agent", AGENT)
        if (existing > 0) builder.header("Range", "bytes=$existing-")
        client.newCall(builder.build()).execute().use { response ->
            if (response.code !in listOf(200, 206)) throw IOException("Download failed: HTTP ${response.code}")
            // A 200 for a request that asked for a range means the server ignored
            // it; appending would corrupt the file, so restart instead.
            val append = existing > 0 && response.code == 206 &&
                response.header("Content-Range").orEmpty().startsWith("bytes $existing-")
            val start = if (append) existing else 0L
            if (!append && existing > 0) partial.delete()
            val declared = response.body?.contentLength() ?: -1L
            val total = if (declared >= 0) start + declared else remote.bytes
            val reserve = 64L * 1024 * 1024
            if (total > 0 && total + reserve > freeBytes()) throw IOException("Spazio insufficiente: servono ${total / 1048576} MiB più una riserva")
            val body = response.body ?: throw IOException("Risposta vuota")
            var done = start
            body.byteStream().use { input ->
                java.io.FileOutputStream(partial, append).buffered().use { output ->
                    val buffer = ByteArray(256 * 1024)
                    while (true) {
                        val read = input.read(buffer)
                        if (read < 0) break
                        if (done + read > freeBytes() + reserve) throw IOException("Spazio esaurito durante il download")
                        output.write(buffer, 0, read)
                        done += read
                        onProgress(done, total)
                    }
                }
            }
            if (total > 0 && done != total) throw IOException("Download incompleto: $done di $total byte")
            if (remote.bytes > 0 && done != remote.bytes) throw IOException("Dimensione inattesa: $done byte")
            verifyGguf(partial)
            if (!partial.renameTo(destination)) {
                partial.delete()
                throw IOException("Cannot finish download")
            }
            marker.delete()
            destination
        }
    }

    private companion object {
        const val AGENT = "PocketWorkbench/1"

        fun verifyGguf(file: File) {
            require(file.length() >= 4) { "Il file scaricato è vuoto" }
            val head = ByteArray(4)
            file.inputStream().use { if (it.read(head) != 4) throw IOException("Il file non è leggibile") }
            require(head.contentEquals(byteArrayOf(0x47, 0x47, 0x55, 0x46.toByte()))) {
                "Il file scaricato non è un modello GGUF"
            }
        }
    }
}