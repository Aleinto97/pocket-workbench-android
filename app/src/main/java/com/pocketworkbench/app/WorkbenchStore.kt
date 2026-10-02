package com.pocketworkbench.app

import android.content.Context
import android.os.StatFs
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.IOException
import java.util.UUID

/**
 * Everything the app keeps on disk, in the layout the plan fixes:
 *
 * ```
 * filesDir/
 *   models/                    GGUF weights, imported or downloaded
 *   workspaces/<project-id>/   the files a session works on
 *   sessions/<session-id>/     events.jsonl plus the session header
 *   settings/                  configuration
 *   logs/                      rotated diagnostics
 * ```
 *
 * Three separate ideas, deliberately: a GGUF is not a loaded model, and neither
 * is a workspace. A model serves many projects and never ends up inside a
 * project export.
 */
class WorkbenchStore(private val context: Context) {
    val root: File = context.filesDir
    val models: File = File(root, "models").apply { mkdirs() }
    val workspaces: File = File(root, "workspaces").apply { mkdirs() }
    val sessions: File = File(root, "sessions").apply { mkdirs() }
    val settings: File = File(root, "settings").apply { mkdirs() }
    val logs: File = File(root, "logs").apply { mkdirs() }

    // ---------------------------------------------------------------- projects

    data class Project(val id: String, val name: String, val updatedAt: Long, val workspace: File) {
        /** Display label: the folder name, which is the project id by construction. */
        val label: String get() = name
    }

    fun projects(): List<Project> =
        workspaces.listFiles { file -> file.isDirectory }
            ?.map { Project(it.name, it.name, it.lastModified(), it) }
            ?.sortedByDescending { it.updatedAt }
            .orEmpty()

    fun project(id: String): Project {
        require(SAFE_ID.matches(id)) { "Invalid project id" }
        val dir = File(workspaces, id)
        if (!dir.isDirectory) dir.mkdirs()
        return Project(id, id, dir.lastModified(), dir)
    }

    fun createProject(name: String): Project {
        val clean = name.trim().replace(Regex("[^A-Za-z0-9._ -]"), "").take(48).ifBlank { "project" }
        val id = "${clean.lowercase().replace(' ', '-')}-${UUID.randomUUID().toString().take(6)}"
        File(workspaces, id).mkdirs()
        return project(id)
    }

    fun deleteProject(id: String) {
        require(SAFE_ID.matches(id)) { "Invalid project id" }
        File(workspaces, id).deleteRecursively()
    }

    // ---------------------------------------------------------------- sessions

    data class Session(
        val id: String,
        val projectId: String,
        val title: String,
        val updatedAt: Long,
        val log: File,
    )

    fun sessions(): List<Session> {
        val out = mutableListOf<Session>()
        sessions.listFiles { file -> file.isFile && file.name.endsWith(".jsonl") }
            ?.sortedByDescending { it.lastModified() }
            ?.forEach { file ->
                val id = file.name.removeSuffix(".jsonl")
                val header = runCatching { JSONObject(file.useLines { lines -> lines.firstOrNull().orEmpty() }) }.getOrNull()
                val projectId = header?.optString("workspace_id").orEmpty().ifBlank { "default" }
                val title = header?.optString("title").orEmpty().ifBlank { "Session" }
                out += Session(id, projectId, title, file.lastModified(), file)
            }
        return out
    }

    fun session(id: String): Session {
        require(SAFE_ID.matches(id)) { "Invalid session id" }
        return Session(id, "default", "Session", 0L, File(sessions, "$id.jsonl"))
    }

    fun newSession(projectId: String, title: String = "New session"): Session {
        val id = "s-${UUID.randomUUID().toString().take(12)}"
        return Session(id, projectId, title, System.currentTimeMillis(), File(sessions, "$id.jsonl"))
    }

    fun deleteSession(id: String) {
        require(SAFE_ID.matches(id)) { "Invalid session id" }
        File(sessions, "$id.jsonl").delete()
        File(sessions, "$id.jsonl.tmp").delete()
    }

    // ------------------------------------------------------------------ models

    data class ModelEntry(
        val id: String,
        val name: String,
        val file: File,
        val bytes: Long,
        val selected: Boolean,
    )

    private val selection = File(settings, "selected-model.json")

    fun models(): List<ModelEntry> {
        val selected = runCatching { JSONObject(selection.readText()).optString("id") }.getOrNull()
        return models.listFiles { file -> file.isFile && file.name.endsWith(".gguf", true) }
            ?.map { ModelEntry(it.nameWithoutExtension, it.nameWithoutExtension, it, it.length(), it.name == selected) }
            ?.sortedBy { it.name }
            .orEmpty()
    }

    fun selectModel(name: String) {
        val file = File(models, "$name.gguf")
        require(file.isFile) { "Model not found: $name" }
        selection.writeText(JSONObject().put("id", file.name).put("at", System.currentTimeMillis()).toString())
    }

    fun selectedModel(): ModelEntry? = models().firstOrNull { it.selected }

    fun deleteModel(name: String) {
        require(name.isNotBlank() && !name.contains('/')) { "Invalid model name" }
        File(models, "$name.gguf").delete()
        File(models, "$name.gguf.part").delete()
    }

    fun freeBytes(): Long = runCatching { StatFs(root.absolutePath).availableBytes }.getOrDefault(0L)

    // ------------------------------------------------------------------ helpers

    /** Keeps `logs/` bounded: a diagnostic log must never fill the device. */
    fun rotateLogs(keepBytes: Long = 512 * 1024, keepFiles: Int = 4) {
        runCatching {
            val files = logs.listFiles { file -> file.isFile }?.sortedBy { it.lastModified() } ?: return
            var total = files.sumOf { it.length() }
            var removed = 0
            files.reversed().forEach { file ->
                if (total > keepBytes || removed > keepFiles) {
                    val size = file.length()
                    if (file.delete()) {
                        total -= size
                        removed += 1
                    }
                }
            }
        }
    }

    fun settingsJson(): JSONObject {
        val file = File(settings, "config.json")
        return runCatching { JSONObject(file.readText()) }.getOrElse { JSONObject() }
    }

    fun saveSettings(value: JSONObject) {
        val file = File(settings, "config.json")
        val temp = File(settings, "config.json.tmp")
        temp.writeText(value.toString())
        if (!temp.renameTo(file)) {
            temp.delete()
            throw IOException("Cannot save settings")
        }
    }

    /** Verified import from a content:// URI, streamed with progress. */
    suspend fun importFromUri(uri: android.net.Uri, name: String): ModelEntry? = withContext(Dispatchers.IO) {
        val clean = name.substringAfterLast('/').removeSuffix(".gguf").take(120)
        if (clean.isBlank()) throw IOException("Nome file non valido")
        val target = File(models, "$clean.gguf")
        if (target.isFile && target.length() > 0) return@withContext models().firstOrNull { it.id == clean }
        val partial = File(models, "$clean.gguf.part")
        context.contentResolver.openInputStream(uri)?.use { input ->
            java.io.FileOutputStream(partial).use { output ->
                val buffer = ByteArray(256 * 1024)
                while (true) {
                    val read = input.read(buffer)
                    if (read < 0) break
                    output.write(buffer, 0, read)
                }
            }
        } ?: throw IOException("Impossibile leggere il file selezionato")
        require(partial.length() > 4) { "Il file è vuoto" }
        val head = ByteArray(4)
        partial.inputStream().use { if (it.read(head) != 4) partial.delete() }
        require(head.contentEquals(byteArrayOf(0x47, 0x47, 0x55, 0x46.toByte()))) {
            partial.delete()
            "Quel file non è un modello GGUF"
        }
        if (!partial.renameTo(target)) {
            partial.delete()
            throw IOException("Cannot finish import")
        }
        models().firstOrNull { it.id == clean }
    }

    fun hub(): ModelHub = ModelHub(models) { freeBytes() }

    /**
     * Qwen3.8-4B Distill preset: the reference model for the Hexagon NPU path.
     * Hybrid qwen35 tensors — the in-process Rust engine cannot load it, only
     * the GenieX runner can, so importing it also implies the hexagon-npu backend.
     */
    fun stagingModel(): File? {
        val file = File("$QWEN38_STAGING_DIR/$QWEN38_STAGING_FILE")
        return if (file.isFile && file.canRead() && file.length() > 0) file else null
    }

    /** Copies the staging GGUF from /data/local/tmp/npb into filesDir/models. */
    suspend fun importFromStaging(onProgress: (Long, Long) -> Unit): ModelEntry? = withContext(Dispatchers.IO) {
        val source = stagingModel() ?: throw IOException("File di staging non trovato in $QWEN38_STAGING_DIR")
        val clean = QWEN38_FILE.removeSuffix(".gguf")
        val target = File(models, "$clean.gguf")
        if (target.isFile && target.length() == source.length()) {
            return@withContext models().firstOrNull { it.id == clean }
        }
        if (target.exists()) target.delete()
        require(source.length() + 64L * 1024 * 1024 < freeBytes()) { "Spazio insufficiente per copiare il modello" }
        val partial = File(models, "$clean.gguf.part")
        source.inputStream().use { input ->
            java.io.FileOutputStream(partial).use { output ->
                val buffer = ByteArray(1024 * 1024)
                var done = 0L
                val total = source.length()
                while (true) {
                    val read = input.read(buffer)
                    if (read < 0) break
                    output.write(buffer, 0, read)
                    done += read
                    onProgress(done, total)
                }
            }
        }
        require(partial.length() == source.length()) {
            partial.delete()
            "Copia incompleta: ${partial.length()} di ${source.length()} byte"
        }
        val head = ByteArray(4)
        partial.inputStream().use { it.read(head) }
        require(head.contentEquals(byteArrayOf(0x47, 0x47, 0x55, 0x46.toByte()))) {
            partial.delete()
            "Il file di staging non è un modello GGUF"
        }
        if (!partial.renameTo(target)) {
            partial.delete()
            throw IOException("Cannot finish import")
        }
        models().firstOrNull { it.id == clean }
    }

    companion object {
        const val QWEN38_REPO = "empero-ai/Qwen3.8-4B-Distill-GGUF"
        const val QWEN38_FILE = "Qwen3.8-4B-Q4_K_M.gguf"
        const val QWEN38_BYTES = 2783446304L
        const val QWEN38_STAGING_DIR = "/data/local/tmp/npb"
        const val QWEN38_STAGING_FILE = "qwen38-4b-distill-q4km.gguf"
        // Llama 3.2 3B Instruct: classic GQA attention the in-process Rust engine
        // loads (separate Q/K/V, standard RoPE). Resident model, real streaming,
        // real tok/s — and the strongest tool-use record at 3B.
        const val LLAMA32_REPO = "unsloth/Llama-3.2-3B-Instruct-GGUF"
        const val LLAMA32_FILE = "Llama-3.2-3B-Instruct-Q4_K_M.gguf"
        private val SAFE_ID = Regex("[A-Za-z0-9._-]{1,96}")
    }
}