package com.pocketworkbench.app

import android.content.Context
import android.net.Uri
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.util.concurrent.TimeUnit
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream

/** Local project tools. Files stay under the app's workspace; commands use Android's shell. */
class WorkspaceTools(private val context: Context, workspace: File) {
    private val root = workspace.canonicalFile
    @Volatile private var running: Process? = null

    val signatures = listOf(
        "workspace_list(path?) — list project files (path defaults to .)",
        "workspace_read(path) — read a UTF-8 project file (up to 12 KB)",
        "workspace_write(path,content) — create or replace a UTF-8 project file; creates parent folders",
        "workspace_run(command) — run an Android shell command in the project directory (90 second limit)"
    )
    val names = setOf("workspace_list", "workspace_read", "workspace_write", "workspace_run")

    private fun resolve(path: String): File {
        val target = File(root, path.ifBlank { "." }).canonicalFile
        require(target == root || target.path.startsWith(root.path + File.separator)) { "Path is outside the workspace" }
        return target
    }

    fun list(path: String = "."): List<File> {
        val dir = resolve(path)
        require(dir.isDirectory) { "Folder not found" }
        return dir.listFiles().orEmpty().filter {
            val canonical = it.canonicalFile
            canonical == root || canonical.path.startsWith(root.path + File.separator)
        }.sortedWith(compareBy<File> { !it.isDirectory }.thenBy { it.name.lowercase() })
    }

    fun preview(path: String): String {
        val file = resolve(path)
        require(file.isFile) { "File not found" }
        require(file.length() <= 128_000) { "File is too large to preview" }
        val bytes = file.readBytes()
        require(!bytes.contains(0.toByte())) { "Binary file: export it to inspect elsewhere" }
        return String(bytes, Charsets.UTF_8)
    }

    fun stop() { running?.destroyForcibly() }

    suspend fun execute(name: String, args: JSONObject): String = withContext(Dispatchers.IO) {
        when (name) {
            "workspace_list" -> {
                val path = args.optString("path", ".")
                val items = JSONArray()
                list(path).take(80).forEach { items.put(JSONObject().put("name", it.name)
                    .put("type", if (it.isDirectory) "folder" else "file").put("bytes", if (it.isFile) it.length() else 0)) }
                JSONObject().put("path", path).put("items", items).toString()
            }
            "workspace_read" -> {
                val file = resolve(args.getString("path"))
                require(file.isFile && file.length() <= 128_000) { "File missing or too large" }
                val content = preview(args.getString("path"))
                JSONObject().put("content", content.take(12_000))
                    .put("truncated", content.length > 12_000).toString()
            }
            "workspace_write" -> {
                val file = resolve(args.getString("path"))
                require(file != root) { "Choose a file path" }
                val content = args.getString("content")
                require(content.toByteArray(Charsets.UTF_8).size <= 128_000) { "Write is limited to 128 KB" }
                require(file.parentFile?.mkdirs() == true || file.parentFile?.isDirectory == true) { "Cannot create folder" }
                file.writeText(content, Charsets.UTF_8)
                JSONObject().put("path", root.toPath().relativize(file.toPath()).toString()).put("bytes", file.length()).toString()
            }
            "workspace_run" -> {
                val command = args.getString("command")
                require(command.isNotBlank() && command.length <= 4000) { "Command missing or too long" }
                val output = File.createTempFile("workbench-cmd-", ".log", context.cacheDir)
                var process: Process? = null
                try {
                    process = ProcessBuilder("/system/bin/sh", "-c", command).directory(root)
                        .redirectErrorStream(true).redirectOutput(output).start()
                    running = process
                    val finished = process.waitFor(90, TimeUnit.SECONDS)
                    if (!finished) process.destroyForcibly()
                    val text = output.inputStream().buffered().use { input ->
                        if (output.length() > 5000) input.skip(output.length() - 5000)
                        input.readBytes().toString(Charsets.UTF_8).takeLast(5000)
                    }
                    JSONObject().put("exit_code", if (finished) process.exitValue() else -1)
                        .put("timed_out", !finished).put("output", text).toString()
                } finally {
                    process?.destroyForcibly(); running = null; output.delete()
                }
            }
            else -> error("Unknown workspace tool")
        }
    }

    suspend fun export(path: String, uri: Uri) = withContext(Dispatchers.IO) {
        val source = resolve(path)
        require(source.exists()) { "File not found" }
        val output = requireNotNull(context.contentResolver.openOutputStream(uri)) { "Cannot open destination" }
        output.use { stream ->
            if (source.isFile) source.inputStream().use { it.copyTo(stream) }
            else ZipOutputStream(stream).use { zip ->
                source.walkTopDown().filter { it.isFile }.forEach { file ->
                    val safe = resolve(root.toPath().relativize(file.toPath()).toString())
                    require(safe == file.canonicalFile) { "Workspace contains an external link" }
                    val relative = source.toPath().relativize(file.toPath()).toString().replace(File.separatorChar, '/')
                    zip.putNextEntry(ZipEntry("${source.name}/$relative"))
                    file.inputStream().use { it.copyTo(zip) }
                    zip.closeEntry()
                }
            }
        }
    }
}
