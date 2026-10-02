package com.pocketworkbench.app

import android.content.Context
import android.os.StatFs
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request
import org.tukaani.xz.XZInputStream
import org.json.JSONObject
import java.io.BufferedInputStream
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.IOException
import java.nio.file.Files
import java.nio.file.InvalidPathException
import java.nio.file.attribute.PosixFilePermission
import java.security.MessageDigest
import java.util.EnumSet
import java.util.concurrent.TimeUnit

/**
 * Optional Debian userland for the agent's shell tool.
 *
 * Android's own shell (toybox) has no package manager, compilers or scripting
 * languages. This module downloads a Termux-built Debian trixie rootfs once,
 * verifies its SHA-256 and extracts it under filesDir/linux. The Rust runtime
 * then executes `system="linux"` commands through the bundled static proot
 * (same UID, no root needed), with the project bound at /workspace.
 *
 * What this is not: a sandbox, a VM, or part of the APK — the ~35 MB archive
 * is fetched on demand and deleted after a successful extract. apt inside the
 * container reaches the network like any app traffic.
 */
class LinuxModule(private val context: Context) {

    val dir: File = File(context.filesDir, "linux").apply { mkdirs() }
    val rootfs: File = File(dir, "debian")
    val tmp: File = File(dir, "tmp").apply { mkdirs() }
    private val archive = File(dir, "debian-trixie.tar.xz")
    private val versionFile = File(dir, "version.json")
    val toolsDir: File = File(context.filesDir, "linux-tools").apply { mkdirs() }

    private val client = OkHttpClient.Builder()
        .connectTimeout(25, TimeUnit.SECONDS)
        .readTimeout(60, TimeUnit.SECONDS)
        .build()

    data class Status(val installed: Boolean, val version: String?, val bytes: Long)

    fun status(): Status {
        val version = runCatching { JSONObject(versionFile.readText()).optString("version") }
            .getOrNull()?.ifBlank { null }
        val ok = version == VERSION &&
            File(rootfs, "bin/bash").isFile &&
            File(rootfs, "etc/debian_version").isFile
        val bytes = if (ok) {
            runCatching { rootfs.walkTopDown().filter { it.isFile }.sumOf { it.length() } }.getOrDefault(0L)
        } else 0L
        return Status(ok, version, bytes)
    }

    /** Config block forwarded to the Rust runtime through the session config. */
    fun configJson(nativeDir: String): JSONObject = JSONObject()
        .put("proot", File(nativeDir, "libproot.so").absolutePath)
        .put("rootfs", rootfs.absolutePath)
        .put("tmp", tmp.absolutePath)
        .put("tools", toolsDir.absolutePath)

    /**
     * Symlink for the bundled static curl, used by Android-shell mode.
     * Independent of the rootfs: works even when Debian is not installed.
     */
    fun ensureTools(nativeDir: String) {
        val target = File(nativeDir, "libpocketcurl.so")
        if (!target.isFile) return
        runCatching {
            val link = File(toolsDir, "curl")
            if (link.exists() || Files.isSymbolicLink(link.toPath())) link.delete()
            Files.createSymbolicLink(link.toPath(), target.toPath())
        }
    }

    suspend fun install(onProgress: (Long, Long) -> Unit) = withContext(Dispatchers.IO) {
        dir.mkdirs()
        tmp.mkdirs()
        downloadResumable(onProgress)
        verifySha256()
        extractRootfs()
        File(rootfs, "etc/resolv.conf").apply {
            parentFile?.mkdirs()
            // Android has no host resolv.conf (DNS goes through netd); the
            // container needs real resolvers or apt goes nowhere.
            writeText("nameserver 1.1.1.1\nnameserver 8.8.8.8\n")
        }
        versionFile.writeText(
            JSONObject().put("version", VERSION).put("at", System.currentTimeMillis()).toString(),
        )
        archive.delete()
        if (!status().installed) throw IOException("Estrazione completata ma Debian non valido")
    }

    fun uninstall() {
        rootfs.deleteRecursively()
        versionFile.delete()
        archive.delete()
        tmp.deleteRecursively()
        tmp.mkdirs()
    }

    // ------------------------------------------------------------- download

    private suspend fun downloadResumable(onProgress: (Long, Long) -> Unit) = withContext(Dispatchers.IO) {
        val partial = File(dir, "debian-trixie.tar.xz.part")
        val existing = if (partial.isFile) partial.length() else 0L
        val builder = Request.Builder().url(URL).header("User-Agent", "PocketWorkbench/1")
        if (existing > 0) builder.header("Range", "bytes=$existing-")
        client.newCall(builder.build()).execute().use { response ->
            if (response.code !in listOf(200, 206)) throw IOException("Download fallito: HTTP ${response.code}")
            val append = existing > 0 && response.code == 206 &&
                response.header("Content-Range").orEmpty().startsWith("bytes $existing-")
            val start = if (append) existing else 0L
            if (!append) partial.delete()
            val declared = response.body?.contentLength() ?: -1L
            val total = if (declared >= 0) start + declared else -1L
            if (total > 0 && total + 64L * 1024 * 1024 > freeBytes()) {
                throw IOException("Spazio insufficiente per Debian (servono ~${total / 1048576} MiB)")
            }
            val body = response.body ?: throw IOException("Risposta vuota")
            var done = start
            body.byteStream().use { input ->
                FileOutputStream(partial, append).buffered().use { output ->
                    val buffer = ByteArray(256 * 1024)
                    while (true) {
                        val read = input.read(buffer)
                        if (read < 0) break
                        output.write(buffer, 0, read)
                        done += read
                        onProgress(done, total)
                    }
                }
            }
            if (total > 0 && done != total) throw IOException("Download incompleto: $done di $total byte")
            if (!partial.renameTo(archive)) throw IOException("Cannot finish download")
        }
    }

    private fun verifySha256() {
        val digest = MessageDigest.getInstance("SHA-256")
        FileInputStream(archive).buffered().use { input ->
            val buffer = ByteArray(256 * 1024)
            while (true) {
                val read = input.read(buffer)
                if (read < 0) break
                digest.update(buffer, 0, read)
            }
        }
        val actual = digest.digest().joinToString("") { "%02x".format(it) }
        if (!actual.equals(SHA256, ignoreCase = true)) {
            archive.delete()
            throw IOException("Checksum Debian non corrispondente: file scartato")
        }
    }

    // ------------------------------------------------------------- extract

    private fun extractRootfs() {
        rootfs.deleteRecursively()
        rootfs.mkdirs()
        XZInputStream(BufferedInputStream(FileInputStream(archive))).use { xz ->
            TarReader(xz).use { tar ->
                while (true) {
                    val entry = tar.next() ?: break
                    // The Termux tarball nests everything one level deep.
                    val stripped = entry.name.trimStart('/').substringAfter('/', "")
                    val out = targetFor(rootfs, stripped) ?: continue
                    when (entry.type) {
                        '5' -> {
                            out.mkdirs()
                            applyMode(out, entry.mode)
                        }
                        '2' -> {
                            out.parentFile?.mkdirs()
                            // A premature mkdirs (from an entry processed before
                            // its parent symlink) would shadow the link: drop an
                            // empty directory we created ourselves, never data.
                            if (out.isDirectory && (out.list()?.isEmpty() == true)) {
                                out.delete()
                            } else {
                                runCatching { Files.deleteIfExists(out.toPath()) }
                            }
                            try {
                                Files.createSymbolicLink(out.toPath(), java.nio.file.Paths.get(entry.linkName))
                            } catch (e: InvalidPathException) {
                                // A few upstream entries (Hungarian CA certs) carry
                                // non-UTF-8 names no Path API can represent: skip
                                // those links instead of failing a 180 MB install.
                                continue
                            }
                        }
                        '1' -> {
                            out.parentFile?.mkdirs()
                            // Absolute links point inside the rootfs (minus the
                            // tarball's top-level dir); relative links resolve
                            // against the entry's own directory.
                            val target = if (entry.linkName.startsWith('/')) {
                                File(rootfs, entry.linkName.trimStart('/').substringAfter('/', entry.linkName))
                            } else {
                                File(out.parentFile, entry.linkName)
                            }
                            runCatching { Files.deleteIfExists(out.toPath()) }
                            try {
                                Files.createLink(out.toPath(), target.toPath())
                            } catch (e: InvalidPathException) {
                                continue
                            }
                        }
                        '0' -> {
                            out.parentFile?.mkdirs()
                            FileOutputStream(out).use { output -> tar.copyTo(output) }
                            applyMode(out, entry.mode)
                        }
                        else -> Unit
                    }
                }
            }
        }
        require(File(rootfs, "bin/bash").isFile) { "Rootfs incompleto: manca /bin/bash" }
    }

    private fun applyMode(file: File, mode: Int) {
        runCatching {
            val perms = EnumSet.noneOf(PosixFilePermission::class.java)
            if (mode and 0x100 != 0) perms.add(PosixFilePermission.OWNER_READ)
            if (mode and 0x80 != 0) perms.add(PosixFilePermission.OWNER_WRITE)
            if (mode and 0x40 != 0) perms.add(PosixFilePermission.OWNER_EXECUTE)
            if (mode and 0x20 != 0) perms.add(PosixFilePermission.GROUP_READ)
            if (mode and 0x10 != 0) perms.add(PosixFilePermission.GROUP_WRITE)
            if (mode and 0x8 != 0) perms.add(PosixFilePermission.GROUP_EXECUTE)
            if (mode and 0x4 != 0) perms.add(PosixFilePermission.OTHERS_READ)
            if (mode and 0x2 != 0) perms.add(PosixFilePermission.OTHERS_WRITE)
            if (mode and 0x1 != 0) perms.add(PosixFilePermission.OTHERS_EXECUTE)
            Files.setPosixFilePermissions(file.toPath(), perms)
        }
    }

    private fun freeBytes(): Long =
        runCatching { StatFs(context.filesDir.absolutePath).availableBytes }.getOrDefault(0L)

    companion object {
        const val VERSION = "debian-trixie-pd-v4.26.0"
        const val URL =
            "https://github.com/termux/proot-distro/releases/download/v4.26.0/debian-trixie-aarch64-pd-v4.26.0.tar.xz"
        const val SHA256 = "cda75346f2c9e09e8a802665745b5a7e2bd6d8584dbf1c86c8c57ef54c4e2d3c"

        /**
         * Where a stripped archive path lands in the Debian tree, or null when the
         * entry must not be materialised (the top dir itself, /dev which PRoot
         * binds from the host). Throws on anything escaping the tree: a hostile
         * tarball must fail the install, never write outside it.
         *
         * Static so the offline unit test can pin the policy without an
         * Android Context.
         */
        @JvmStatic
        fun targetFor(root: File, stripped: String): File? {
            if (stripped.isEmpty() || stripped == "dev" || stripped.startsWith("dev/")) return null
            val rootCanonical = root.canonicalFile
            val out = File(root, stripped)
            require(out.canonicalPath == rootCanonical.path ||
                out.canonicalPath.startsWith(rootCanonical.path + File.separator)) {
                "Percorso fuori dal rootfs: $stripped"
            }
            return out
        }
    }
}
