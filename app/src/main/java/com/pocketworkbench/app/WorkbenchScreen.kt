package com.pocketworkbench.app

import android.content.Intent
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.RowScope
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.ChatBubbleOutline
import androidx.compose.material.icons.filled.DeleteOutline
import androidx.compose.material.icons.filled.Download
import androidx.compose.material.icons.filled.FolderOpen
import androidx.compose.material.icons.filled.History
import androidx.compose.material.icons.filled.Memory
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material.icons.filled.Speed
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.FilterChip
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.NavigationRail
import androidx.compose.material3.NavigationRailItem
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.compose.ui.unit.sp
import kotlinx.coroutines.launch
import org.json.JSONObject
import java.io.File

private enum class Page(val label: String) {
    Chat("Chat"), Models("Modelli"), Files("File"), Sessions("Sessioni"), Stats("Statistiche"), Settings("Impostazioni")
}

private val Ink = Color(0xFF070C18)
private val PanelColor = Color(0xFF111B31)
private val Accent = Color(0xFF7CF5D3)
private val MutedColor = Color(0xFF8EA2C8)
private val PaperColor = Color(0xFFDDE7FF)

/**
 * The app shell.
 *
 * Six pages, no WebView and no hidden runtime: the chat is native, and each page
 * shows what the runtime actually reported rather than a promise about it.
 */
@Composable
fun Workbench(reducedMotion: Boolean) {
    val model: AgentViewModel = viewModel()
    val state by model.state.collectAsState()
    val context = LocalContext.current
    var page by remember { mutableStateOf(Page.Chat) }
    var revision by remember { mutableIntStateOf(0) }
    val refresh = { revision += 1 }
    val store = model.storeRef

    // One session per project selection, created on first use.
    LaunchedEffect(revision) {
        if (state.sessionId == null) {
            val project = model.projects.firstOrNull() ?: store.createProject("default")
            model.newSession(project.id)
        }
    }

    Surface(Modifier.fillMaxSize(), color = Ink) {
        androidx.compose.foundation.layout.BoxWithConstraints(Modifier.fillMaxSize()) {
            val wide = maxWidth >= 840.dp
            Column(Modifier.fillMaxSize()) {
                Row(
                    Modifier.fillMaxWidth().padding(horizontal = 18.dp, vertical = 10.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Text("POCKET WORKBENCH", color = Accent, fontWeight = FontWeight.Bold, fontSize = 14.sp)
                    Spacer(Modifier.weight(1f))
                    Text(
                        listOfNotNull(state.modelName, state.backend).joinToString(" · ").ifBlank { "nessun modello caricato" },
                        color = MutedColor,
                        fontSize = 12.sp,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis,
                        modifier = Modifier.width(240.dp),
                    )
                }
                HorizontalDivider(color = Color(0xFF22304A))
                Row(Modifier.weight(1f)) {
                    if (wide) {
                        NavigationRail(containerColor = Ink) {
                            Spacer(Modifier.height(8.dp))
                            Page.entries.forEach { destination ->
                                NavigationRailItem(
                                    selected = page == destination,
                                    onClick = { page = destination; refresh() },
                                    icon = { Icon(iconFor(destination), null) },
                                    label = { Text(destination.label, fontSize = 10.sp) },
                                )
                            }
                        }
                    }
                    Box(Modifier.weight(1f).fillMaxHeight()) {
                        when (page) {
                            Page.Chat -> AgentChatScreen(model, reducedMotion)
                            Page.Models -> ModelsPage(model, revision) { refresh() }
                            Page.Files -> FilesPage(model, revision)
                            Page.Sessions -> SessionsPage(model, revision) { refresh() }
                            Page.Stats -> StatsPage(model, revision)
                            Page.Settings -> SettingsPage(model, revision, refresh)
                        }
                    }
                }
                HorizontalDivider(color = Color(0xFF22304A))
                if (wide) {
                    FooterBar(model, state)
                } else {
                    Text(
                        state.status.ifBlank { "Pronto" },
                        color = MutedColor,
                        fontSize = 11.sp,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis,
                        modifier = Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 4.dp),
                    )
                    NavigationBar(containerColor = Ink) {
                        Page.entries.forEach { destination ->
                            NavigationBarItem(
                                selected = page == destination,
                                onClick = { page = destination; refresh() },
                                icon = { Icon(iconFor(destination), null, Modifier.size(18.dp)) },
                                label = { Text(destination.label, fontSize = 10.sp) },
                            )
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun FooterBar(model: AgentViewModel, state: AgentUiState) {
    Row(Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 6.dp), verticalAlignment = Alignment.CenterVertically) {
        Text(state.status.ifBlank { "Pronto" }, color = MutedColor, fontSize = 11.sp, modifier = Modifier.weight(1f))
        Text(state.lastStop.orEmpty(), color = MutedColor, fontSize = 11.sp)
    }
}

private fun iconFor(page: Page) = when (page) {
    Page.Chat -> Icons.Default.ChatBubbleOutline
    Page.Models -> Icons.Default.Memory
    Page.Files -> Icons.Default.FolderOpen
    Page.Sessions -> Icons.Default.History
    Page.Stats -> Icons.Default.Speed
    Page.Settings -> Icons.Default.Settings
}

@Composable
private fun Section(title: String, subtitle: String = "", content: @Composable androidx.compose.foundation.layout.ColumnScope.() -> Unit) {
    Column(Modifier.fillMaxSize().padding(20.dp)) {
        Text(title, style = MaterialTheme.typography.headlineSmall, color = PaperColor)
        if (subtitle.isNotBlank()) {
            Text(subtitle, color = MutedColor, fontSize = 12.sp, modifier = Modifier.padding(top = 4.dp, bottom = 12.dp))
        }
        content()
    }
}

@Composable
private fun Card(content: @Composable () -> Unit) {
    Surface(color = PanelColor, shape = RoundedCornerShape(14.dp), modifier = Modifier.fillMaxWidth().padding(vertical = 5.dp)) {
        Column(Modifier.padding(14.dp)) { content() }
    }
}

// ------------------------------------------------------------------- models

@Composable
private fun ModelsPage(model: AgentViewModel, revision: Int, onChanged: () -> Unit) {
    val store = model.storeRef
    val state by model.state.collectAsState()
    var busy by remember { mutableStateOf<String?>(null) }
    var failure by remember { mutableStateOf<String?>(null) }
    var results by remember { mutableStateOf<List<ModelHub.Remote>>(emptyList()) }
    var query by remember { mutableStateOf("") }
    var downloads by remember { mutableStateOf<Map<String, Pair<Long, Long>>>(emptyMap()) }
    var deleting by remember { mutableStateOf<WorkbenchStore.ModelEntry?>(null) }
    val scope = androidx.compose.runtime.rememberCoroutineScope()
    val available = store.models().isNotEmpty()

    val importer = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri == null) return@rememberLauncherForActivityResult
        val name = uri.lastPathSegment?.substringAfterLast('/')?.removeSuffix(".gguf").orEmpty()
        busy = "Importo $name"
        failure = null
        scope.launch {
            val outcome = runCatching { store.importFromUri(uri, name) }
            busy = null
            outcome.onFailure { failure = it.message }
            outcome.onSuccess { onChanged() }
        }
    }

    Section(
        "Modelli",
        "I pesi GGUF stanno in filesDir/models e non finiscono mai in un progetto. Il runtime li tiene in memoria finché non li rilasci.",
    ) {
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(onClick = { importer.launch(arrayOf("*/*")) }, enabled = busy == null) {
                Icon(Icons.Default.FolderOpen, null, Modifier.size(18.dp)); Spacer(Modifier.width(6.dp)); Text("Importa GGUF")
            }
            Button(onClick = model::loadSelectedModel, enabled = available && !state.busy && busy == null) {
                Icon(Icons.Default.Download, null, Modifier.size(18.dp)); Spacer(Modifier.width(6.dp)); Text("Carica")
            }
            OutlinedButton(onClick = model::unloadModel, enabled = state.modelName != null) { Text("Rilascia") }
        }
        // Preset Qwen3.8-4B Distill: reference weights for the Hexagon NPU path.
        // They only run through the GenieX runner — the Rust engine cannot load
        // hybrid qwen35 tensors — so a successful import also selects the model
        // and switches the backend to hexagon-npu.
        val staging = remember(revision) { store.stagingModel() }
        Card {
            Text("Qwen3.8-4B Distill — NPU (sperimentale)", color = PaperColor, fontWeight = FontWeight.Bold, fontSize = 14.sp)
            Text(
                "2,59 GiB · ${WorkbenchStore.QWEN38_REPO} · Solo backend Hexagon NPU, non testato end-to-end.",
                color = MutedColor,
                fontSize = 11.sp,
                modifier = Modifier.padding(top = 2.dp, bottom = 8.dp),
            )
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Button(
                    enabled = busy == null,
                    onClick = {
                        busy = "Scarico ${WorkbenchStore.QWEN38_FILE}"
                        failure = null
                        scope.launch {
                            val outcome = runCatching {
                                val remote = store.hub().files(WorkbenchStore.QWEN38_REPO)
                                    .firstOrNull { it.filename == WorkbenchStore.QWEN38_FILE }
                                    ?: throw java.io.IOException("File ${WorkbenchStore.QWEN38_FILE} non trovato nel repository")
                                store.hub().download(remote) { done, total ->
                                    downloads = downloads + (remote.filename to (done to total))
                                }
                            }
                            busy = null
                            downloads = downloads - WorkbenchStore.QWEN38_FILE
                            outcome.onFailure { failure = it.message }
                            outcome.onSuccess {
                                runCatching { model.selectModel(WorkbenchStore.QWEN38_FILE.removeSuffix(".gguf")) }
                                model.updateSetting("backend", "hexagon-npu")
                                onChanged()
                            }
                        }
                    },
                ) { Text("Scarica") }
                if (staging != null) {
                    OutlinedButton(
                        enabled = busy == null,
                        onClick = {
                            busy = "Copio da staging (${formatBytes(staging.length())})"
                            failure = null
                            scope.launch {
                                val outcome = runCatching {
                                    store.importFromStaging { done, total ->
                                        downloads = downloads + (WorkbenchStore.QWEN38_FILE to (done to total))
                                    }
                                }
                                busy = null
                                downloads = downloads - WorkbenchStore.QWEN38_FILE
                                outcome.onFailure { failure = it.message }
                                outcome.onSuccess {
                                    runCatching { model.selectModel(WorkbenchStore.QWEN38_FILE.removeSuffix(".gguf")) }
                                    model.updateSetting("backend", "hexagon-npu")
                                    onChanged()
                                }
                            }
                        },
                    ) { Text("Copia da staging") }
                } else {
                    Text("Nessun file in ${WorkbenchStore.QWEN38_STAGING_DIR}", color = MutedColor, fontSize = 11.sp, modifier = Modifier.padding(top = 12.dp))
                }
            }
        }
        // Llama 3.2 3B: runs on the resident Rust engine (CPU), so answers
        // stream token by token with real tok/s and no per-turn model reload.
        Card {
            Text("Llama 3.2 3B Instruct — CPU (consigliato)", color = PaperColor, fontWeight = FontWeight.Bold, fontSize = 14.sp)
            Text(
                "≈1,9 GiB · ${WorkbenchStore.LLAMA32_REPO} · Motore interno sempre caldo, streaming vero, miglior tool-use a 3B.",
                color = MutedColor,
                fontSize = 11.sp,
                modifier = Modifier.padding(top = 2.dp, bottom = 8.dp),
            )
            Button(
                enabled = busy == null,
                onClick = {
                    busy = "Scarico ${WorkbenchStore.LLAMA32_FILE}"
                    failure = null
                    scope.launch {
                        val outcome = runCatching {
                            val remote = store.hub().files(WorkbenchStore.LLAMA32_REPO)
                                .firstOrNull { it.filename == WorkbenchStore.LLAMA32_FILE }
                                ?: throw java.io.IOException("File ${WorkbenchStore.LLAMA32_FILE} non trovato nel repository")
                            store.hub().download(remote) { done, total ->
                                downloads = downloads + (remote.filename to (done to total))
                            }
                        }
                        busy = null
                        downloads = downloads - WorkbenchStore.LLAMA32_FILE
                        outcome.onFailure { failure = it.message }
                        outcome.onSuccess {
                            runCatching { model.selectModel(WorkbenchStore.LLAMA32_FILE.removeSuffix(".gguf")) }
                            model.updateSetting("backend", "rust-cpu")
                            onChanged()
                        }
                    }
                },
            ) { Text("Scarica") }
        }
        Row(Modifier.fillMaxWidth().padding(top = 10.dp), verticalAlignment = Alignment.CenterVertically) {
            OutlinedTextField(
                value = query,
                onValueChange = { query = it },
                label = { Text("Cerca su Hugging Face") },
                singleLine = true,
                modifier = Modifier.weight(1f),
            )
            Spacer(Modifier.width(8.dp))
            OutlinedButton(
                enabled = query.isNotBlank() && busy == null,
                onClick = {
                    busy = "Cerco $query"
                    failure = null
                    scope.launch {
                        val outcome = runCatching { store.hub().search(query) }
                        busy = null
                        outcome.onSuccess { results = it }
                        outcome.onFailure { failure = it.message }
                    }
                },
            ) { Text("Cerca") }
        }
        busy?.let { Text(it, color = MutedColor, fontSize = 12.sp, modifier = Modifier.padding(top = 8.dp)) }
        failure?.let { Text(it, color = Color(0xFFFFB4A0), fontSize = 12.sp, modifier = Modifier.padding(top = 6.dp)) }
        downloads.forEach { (name, progress) ->
            val (done, total) = progress
            Column(Modifier.fillMaxWidth().padding(top = 6.dp)) {
                Text("$name — ${formatBytes(done)}${if (total > 0) " / ${formatBytes(total)}" else ""}", color = MutedColor, fontSize = 11.sp)
                Spacer(Modifier.height(4.dp))
                if (total > 0) {
                    LinearProgressIndicator(
                        progress = { (done.toFloat() / total).coerceIn(0f, 1f) },
                        modifier = Modifier.fillMaxWidth(),
                    )
                } else {
                    LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
                }
            }
        }
        LazyColumn(
            modifier = Modifier.fillMaxWidth().weight(1f),
            contentPadding = PaddingValues(top = 12.dp),
            verticalArrangement = Arrangement.spacedBy(6.dp),
        ) {
            items(store.models(), key = { it.id }) { entry ->
                Card {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Column(Modifier.weight(1f)) {
                            Text(entry.name, maxLines = 1, overflow = TextOverflow.Ellipsis, color = PaperColor)
                            Text(
                                formatBytes(entry.bytes) + if (entry.selected) " · selezionato" else "",
                                color = MutedColor,
                                fontSize = 11.sp,
                            )
                        }
                        FilterChip(selected = entry.selected, onClick = { model.selectModel(entry.id) }, label = { Text("Usa") })
                        IconButton(onClick = { deleting = entry }) {
                            Icon(Icons.Default.DeleteOutline, "Elimina ${entry.name}", tint = MutedColor)
                        }
                    }
                }
            }
            if (results.isNotEmpty()) {
                item { Text("RISULTATI HUGGING FACE", color = Accent, fontSize = 11.sp, modifier = Modifier.padding(top = 12.dp)) }
                items(results, key = { "${it.repo}/${it.filename}" }) { remote ->
                    Card {
                        Text(remote.filename, maxLines = 1, overflow = TextOverflow.Ellipsis, color = PaperColor, fontSize = 13.sp)
                        Text(
                            "${remote.repo} · ${if (remote.bytes > 0) formatBytes(remote.bytes) else "dimensione sconosciuta"}" +
                                if (remote.gated) " · ad accesso riservato" else "",
                            color = MutedColor,
                            fontSize = 11.sp,
                        )
                        Button(
                            enabled = !remote.gated && busy == null,
                            onClick = {
                                val label = remote.filename
                                busy = "Scarico $label"
                                failure = null
                                scope.launch {
                                    val outcome = runCatching {
                                        store.hub().download(remote) { done, total ->
                                            downloads = downloads + (label to (done to total))
                                        }
                                    }
                                    busy = null
                                    downloads = downloads - label
                                    outcome.onFailure { failure = it.message }
                                    outcome.onSuccess { onChanged() }
                                }
                            },
                        ) { Text("Scarica") }
                    }
                }
            }
        }
    }
    deleting?.let { target ->
        AlertDialog(
            onDismissRequest = { deleting = null },
            title = { Text("Eliminare il modello?") },
            text = { Text("${target.name} sarà rimosso da questo dispositivo. I file dei progetti non lo contengono, quindi nessun progetto cambia.") },
            confirmButton = {
                TextButton(onClick = {
                    runCatching { store.deleteModel(target.id) }
                    deleting = null
                    onChanged()
                }) { Text("Elimina") }
            },
            dismissButton = { TextButton(onClick = { deleting = null }) { Text("Annulla") } },
        )
    }
}

// -------------------------------------------------------------------- files

@Composable
private fun FilesPage(model: AgentViewModel, revision: Int) {
    val store = model.storeRef
    val state by model.state.collectAsState()
    var folder by remember(state.projectId, revision) { mutableStateOf("") }
    var open by remember { mutableStateOf<String?>(null) }
    val root = File(store.workspaces, state.projectId.ifBlank { "default" })
    val listing = remember(folder, revision) { listFolder(root, folder) }

    Column(Modifier.fillMaxSize().padding(20.dp)) {
        Text("File del progetto", style = MaterialTheme.typography.headlineSmall, color = PaperColor)
        Text(
            "workspaces/${state.projectId.ifBlank { "default" }}${if (folder.isBlank()) "" else "/$folder"}",
            color = MutedColor,
            fontSize = 12.sp,
            modifier = Modifier.padding(top = 2.dp, bottom = 8.dp),
        )
        Row(verticalAlignment = Alignment.CenterVertically) {
            if (folder.isNotEmpty()) {
                TextButton(onClick = { folder = folder.substringBeforeLast('/', ""); open = null }) { Text("← Indietro") }
            }
            OutlinedButton(onClick = { folder = ""; open = null }) { Text("Radice") }
        }
        LazyColumn(Modifier.weight(1f).fillMaxWidth(), verticalArrangement = Arrangement.spacedBy(4.dp)) {
            if (listing.isEmpty()) {
                item { Text("Nessun file. Chiedi all'agente di crearne uno.", color = MutedColor, fontSize = 13.sp, modifier = Modifier.padding(12.dp)) }
            }
            items(listing, key = { it.absolutePath }) { file ->
                val relative = if (folder.isBlank()) file.name else "$folder/${file.name}"
                Surface(
                    color = PanelColor,
                    shape = RoundedCornerShape(10.dp),
                    modifier = Modifier.fillMaxWidth().clickable {
                        if (file.isDirectory) {
                            folder = relative
                            open = null
                        } else {
                            open = relative
                        }
                    },
                ) {
                    Row(Modifier.padding(12.dp), verticalAlignment = Alignment.CenterVertically) {
                        Text(if (file.isDirectory) "▸" else "·", color = Accent)
                        Spacer(Modifier.width(8.dp))
                        Text(file.name, maxLines = 1, overflow = TextOverflow.Ellipsis, color = PaperColor, modifier = Modifier.weight(1f))
                        Text(formatBytes(file.length()), color = MutedColor, fontSize = 11.sp)
                    }
                }
            }
        }
        open?.let { path ->
            val body = remember(path, revision) {
                runCatching {
                    val file = File(root, path)
                    if (file.length() > 256_000) "File troppo grande per l'anteprima." else file.readText()
                }.getOrElse { "Impossibile aprire il file." }
            }
            AlertDialog(
                onDismissRequest = { open = null },
                title = { Text(path.substringAfterLast('/')) },
                text = {
                    SelectionContainer {
                        Text(
                            body,
                            modifier = Modifier.height(320.dp).verticalScroll(rememberScrollState()),
                            fontSize = 12.sp,
                        )
                    }
                },
                confirmButton = { TextButton(onClick = { open = null }) { Text("Chiudi") } },
            )
        }
    }
}

private fun listFolder(root: File, relative: String): List<File> {
    val dir = if (relative.isBlank()) root else File(root, relative)
    if (!dir.isDirectory) return emptyList()
    return dir.listFiles()?.sortedWith(compareByDescending<File> { it.isDirectory }.thenBy { it.name.lowercase() }) ?: emptyList()
}

private fun formatBytes(bytes: Long): String = when {
    bytes >= 1_073_741_824 -> "%.2f GiB".format(bytes / 1073741824.0)
    bytes >= 1_048_576 -> "%.1f MiB".format(bytes / 1048576.0)
    bytes >= 1024 -> "${bytes / 1024} KiB"
    else -> "$bytes B"
}

// ----------------------------------------------------------------- sessions

@Composable
private fun SessionsPage(model: AgentViewModel, revision: Int, onChanged: () -> Unit) {
    val store = model.storeRef
    val state by model.state.collectAsState()
    var deleting by remember { mutableStateOf<WorkbenchStore.Session?>(null) }

    Section("Sessioni", "Ogni sessione è un registro JSONL append-only. Riaprirla ricostruisce il contesto e segnala un turno interrotto invece di riprenderlo da capo.") {
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Button(onClick = { model.newSession(); onChanged() }) {
                Icon(Icons.Default.Add, null, Modifier.size(18.dp)); Spacer(Modifier.width(6.dp)); Text("Nuova sessione")
            }
        }
        Row(
            Modifier.fillMaxWidth().padding(vertical = 10.dp).horizontalScrollCompat(),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            model.projects.forEach { project ->
                FilterChip(
                    selected = state.projectId == project.id,
                    onClick = { model.openProject(project.id); onChanged() },
                    label = { Text(project.name) },
                )
            }
        }
        LazyColumn(Modifier.weight(1f).fillMaxWidth(), verticalArrangement = Arrangement.spacedBy(6.dp)) {
            items(model.sessions, key = { it.id }) { session ->
                Card {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Column(Modifier.weight(1f)) {
                            Text(session.title, color = PaperColor, maxLines = 1, overflow = TextOverflow.Ellipsis)
                            Text(
                                "${session.projectId} · ${formatBytes(session.log.length())} · ${formatStamp(session.updatedAt)}",
                                color = MutedColor,
                                fontSize = 11.sp,
                            )
                        }
                        TextButton(onClick = { model.selectSession(session); onChanged() }, enabled = !state.busy) { Text("Apri") }
                        IconButton(onClick = { deleting = session }) {
                            Icon(Icons.Default.DeleteOutline, "Elimina ${session.title}", tint = MutedColor)
                        }
                    }
                }
            }
        }
    }
    deleting?.let { target ->
        AlertDialog(
            onDismissRequest = { deleting = null },
            title = { Text("Eliminare la sessione?") },
            text = { Text("Il registro ${target.log.name} sarà rimosso. I file del progetto non vengono toccati.") },
            confirmButton = {
                TextButton(onClick = {
                    runCatching { store.deleteSession(target.id) }
                    deleting = null
                    onChanged()
                }) { Text("Elimina") }
            },
            dismissButton = { TextButton(onClick = { deleting = null }) { Text("Annulla") } },
        )
    }
}

private fun formatStamp(at: Long): String =
    if (at <= 0) "" else android.text.format.DateFormat.format("dd/MM HH:mm", at).toString()

@Composable
private fun Modifier.horizontalScrollCompat(): Modifier = this.horizontalScroll(rememberScrollState())

// -------------------------------------------------------------------- stats

@Composable
private fun StatsPage(model: AgentViewModel, revision: Int) {
    val state by model.state.collectAsState()
    Section(
        "Statistiche",
        "I valori sono quelli misurati dal runtime su questo dispositivo, non stime.",
    ) {
        Column(Modifier.verticalScroll(rememberScrollState())) {
            Card {
                Text("Backend", color = Accent, fontSize = 12.sp)
                Text(state.backend ?: "nessuno", color = PaperColor, fontSize = 20.sp, fontWeight = FontWeight.Bold)
                if (state.backendFallback) {
                    Text("Attenzione: è stato usato un fallback rispetto al backend richiesto.", color = Color(0xFFFFB4A0), fontSize = 12.sp)
                }
            }
            Card {
                Text("Modello", color = Accent, fontSize = 12.sp)
                Text(state.modelName ?: "non caricato", color = PaperColor, fontSize = 16.sp)
                Row {
                    Spacer(Modifier.weight(1f))
                    OutlinedButton(onClick = model::loadSelectedModel, enabled = !state.busy) { Text("Carica") }
                    Spacer(Modifier.width(8.dp))
                    OutlinedButton(onClick = model::unloadModel, enabled = state.modelName != null) { Text("Rilascia") }
                }
            }
            Card {
                Text("Ultimo turno", color = Accent, fontSize = 12.sp)
                Text(state.lastUsage.ifBlank { "nessuna misura" }, color = PaperColor, fontSize = 14.sp)
                Text("Fine: ${state.lastStop ?: "-"}", color = MutedColor, fontSize = 12.sp)
            }
            Card {
                Text("Contesto", color = Accent, fontSize = 12.sp)
                Text(
                    state.contextInfo.ifBlank { "nessuna misura: completa un turno" },
                    color = PaperColor,
                    fontSize = 13.sp,
                )
                Text(
                    "Token per categoria e hash del prefix: se l'hash non cambia, il prefill riusa la cache.",
                    color = MutedColor,
                    fontSize = 11.sp,
                )
            }
            Card {
                Text("Limiti", color = Accent, fontSize = 12.sp)
                val settings = model.storeRef.settingsJson()
                listOf(
                    "passi per turno" to settings.optInt("max_steps", 12),
                    "contesto" to settings.optInt("context_tokens", 8192),
                    "token massimi per risposta" to settings.optInt("max_tokens", 512),
                    "coda messaggi" to settings.optInt("max_queued", 8),
                    "thread" to settings.optInt("threads", 4),
                ).forEach { (label, value) ->
                    Text("$label: $value", color = PaperColor, fontSize = 13.sp)
                }
            }
        }
    }
}

// ----------------------------------------------------------------- settings

@Composable
private fun SettingsPage(model: AgentViewModel, revision: Int, onChanged: () -> Unit) {
    val store = model.storeRef
    val settings = remember(revision) { store.settingsJson() }
    fun update(key: String, value: Any) {
        model.updateSetting(key, value)
        onChanged()
    }
    Section("Impostazioni", "Le impostazioni sono scritte in filesDir/settings e valgono per tutte le sessioni.") {
        Column(Modifier.verticalScroll(rememberScrollState())) {
            Card {
                Text("Backend", color = Accent, fontSize = 12.sp)
                Row(Modifier.fillMaxWidth().horizontalScrollCompat(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    listOf("rust-cpu" to "CPU", "rust-opencl" to "OpenCL", "hexagon-npu" to "Hexagon NPU").forEach { (id, label) ->
                        FilterChip(
                            selected = settings.optString("backend", "rust-cpu") == id,
                            onClick = { update("backend", id) },
                            label = { Text(label) },
                        )
                    }
                }
                Text(
                    "hexagon-npu usa il runner precompilato GenieX: avvia un processo per richiesta e stampa i token solo alla fine. È sperimentale.",
                    color = MutedColor,
                    fontSize = 11.sp,
                    modifier = Modifier.padding(top = 6.dp),
                )
            }
            Card {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Column(Modifier.weight(1f)) {
                        Text("OpenCL", color = PaperColor)
                        Text("Usa la GPU se il backend lo supporta.", color = MutedColor, fontSize = 11.sp)
                    }
                    Switch(
                        checked = settings.optBoolean("use_gpu", false),
                        onCheckedChange = { update("use_gpu", it) },
                    )
                }
            }
            Card {
                Text("Contesto: ${settings.optInt("context_tokens", 8192)} token", color = PaperColor)
                Row(Modifier.fillMaxWidth().horizontalScrollCompat(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    listOf(4096, 8192, 16384, 32768).forEach { size ->
                        FilterChip(
                            selected = settings.optInt("context_tokens", 8192) == size,
                            onClick = { update("context_tokens", size) },
                            label = { Text("${size / 1024}K") },
                        )
                    }
                }
                Spacer(Modifier.height(8.dp))
                Text("Thread: ${settings.optInt("threads", 4)}", color = PaperColor)
                Row(Modifier.fillMaxWidth().horizontalScrollCompat(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    listOf(1, 2, 4, 6, 8).forEach { count ->
                        FilterChip(
                            selected = settings.optInt("threads", 4) == count,
                            onClick = { update("threads", count) },
                            label = { Text("$count") },
                        )
                    }
                }
            }
            Card {
                Text("Limiti del turno", color = Accent, fontSize = 12.sp)
                NumberSetting({ key, value -> update(key, value) }, "max_steps", "passi per turno", settings, 1, 32)
                NumberSetting({ key, value -> update(key, value) }, "max_tokens", "token massimi per risposta", settings, 64, 2048, 64)
                NumberSetting({ key, value -> update(key, value) }, "max_queued", "messaggi in coda", settings, 0, 16)
                NumberSetting({ key, value -> update(key, value) }, "max_tool_errors", "errori di strumento consecutivi", settings, 1, 10)
                Text(
                    "Ogni limite è una causa di arresto visibile nel transcript, mai una risposta finale inventata.",
                    color = MutedColor,
                    fontSize = 11.sp,
                )
            }
            Card {
                Text("Compaction", color = Accent, fontSize = 12.sp)
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Column(Modifier.weight(1f)) {
                        Text("Riepilogo modello", color = PaperColor)
                        Text(
                            "La cronologia compattata è riassunta dal modello invece che dalla regola fattuale. Spento di default: i modelli piccoli inventano.",
                            color = MutedColor,
                            fontSize = 11.sp,
                        )
                    }
                    Switch(
                        checked = settings.optBoolean("use_model_summary", false),
                        onCheckedChange = { update("use_model_summary", it) },
                    )
                }
                Text(
                    "Vale dalla prossima sessione. Prima i vecchi output vengono comunque troncati (microcompaction).",
                    color = MutedColor,
                    fontSize = 11.sp,
                )
            }
            Card {
                Text("Capacità", color = Accent, fontSize = 12.sp)
                Text(
                    "Gli strumenti risolvono i percorsi e si rifiutano di uscire dal workspace. Questo non è una sandbox del kernel: la shell è quella di Android e raggiunge ciò che l'UID può raggiungere.",
                    color = MutedColor,
                    fontSize = 12.sp,
                )
            }
            LinuxCard(model, revision, onChanged)
            Card {
                Text("Diagnostica", color = Accent, fontSize = 12.sp)
                Row {
                    OutlinedButton(onClick = { store.rotateLogs() }) { Text("Ruota i log") }
                    Spacer(Modifier.width(8.dp))
                    Text("${formatBytes(store.freeBytes())} liberi", color = MutedColor, fontSize = 12.sp, modifier = Modifier.padding(top = 14.dp))
                }
            }
        }
    }
}

@Composable
private fun LinuxCard(model: AgentViewModel, revision: Int, onChanged: () -> Unit) {
    val context = LocalContext.current
    val module = remember(revision) { LinuxModule(context) }
    val status = remember(revision) { module.status() }
    var busy by remember { mutableStateOf<String?>(null) }
    var failure by remember { mutableStateOf<String?>(null) }
    var progress by remember { mutableStateOf("") }
    val scope = rememberCoroutineScope()
    Card {
        Text("Linux (Debian) per la shell", color = Accent, fontSize = 12.sp)
        Text(
            "Debian Trixie sotto PRoot per system=\"linux\": apt, gcc, python3, git, curl — il progetto sta in /workspace. " +
                "Scaricato una volta sola (~35 MB, ~180 MB su disco), non dentro l'APK.",
            color = MutedColor,
            fontSize = 12.sp,
            modifier = Modifier.padding(top = 4.dp),
        )
        if (status.installed) {
            Text(
                "Installato: ${status.version} · ${formatBytes(status.bytes)} su disco",
                color = PaperColor,
                fontSize = 13.sp,
                modifier = Modifier.padding(top = 8.dp),
            )
            Row(Modifier.padding(top = 8.dp)) {
                OutlinedButton(
                    enabled = busy == null,
                    onClick = {
                        busy = "Rimozione"
                        failure = null
                        scope.launch {
                            runCatching {
                                kotlinx.coroutines.withContext(kotlinx.coroutines.Dispatchers.IO) { module.uninstall() }
                            }.onFailure { failure = it.message }
                            busy = null
                            onChanged()
                        }
                    },
                ) { Text("Rimuovi") }
            }
        } else {
            if (busy == null) {
                Text("Non installato.", color = MutedColor, fontSize = 13.sp, modifier = Modifier.padding(top = 8.dp))
            } else {
                Text(busy.orEmpty(), color = PaperColor, fontSize = 13.sp, modifier = Modifier.padding(top = 8.dp))
            }
            if (progress.isNotBlank()) {
                Text(progress, color = MutedColor, fontSize = 11.sp)
            }
            Row(Modifier.padding(top = 8.dp)) {
                Button(
                    enabled = busy == null,
                    onClick = {
                        busy = "Download Debian…"
                        failure = null
                        progress = ""
                        scope.launch {
                            runCatching {
                                module.install { done, total ->
                                    progress = if (total > 0) {
                                        "${formatBytes(done)} / ${formatBytes(total)}"
                                    } else {
                                        formatBytes(done)
                                    }
                                }
                            }.onFailure { failure = it.message }
                            busy = null
                            progress = ""
                            onChanged()
                        }
                    },
                ) { Text("Installa") }
            }
        }
        failure?.let { Text(it, color = Color(0xFFFFB4A0), fontSize = 12.sp, modifier = Modifier.padding(top = 6.dp)) }
    }
}

@Composable
private fun NumberSetting(
    onChange: (String, Any) -> Unit,
    key: String,
    label: String,
    settings: JSONObject,
    min: Int,
    max: Int,
    step: Int = 1,
) {
    val current = settings.optInt(key, min)
    Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.padding(top = 6.dp)) {
        Text(label, color = PaperColor, modifier = Modifier.weight(1f), fontSize = 13.sp)
        TextButton(onClick = { onChange(key, (current - step).coerceAtLeast(min)) }) { Text("−") }
        Text(current.toString(), color = Accent)
        TextButton(onClick = { onChange(key, (current + step).coerceAtMost(max)) }) { Text("+") }
    }
}