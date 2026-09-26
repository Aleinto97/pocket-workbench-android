package com.pocketworkbench.app

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Bundle
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.gestures.scrollBy
import androidx.compose.foundation.interaction.collectIsDraggedAsState
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.verticalScroll
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.Send
import androidx.compose.material.icons.filled.AccountCircle
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.ChatBubbleOutline
import androidx.compose.material.icons.filled.Code
import androidx.compose.material.icons.filled.ContentCopy
import androidx.compose.material.icons.filled.DeleteOutline
import androidx.compose.material.icons.filled.DeleteSweep
import androidx.compose.material.icons.filled.Download
import androidx.compose.material.icons.filled.FolderOpen
import androidx.compose.material.icons.filled.Logout
import androidx.compose.material.icons.filled.Mic
import androidx.compose.material.icons.filled.OpenInNew
import androidx.compose.material.icons.filled.Search
import androidx.compose.material.icons.filled.Share
import androidx.compose.material.icons.filled.SmartToy
import androidx.compose.material.icons.filled.Speed
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material.icons.filled.Terminal
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.viewmodel.compose.viewModel
import kotlinx.coroutines.launch

private val Indigo = Color(0xFF94B5FF)
private val Dark = Color(0xFF101624)
private val Panel = Color(0xFF1B2535)
private val Pale = Color(0xFFDDE7FF)
private val Console = Color(0xFF0A0F17)
private val DividerColor = Color(0xFF344356)

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            val vm: WorkbenchViewModel = viewModel()
            val microphone = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { if (it) vm.startRecording() }
            val pickFile = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { if (it != null) vm.importModel(it) }
            MaterialTheme(colorScheme = darkColorScheme(primary = Indigo, background = Dark, surface = Panel, onSurface = Pale)) {
                Workbench(vm, onMic = {
                    if (vm.listening || checkSelfPermission(Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) vm.startRecording()
                    else microphone.launch(Manifest.permission.RECORD_AUDIO)
                }, onImport = { pickFile.launch(arrayOf("application/octet-stream", "*/*")) })
            }
        }
    }
}

private enum class Page { Chat, Models, Stats, Workspace, GitHub }

@Composable private fun Workbench(vm: WorkbenchViewModel, onMic: () -> Unit, onImport: () -> Unit) {
    var page by remember { mutableStateOf(Page.Chat) }
    Surface(modifier = Modifier.fillMaxSize(), color = Dark) {
      BoxWithConstraints(Modifier.fillMaxSize()) {
        val wide = maxWidth >= 840.dp
        Column(Modifier.fillMaxSize()) {
            Row(Modifier.fillMaxWidth().padding(horizontal = 20.dp, vertical = 12.dp), verticalAlignment = Alignment.CenterVertically) {
                Text("POCKET WORKBENCH", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold, color = Indigo)
                Spacer(Modifier.weight(1f))
                Text(if (vm.busy) "● Running on device" else "● On-device model", style = MaterialTheme.typography.labelMedium)
            }
            HorizontalDivider(color = DividerColor)
            Row(Modifier.weight(1f)) {
                if (wide) NavigationRail(containerColor = Dark) {
                    Spacer(Modifier.height(12.dp))
                    NavigationRailItem(selected = page == Page.Chat, onClick = { page = Page.Chat }, icon = { Icon(Icons.Default.ChatBubbleOutline, "Chat") }, label = { Text("Chat") })
                    NavigationRailItem(selected = page == Page.Models, onClick = { page = Page.Models }, icon = { Icon(Icons.Default.Download, "Models") }, label = { Text("Models") })
                    NavigationRailItem(selected = page == Page.Stats, onClick = { page = Page.Stats }, icon = { Icon(Icons.Default.Speed, "Stats") }, label = { Text("Stats") })
                    NavigationRailItem(selected = page == Page.Workspace, onClick = { page = Page.Workspace }, icon = { Icon(Icons.Default.FolderOpen, "Files") }, label = { Text("Files") })
                    NavigationRailItem(selected = page == Page.GitHub, onClick = { page = Page.GitHub }, icon = { Icon(Icons.Default.Code, "GitHub") }, label = { Text("GitHub") })
                }
                if (page == Page.Chat && wide) ConversationSidebar(vm, Modifier.width(250.dp).fillMaxHeight())
                Box(Modifier.weight(1f).fillMaxHeight()) {
                    when (page) {
                        Page.Chat -> ChatPage(vm, onMic, !wide)
                        Page.Models -> ModelsPage(vm, onImport)
                        Page.Stats -> StatsScreen(vm)
                        Page.Workspace -> WorkspacePage(vm)
                        Page.GitHub -> GitHubPage(vm)
                    }
                }
            }
            HorizontalDivider(color = DividerColor)
            Text(vm.status, modifier = Modifier.fillMaxWidth().padding(horizontal = 24.dp, vertical = 8.dp), style = MaterialTheme.typography.bodySmall, maxLines = 2)
            if (!wide) NavigationBar(containerColor = Dark) {
                listOf(Page.Chat, Page.Models, Page.Stats, Page.Workspace, Page.GitHub).forEach { destination ->
                    val icon = when (destination) {
                        Page.Chat -> Icons.Default.ChatBubbleOutline
                        Page.Models -> Icons.Default.Download
                        Page.Stats -> Icons.Default.Speed
                        Page.Workspace -> Icons.Default.FolderOpen
                        Page.GitHub -> Icons.Default.Code
                    }
                    NavigationBarItem(selected = page == destination, onClick = { page = destination },
                        icon = { Icon(icon, null) }, label = { Text(if (destination == Page.Workspace) "Files" else destination.name) })
                }
            }
        }
      }
    }
}
@Composable private fun ConversationSidebar(vm: WorkbenchViewModel, modifier: Modifier = Modifier) {
    var deleting by remember { mutableStateOf<Conversation?>(null) }
    deleting?.let { target -> AlertDialog(onDismissRequest = { deleting = null },
        title = { Text("Delete conversation?") }, text = { Text("${target.title} will be removed from this device.") },
        confirmButton = { TextButton(onClick = { vm.deleteChat(target.id); deleting = null }) { Text("Delete") } },
        dismissButton = { TextButton(onClick = { deleting = null }) { Text("Cancel") } }) }
    Column(modifier.padding(12.dp)) {
        Button(onClick = vm::newChat, modifier = Modifier.fillMaxWidth(), enabled = !vm.busy) { Icon(Icons.Default.Add, null); Spacer(Modifier.width(8.dp)); Text("New chat") }
        Spacer(Modifier.height(12.dp))
        Text("CONVERSATIONS", style = MaterialTheme.typography.labelSmall, color = Indigo)
        LazyColumn {
            items(vm.conversations, key = { it.id }) { conversation ->
                Row(verticalAlignment = Alignment.CenterVertically) {
                    TextButton(onClick = { vm.selectChat(conversation.id) }, modifier = Modifier.weight(1f), enabled = !vm.busy) {
                        Text(conversation.title, maxLines = 1, overflow = TextOverflow.Ellipsis,
                            color = if (vm.activeId == conversation.id) Indigo else Pale)
                    }
                    IconButton(onClick = { deleting = conversation }, enabled = !vm.busy, modifier = Modifier.size(48.dp)) {
                        Icon(Icons.Default.DeleteOutline, "Delete ${conversation.title}", modifier = Modifier.size(18.dp))
                    }
                }
            }
        }
    }
}
@Composable private fun ChatPage(vm: WorkbenchViewModel, onMic: () -> Unit, compact: Boolean) {
    var draft by remember { mutableStateOf("") }
    var showChats by remember { mutableStateOf(false) }
    val clipboard = LocalClipboardManager.current
    val context = LocalContext.current
    LaunchedEffect(vm.transcript) { if (vm.transcript.isNotBlank()) { draft = vm.transcript; vm.clearTranscript() } }
    val chat = vm.active
    val scroll = rememberLazyListState()
    var followLatest by remember(chat?.id) { mutableStateOf(true) }
    val dragged by scroll.interactionSource.collectIsDraggedAsState()
    LaunchedEffect(dragged) { if (dragged) followLatest = false }
    val messages = chat?.messages ?: emptyList()
    val lastText = messages.lastOrNull()?.text ?: ""
    LaunchedEffect(chat?.id, messages.size, lastText.length, followLatest) {
        if (followLatest && messages.isNotEmpty()) {
            scroll.scrollToItem(messages.lastIndex)
            scroll.scrollBy(100000f) // keep the end of a long streaming reply visible
        }
    }
    Column(Modifier.fillMaxSize().padding(horizontal = if (compact) 16.dp else 32.dp)) {
        Row(Modifier.fillMaxWidth().padding(vertical = 10.dp), verticalAlignment = Alignment.CenterVertically) {
            if (compact) {
                TextButton(onClick = { showChats = true }) { Text("History") }
                if (showChats) AlertDialog(onDismissRequest = { showChats = false }, title = { Text("Conversations") }, text = {
                    Column { vm.conversations.forEach { c -> TextButton(onClick = { vm.selectChat(c.id); showChats = false }) { Text(c.title) } } }
                }, confirmButton = { TextButton(onClick = { vm.newChat(); showChats = false }) { Text("New chat") } })
            }
            Text(chat?.title ?: "Chat", style = MaterialTheme.typography.titleLarge, maxLines = 1, modifier = Modifier.weight(1f))
            IconButton(onClick = {
                clipboard.setText(AnnotatedString(chat?.messages?.joinToString("\n\n") { "${it.role}: ${it.text}" }.orEmpty()))
                Toast.makeText(context, "Chat copied", Toast.LENGTH_SHORT).show()
            }, modifier = Modifier.semantics { contentDescription = "Copy conversation" }) { Icon(Icons.Default.ContentCopy, "Copy conversation") }
        }
        Row(Modifier.fillMaxWidth().padding(bottom = 8.dp).horizontalScroll(rememberScrollState()),
            verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            if (vm.ghLoggedIn) FilterChip(selected = vm.agentMode, onClick = { vm.toggleAgentMode() },
                label = { Text("Agent") }, leadingIcon = { Icon(Icons.Default.SmartToy, null, Modifier.size(18.dp)) })
            var expanded by remember { mutableStateOf(false) }
            Box {
                OutlinedButton(onClick = { expanded = true }) { Text(vm.selectedModel?.name?.take(24) ?: "Choose model") }
                DropdownMenu(expanded = expanded, onDismissRequest = { expanded = false }) {
                    vm.installed.filterNot { it.speech }.forEach { model -> DropdownMenuItem(text = { Text(model.name) }, onClick = { vm.chooseModel(model); expanded = false }) }
                    if (vm.installed.none { !it.speech }) DropdownMenuItem(text = { Text("Download a GGUF in Models") }, onClick = { expanded = false })
                }
            }
            if (vm.selectedModel?.name?.contains("MiniCPM5", ignoreCase = true) == true) {
                FilterChip(selected = vm.directAnswer, enabled = !vm.busy,
                    onClick = { vm.applyDirectAnswer(true) }, label = { Text("Direct answer") })
                FilterChip(selected = !vm.directAnswer, enabled = !vm.busy,
                    onClick = { vm.applyDirectAnswer(false) }, label = { Text("Automatic reasoning") })
            }
        }
        HorizontalDivider()
        if (vm.busy) Surface(color = Panel, shape = RoundedCornerShape(10.dp), modifier = Modifier.fillMaxWidth().padding(top = 8.dp)) {
            Column(Modifier.padding(12.dp)) {
                Text(vm.toolStatus.ifBlank { vm.status }, style = MaterialTheme.typography.titleSmall, color = Indigo)
                Text("${vm.liveBackend.ifBlank { if (vm.useGpu) "GPU requested · verifying backend" else "CPU requested" }} · ${vm.contextTokens / 1024}K context · thermal ${vm.liveThermal}",
                    style = MaterialTheme.typography.bodySmall, color = if (vm.liveThermal in listOf("severe", "critical", "emergency")) Color(0xFFFFB4A0) else Color.LightGray)
                if (vm.liveThermal in listOf("severe", "critical", "emergency")) Text("Device is hot; generation may slow down.", style = MaterialTheme.typography.bodySmall, color = Color(0xFFFFB4A0))
            }
        }
        if (messages.isEmpty()) Box(Modifier.weight(1f).fillMaxWidth(), contentAlignment = Alignment.Center) {
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                Text("What would you like to work on?", style = MaterialTheme.typography.headlineMedium)
                Text("Your chat runs on your tablet, even when offline.", color = Color.LightGray)
            }
        } else Box(Modifier.weight(1f).fillMaxWidth()) {
          LazyColumn(state = scroll, modifier = Modifier.fillMaxSize(), contentPadding = PaddingValues(vertical = 24.dp), verticalArrangement = Arrangement.spacedBy(14.dp)) {
            items(messages.size) { i ->
                val message = messages[i]
                when (message.role) {
                    "tool_result" -> ToolResultCard(message.text)
                    else -> Row(Modifier.fillMaxWidth(), horizontalArrangement = if (message.role == "user") Arrangement.End else Arrangement.Start) {
                        Surface(color = if (message.role == "user") Color(0xFF304C7C) else Panel, shape = RoundedCornerShape(16.dp), modifier = Modifier.widthIn(max = 760.dp)) {
                            Column(Modifier.padding(16.dp)) {
                                Text(if (message.role == "user") "YOU" else "ASSISTANT", style = MaterialTheme.typography.labelSmall, color = Indigo)
                                TextButton(onClick = { clipboard.setText(AnnotatedString(message.text)) }) { Text("Copy") }
                                Spacer(Modifier.height(6.dp))
                                if (message.role == "assistant") AssistantReply(message.text) else Text(message.text, style = MaterialTheme.typography.bodyLarge)
                                if (message.perf.isNotBlank() && message.role == "assistant") {
                                    Spacer(Modifier.height(8.dp))
                                    AssistChip(onClick = {
                                        clipboard.setText(AnnotatedString("${message.perf}\n${vm.deviceSummary()}"))
                                        Toast.makeText(context, "Run metrics copied", Toast.LENGTH_SHORT).show()
                                    }, label = { Text(message.perf, style = MaterialTheme.typography.labelSmall) })
                                }
                            }
                        }
                    }
                }
            }
          }
          if (!followLatest) FilledTonalButton(onClick = { followLatest = true }, modifier = Modifier.align(Alignment.BottomEnd).padding(12.dp)) {
              Text("Latest ↓")
          }
        }
        if (vm.listening) Text("Recording… Tap the microphone to stop. Maximum 30 seconds.", color = Indigo)
        Row(Modifier.fillMaxWidth().padding(vertical = 12.dp), verticalAlignment = Alignment.Bottom) {
            OutlinedTextField(value = draft, onValueChange = { draft = it }, modifier = Modifier.weight(1f), minLines = 1, maxLines = 5,
                label = { Text("Message or edit voice transcript") }, placeholder = { Text("Ask anything about your work…") })
            Spacer(Modifier.width(8.dp))
            IconButton(onClick = onMic, modifier = Modifier.size(52.dp).semantics { contentDescription = if (vm.listening) "Stop recording" else "Record voice prompt" }) { Icon(if (vm.listening) Icons.Default.Stop else Icons.Default.Mic, null) }
            if (vm.busy) IconButton(onClick = vm::stop, modifier = Modifier.size(52.dp)) { Icon(Icons.Default.Stop, "Stop generation") }
            else IconButton(onClick = { vm.send(draft); if (vm.selectedModel != null && draft.isNotBlank()) draft = "" }, enabled = draft.isNotBlank(), modifier = Modifier.size(52.dp)) { Icon(Icons.AutoMirrored.Filled.Send, "Send message") }
        }
    }
}
@Composable private fun AssistantReply(raw: String) {
    if (raw.isEmpty()) { Text("Thinking…", style = MaterialTheme.typography.bodyLarge); return }
    val start = raw.indexOf("<think>")
    if (start < 0) { Text(raw, style = MaterialTheme.typography.bodyLarge); return }
    val end = raw.indexOf("</think>", start + 7)
    val thought = raw.substring(start + 7, if (end < 0) raw.length else end).trim()
    val reply = (raw.substring(0, start) + if (end < 0) "" else raw.substring(end + 8)).trim()
    var expanded by remember { mutableStateOf(false) }
    TextButton(onClick = { expanded = !expanded }) {
        Text(if (expanded) "Hide reasoning ↑" else if (end < 0) "Reasoning in progress ↓" else "Show reasoning ↓")
    }
    if (expanded) Surface(color = Console, shape = RoundedCornerShape(8.dp)) {
        Text(thought.ifBlank { "No reasoning text yet." }, modifier = Modifier.padding(12.dp),
            style = MaterialTheme.typography.bodySmall, color = Color.LightGray)
    }
    if (reply.isNotBlank()) Text(reply, style = MaterialTheme.typography.bodyLarge)
    else if (end >= 0) Text("Preparing the final answer…", style = MaterialTheme.typography.bodySmall, color = Color.LightGray)
}
@Composable private fun ToolResultCard(raw: String) {
    var expanded by remember { mutableStateOf(false) }
    val name = raw.substringAfter("[TOOL RESULT name=", "").substringBefore(']').ifBlank { "Tool result" }
    Surface(color = Console, shape = RoundedCornerShape(10.dp), modifier = Modifier.fillMaxWidth().padding(horizontal = 12.dp)) {
        Column(Modifier.padding(horizontal = 12.dp, vertical = 6.dp)) {
            TextButton(onClick = { expanded = !expanded }) { Text("${if (raw.contains("\"ok\":false")) "⚠" else "✓"} $name ${if (expanded) "↑" else "↓"}") }
            if (expanded) Text(raw, style = MaterialTheme.typography.bodySmall, fontFamily = FontFamily.Monospace,
                color = Color(0xFF9FB2CC), modifier = Modifier.padding(bottom = 8.dp))
        }
    }
}
@Composable private fun ModelsPage(vm: WorkbenchViewModel, onImport: () -> Unit) {
    var query by remember { mutableStateOf("") }
    var tab by remember { mutableIntStateOf(0) }
    var deleting by remember { mutableStateOf<LocalModel?>(null) }
    deleting?.let { target -> AlertDialog(onDismissRequest = { deleting = null },
        title = { Text("Delete downloaded model?") }, text = { Text("${target.name} will be removed from this device.") },
        confirmButton = { TextButton(onClick = { vm.deleteModel(target); deleting = null }) { Text("Delete") } },
        dismissButton = { TextButton(onClick = { deleting = null }) { Text("Cancel") } }) }
    Column(Modifier.fillMaxSize().padding(24.dp)) {
        Text("Model library", style = MaterialTheme.typography.headlineMedium)
        TabRow(selectedTabIndex = tab, modifier = Modifier.padding(top = 12.dp)) {
            Tab(selected = tab == 0, onClick = { tab = 0 }, text = { Text("Models & downloads") })
            Tab(selected = tab == 1, onClick = { tab = 1 }, text = { Text("Inference settings") })
        }
        if (tab == 0) {
        Text("Search Hugging Face GGUF models. Only downloaded files are used for inference.", color = Color.LightGray, modifier = Modifier.padding(top = 12.dp))
        Spacer(Modifier.height(16.dp))
        Row(verticalAlignment = Alignment.CenterVertically) {
            OutlinedTextField(query, { query = it }, label = { Text("Model or publisher") }, singleLine = true, modifier = Modifier.weight(1f))
            Spacer(Modifier.width(8.dp))
            Button(onClick = { vm.search(query) }) { Icon(Icons.Default.Search, null); Spacer(Modifier.width(6.dp)); Text("Search") }
        }
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.padding(vertical = 12.dp)) {
            OutlinedButton(onClick = onImport) { Icon(Icons.Default.FolderOpen, null); Spacer(Modifier.width(6.dp)); Text("Import GGUF") }
            OutlinedButton(onClick = vm::installSpeechModel) { Icon(Icons.Default.Mic, null); Spacer(Modifier.width(6.dp)); Text("Download offline speech model") }
        }
        }
        if (tab == 1) Column(Modifier.weight(1f).verticalScroll(rememberScrollState()).padding(top = 12.dp)) {
        Surface(color = Panel, shape = RoundedCornerShape(12.dp), modifier = Modifier.fillMaxWidth()) {
            Column(Modifier.padding(12.dp)) {
                Text("Compute threads: ${vm.genThreads}", style = MaterialTheme.typography.bodyMedium)
                Text("4 is recommended on 8-core phones. Known llama.cpp bug (#28878): 6+ threads can crash during generation on Android. If the app closes itself mid-answer, set 2 or 1.",
                    style = MaterialTheme.typography.bodySmall, color = Color.LightGray)
                Row(horizontalArrangement = Arrangement.spacedBy(6.dp), modifier = Modifier.padding(top = 8.dp).horizontalScroll(rememberScrollState())) {
                    listOf(1, 2, 3, 4, 6, 8).forEach { n ->
                        FilterChip(selected = vm.genThreads == n, onClick = { vm.applyGenThreads(n) }, label = { Text("$n") })
                    }
                }
                Text("Inference backend", modifier = Modifier.padding(top = 12.dp), style = MaterialTheme.typography.bodyMedium)
                Row(modifier = Modifier.horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    FilterChip(selected = vm.useGpu, onClick = { vm.applyGpu(true) }, enabled = !vm.busy, label = { Text("GPU (Vulkan, CPU fallback)") })
                    FilterChip(selected = !vm.useGpu, onClick = { vm.applyGpu(false) }, enabled = !vm.busy, label = { Text("CPU") })
                }
                Text("Actual backend and fallback are shown in Stats after each reply.", style = MaterialTheme.typography.bodySmall, color = Color.LightGray)
                Text("Context window: ${vm.contextTokens} tokens", modifier = Modifier.padding(top = 12.dp), style = MaterialTheme.typography.bodyMedium)
                Row(modifier = Modifier.horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    listOf(4096, 8192, 16384).forEach { n ->
                        FilterChip(selected = vm.contextTokens == n, onClick = { vm.applyContextTokens(n) }, enabled = !vm.busy, label = { Text("${n / 1024}K") })
                    }
                }
                Text("Larger windows need more memory and increase prompt processing time.", style = MaterialTheme.typography.bodySmall, color = Color.LightGray)
            }
        }
        }
        if (tab == 0) LazyColumn(modifier = Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            item { Text("DOWNLOADED", style = MaterialTheme.typography.labelMedium, color = Indigo) }
            items(vm.installed, key = { it.file.absolutePath }) { model ->
                Surface(shape = RoundedCornerShape(12.dp), color = Panel) {
                    Row(Modifier.fillMaxWidth().padding(12.dp), verticalAlignment = Alignment.CenterVertically) {
                        Column(Modifier.weight(1f)) { Text(model.name, maxLines = 1); Text("${model.file.length() / 1048576} MiB · ${if (model.speech) "Speech" else "GGUF"}", style = MaterialTheme.typography.bodySmall) }
                        if (!model.speech) TextButton(onClick = { vm.chooseModel(model) }) { Text("Use") }
                        IconButton(onClick = { deleting = model }) { Icon(Icons.Default.DeleteOutline, "Delete ${model.name}") }
                    }
                }
            }
            item { Spacer(Modifier.height(12.dp)); Text("DOWNLOADS", style = MaterialTheme.typography.labelMedium, color = Indigo) }
            items(vm.transfers, key = { it.id }) { transfer ->
                Column(Modifier.fillMaxWidth().padding(8.dp)) {
                    Text(transfer.id, maxLines = 1); Text(transfer.status, style = MaterialTheme.typography.bodySmall)
                    if (transfer.total > 0) LinearProgressIndicator(progress = { (transfer.done.toFloat() / transfer.total).coerceIn(0f, 1f) }, modifier = Modifier.fillMaxWidth())
                    else LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
                    Text(if (transfer.total > 0) "${transfer.done / 1048576} / ${transfer.total / 1048576} MiB" else "${transfer.done / 1048576} MiB", style = MaterialTheme.typography.bodySmall)
                }
            }
            item { Spacer(Modifier.height(12.dp)); Text("HUGGING FACE RESULTS", style = MaterialTheme.typography.labelMedium, color = Indigo) }
            items(vm.results, key = { "${it.repo}/${it.filename}" }) { remote ->
                Surface(color = Panel, shape = RoundedCornerShape(12.dp)) {
                    Row(Modifier.fillMaxWidth().padding(12.dp), verticalAlignment = Alignment.CenterVertically) {
                        Column(Modifier.weight(1f)) {
                            Text(remote.filename, maxLines = 1, overflow = TextOverflow.Ellipsis)
                            Text("${remote.repo} · ${if (remote.bytes > 0) "${remote.bytes / 1048576} MiB" else "size unknown"}${if (remote.gated) " · gated" else ""}", style = MaterialTheme.typography.bodySmall)
                        }
                        Button(onClick = { vm.download(remote) }, enabled = !remote.gated) { Text("Download") }
                    }
                }
            }
        }
    }
}
@Composable private fun WorkspacePage(vm: WorkbenchViewModel) {
    var folder by remember { mutableStateOf("") }
    var opened by remember { mutableStateOf<String?>(null) }
    var exportPath by remember { mutableStateOf("") }
    val export = rememberLauncherForActivityResult(ActivityResultContracts.CreateDocument("*/*")) { uri ->
        if (uri != null) vm.exportWorkspace(exportPath, uri)
    }
    val files = remember(folder, vm.fileRevision) { runCatching { vm.listWorkspace(folder) }.getOrDefault(emptyList()) }
    fun save(path: String, filename: String) {
        exportPath = path
        export.launch(filename)
    }
    Column(Modifier.fillMaxSize().padding(24.dp)) {
        Text("Files", style = MaterialTheme.typography.headlineMedium)
        Text("Projects created by the assistant appear here. Open a file or export a folder as ZIP.",
            style = MaterialTheme.typography.bodySmall, color = Color.LightGray)
        Row(Modifier.fillMaxWidth().padding(vertical = 12.dp), verticalAlignment = Alignment.CenterVertically) {
            if (folder.isNotEmpty()) TextButton(onClick = { folder = folder.substringBeforeLast('/', ""); opened = null }) { Text("← Back") }
            Text(if (folder.isEmpty()) "Workspace" else "Workspace / $folder", modifier = Modifier.weight(1f), maxLines = 1, overflow = TextOverflow.Ellipsis)
        }
        Row(Modifier.fillMaxWidth().padding(bottom = 8.dp), horizontalArrangement = Arrangement.End) {
            TextButton(onClick = { vm.refreshFiles() }) { Text("Refresh") }
            OutlinedButton(onClick = { save(folder, if (folder.isEmpty()) "workspace.zip" else folder.substringAfterLast('/') + ".zip") }) { Text("Export ZIP") }
        }
        HorizontalDivider()
        LazyColumn(modifier = Modifier.weight(1f).fillMaxWidth(), verticalArrangement = Arrangement.spacedBy(6.dp)) {
            if (files.isEmpty()) item { Text("No files yet. Ask the assistant to create a project in Chat.", modifier = Modifier.padding(16.dp)) }
            items(files, key = { it.absolutePath }) { file ->
                val relative = if (folder.isEmpty()) file.name else "$folder/${file.name}"
                Surface(color = Panel, shape = RoundedCornerShape(10.dp)) {
                    Row(Modifier.fillMaxWidth().padding(8.dp), verticalAlignment = Alignment.CenterVertically) {
                        TextButton(onClick = {
                            if (file.isDirectory) { folder = relative; opened = null } else opened = relative
                        }, modifier = Modifier.weight(1f)) {
                            Text("${if (file.isDirectory) "📁" else "📄"} ${file.name}", maxLines = 1, overflow = TextOverflow.Ellipsis)
                        }
                        TextButton(onClick = { save(relative, file.name + if (file.isDirectory) ".zip" else "") }) { Text("Export") }
                    }
                }
            }
        }
        opened?.let { path ->
            val preview = remember(path, vm.fileRevision) { runCatching { vm.previewWorkspace(path) }.getOrElse { it.message ?: "Cannot open file" } }
            AlertDialog(onDismissRequest = { opened = null }, title = { Text(path.substringAfterLast('/')) },
                text = { SelectionContainer { Text(preview, modifier = Modifier.heightIn(max = 360.dp).verticalScroll(rememberScrollState())) } },
                confirmButton = { TextButton(onClick = { opened = null }) { Text("Close") } })
        }
    }
}
@Composable private fun GitHubPage(vm: WorkbenchViewModel) {
    val context = LocalContext.current
    Column(Modifier.fillMaxSize().padding(24.dp)) {
        Text("GitHub", style = MaterialTheme.typography.headlineMedium)
        Text(vm.ghStatus, style = MaterialTheme.typography.bodyMedium, color = Indigo, modifier = Modifier.padding(vertical = 8.dp))
        if (vm.ghLoggedIn) {
            Surface(color = Panel, shape = RoundedCornerShape(12.dp), modifier = Modifier.fillMaxWidth()) {
                Row(Modifier.padding(16.dp), verticalAlignment = Alignment.CenterVertically) {
                    Icon(Icons.Default.AccountCircle, null, tint = Indigo, modifier = Modifier.size(40.dp))
                    Spacer(Modifier.width(12.dp))
                    Column(Modifier.weight(1f)) {
                        Text(vm.ghLogin, style = MaterialTheme.typography.titleMedium)
                        Text("Scopes: ${vm.ghScopes.ifBlank { "repo, workflow" }}", style = MaterialTheme.typography.bodySmall, color = Color.LightGray)
                    }
                }
            }
            Spacer(Modifier.height(12.dp))
            Surface(color = Panel, shape = RoundedCornerShape(12.dp), modifier = Modifier.fillMaxWidth()) {
                Row(Modifier.padding(16.dp), verticalAlignment = Alignment.CenterVertically) {
                    Column(Modifier.weight(1f)) {
                        Text("Agent tool mode", style = MaterialTheme.typography.titleSmall)
                        Text("Let the local model call GitHub tools (repos, files, issues, PRs, Actions) inside chat.", style = MaterialTheme.typography.bodySmall, color = Color.LightGray)
                    }
                    Switch(checked = vm.agentMode, onCheckedChange = { vm.toggleAgentMode() })
                }
            }
            if (vm.toolStatus.isNotBlank()) Text(vm.toolStatus, style = MaterialTheme.typography.bodySmall, color = Indigo, modifier = Modifier.padding(top = 8.dp))
            Spacer(Modifier.height(16.dp))
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                OutlinedButton(onClick = { vm.testGh() }) { Text("Test connection") }
                OutlinedButton(onClick = { vm.signOutGh() }) { Icon(Icons.Default.Logout, null, Modifier.size(18.dp)); Spacer(Modifier.width(6.dp)); Text("Sign out") }
            }
            Spacer(Modifier.height(16.dp))
            Text("Token is stored encrypted with the Android Keystore and used only for GitHub API calls from this app.", style = MaterialTheme.typography.bodySmall, color = Color.LightGray)
        } else if (vm.ghPolling && vm.ghUserCode.isNotBlank()) {
            Surface(color = Panel, shape = RoundedCornerShape(16.dp), modifier = Modifier.fillMaxWidth()) {
                Column(Modifier.padding(24.dp), horizontalAlignment = Alignment.CenterHorizontally) {
                    Text("1. Open github.com/login/device", style = MaterialTheme.typography.titleMedium)
                    Spacer(Modifier.height(4.dp))
                    Text("2. Enter this one-time code", style = MaterialTheme.typography.titleMedium)
                    Spacer(Modifier.height(12.dp))
                    Surface(color = Console, shape = RoundedCornerShape(12.dp)) {
                        Text(vm.ghUserCode, fontFamily = FontFamily.Monospace, fontSize = 32.sp, color = Indigo, modifier = Modifier.padding(horizontal = 24.dp, vertical = 12.dp))
                    }
                    Spacer(Modifier.height(16.dp))
                    Button(onClick = {
                        try { context.startActivity(Intent(Intent.ACTION_VIEW, Uri.parse("https://github.com/login/device"))) }
                        catch (_: Exception) { Toast.makeText(context, "No browser available; open the URL manually", Toast.LENGTH_SHORT).show() }
                    }) { Icon(Icons.Default.OpenInNew, null, Modifier.size(18.dp)); Spacer(Modifier.width(6.dp)); Text("Open github.com/login/device") }
                    Spacer(Modifier.height(8.dp))
                    OutlinedButton(onClick = { vm.cancelGhLogin() }) { Text("Cancel sign-in") }
                }
            }
        } else {
            Surface(color = Panel, shape = RoundedCornerShape(12.dp), modifier = Modifier.fillMaxWidth()) {
                Column(Modifier.padding(16.dp)) {
                    Text("Sign in with OAuth Device Flow", style = MaterialTheme.typography.titleMedium)
                    Spacer(Modifier.height(8.dp))
                    Text("One-time setup: on github.com go to Settings → Developer settings → OAuth Apps → New OAuth App. Any callback URL works; check \"Enable Device Flow\", then paste the Client ID here.", style = MaterialTheme.typography.bodySmall, color = Color.LightGray)
                    Spacer(Modifier.height(12.dp))
                    var clientId by remember(vm.ghClientId) { mutableStateOf(vm.ghClientId) }
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        OutlinedTextField(clientId, { clientId = it }, label = { Text("OAuth App Client ID") }, singleLine = true, modifier = Modifier.weight(1f))
                        Spacer(Modifier.width(8.dp))
                        OutlinedButton(onClick = { vm.saveGhClientId(clientId) }) { Text("Save") }
                    }
                    Spacer(Modifier.height(12.dp))
                    Button(onClick = { vm.startGhLogin() }, enabled = vm.ghClientId.isNotBlank()) {
                        Icon(Icons.Default.OpenInNew, null, Modifier.size(18.dp)); Spacer(Modifier.width(6.dp)); Text("Sign in with GitHub")
                    }
                }
            }
            Spacer(Modifier.height(12.dp))
            Surface(color = Panel, shape = RoundedCornerShape(12.dp), modifier = Modifier.fillMaxWidth()) {
                Column(Modifier.padding(16.dp)) {
                    Text("Why sign in?", style = MaterialTheme.typography.titleSmall)
                    Text("Local project tools work without sign-in. Connect GitHub to let the assistant read repositories, commit files, open issues and PRs, and inspect or trigger Actions builds.", style = MaterialTheme.typography.bodySmall, color = Color.LightGray)
                }
            }
        }
    }
}
