package com.pocketworkbench.app

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.Send
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.ChatBubbleOutline
import androidx.compose.material.icons.filled.DeleteOutline
import androidx.compose.material.icons.filled.Download
import androidx.compose.material.icons.filled.FolderOpen
import androidx.compose.material.icons.filled.Mic
import androidx.compose.material.icons.filled.Search
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material.icons.filled.Terminal
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import kotlinx.coroutines.launch

private val Indigo = Color(0xFF94B5FF)
private val Dark = Color(0xFF101624)
private val Panel = Color(0xFF1B2535)
private val Pale = Color(0xFFDDE7FF)

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

private enum class Page { Chat, Models, Workspace }
@Composable private fun Workbench(vm: WorkbenchViewModel, onMic: () -> Unit, onImport: () -> Unit) {
    var page by remember { mutableStateOf(Page.Chat) }
    val wide = LocalConfiguration.current.screenWidthDp >= 840
    Surface(modifier = Modifier.fillMaxSize(), color = Dark) {
        Column {
            Row(Modifier.fillMaxWidth().padding(horizontal = 20.dp, vertical = 12.dp), verticalAlignment = Alignment.CenterVertically) {
                Text("POCKET WORKBENCH", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold, color = Indigo)
                Spacer(Modifier.weight(1f))
                Text(if (vm.busy) "● Running on device" else "● Local & private", style = MaterialTheme.typography.labelMedium)
            }
            HorizontalDivider(color = Color(0xFF344356))
            Row(Modifier.weight(1f)) {
                NavigationRail(containerColor = Dark) {
                    Spacer(Modifier.height(20.dp))
                    NavigationRailItem(selected = page == Page.Chat, onClick = { page = Page.Chat }, icon = { Icon(Icons.Default.ChatBubbleOutline, "Chat") }, label = { Text("Chat") })
                    NavigationRailItem(selected = page == Page.Models, onClick = { page = Page.Models }, icon = { Icon(Icons.Default.Download, "Models") }, label = { Text("Models") })
                    NavigationRailItem(selected = page == Page.Workspace, onClick = { page = Page.Workspace }, icon = { Icon(Icons.Default.Terminal, "Workspace") }, label = { Text("Files") })
                }
                if (page == Page.Chat && wide) ConversationSidebar(vm, Modifier.width(250.dp).fillMaxHeight())
                Box(Modifier.weight(1f).fillMaxHeight()) {
                    when (page) {
                        Page.Chat -> ChatPage(vm, onMic, !wide)
                        Page.Models -> ModelsPage(vm, onImport)
                        Page.Workspace -> WorkspacePage(vm)
                    }
                }
            }
            HorizontalDivider(color = Color(0xFF344356))
            Text(vm.status, modifier = Modifier.fillMaxWidth().padding(horizontal = 24.dp, vertical = 8.dp), style = MaterialTheme.typography.bodySmall, maxLines = 2)
        }
    }
}
@Composable private fun ConversationSidebar(vm: WorkbenchViewModel, modifier: Modifier = Modifier) {
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
                    IconButton(onClick = { vm.deleteChat(conversation.id) }, enabled = !vm.busy, modifier = Modifier.size(36.dp)) {
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
    LaunchedEffect(vm.transcript) { if (vm.transcript.isNotBlank()) { draft = vm.transcript; vm.clearTranscript() } }
    val chat = vm.active
    val scroll = rememberLazyListState()
    val messages = chat?.messages ?: emptyList()
    val lastText = messages.lastOrNull()?.text ?: ""
    LaunchedEffect(chat?.id, messages.size, lastText.length) { if (messages.isNotEmpty()) scroll.animateScrollToItem(messages.lastIndex) }
    Column(Modifier.fillMaxSize().padding(horizontal = if (compact) 16.dp else 32.dp)) {
        Row(Modifier.fillMaxWidth().padding(vertical = 10.dp), verticalAlignment = Alignment.CenterVertically) {
            if (compact) {
                TextButton(onClick = { showChats = true }) { Text("History") }
                if (showChats) AlertDialog(onDismissRequest = { showChats = false }, title = { Text("Conversations") }, text = {
                    Column { vm.conversations.forEach { c -> TextButton(onClick = { vm.selectChat(c.id); showChats = false }) { Text(c.title) } } }
                }, confirmButton = { TextButton(onClick = { vm.newChat(); showChats = false }) { Text("New chat") } })
            }
            Text(chat?.title ?: "Chat", style = MaterialTheme.typography.titleLarge, maxLines = 1, modifier = Modifier.weight(1f))
            var expanded by remember { mutableStateOf(false) }
            Box {
                OutlinedButton(onClick = { expanded = true }) { Text(vm.selectedModel?.name?.take(24) ?: "Choose model") }
                DropdownMenu(expanded = expanded, onDismissRequest = { expanded = false }) {
                    vm.installed.filterNot { it.speech }.forEach { model -> DropdownMenuItem(text = { Text(model.name) }, onClick = { vm.chooseModel(model); expanded = false }) }
                    if (vm.installed.none { !it.speech }) DropdownMenuItem(text = { Text("Download a GGUF in Models") }, onClick = { expanded = false })
                }
            }
        }
        HorizontalDivider()
        if (messages.isEmpty()) Box(Modifier.weight(1f).fillMaxWidth(), contentAlignment = Alignment.Center) {
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                Text("What would you like to work on?", style = MaterialTheme.typography.headlineMedium)
                Text("Your chat runs on your tablet, even when offline.", color = Color.LightGray)
            }
        } else LazyColumn(state = scroll, modifier = Modifier.weight(1f).fillMaxWidth(), contentPadding = PaddingValues(vertical = 24.dp), verticalArrangement = Arrangement.spacedBy(14.dp)) {
            items(messages.size) { i ->
                val message = messages[i]
                Row(Modifier.fillMaxWidth(), horizontalArrangement = if (message.role == "user") Arrangement.End else Arrangement.Start) {
                    Surface(color = if (message.role == "user") Color(0xFF304C7C) else Panel, shape = RoundedCornerShape(16.dp), modifier = Modifier.widthIn(max = 760.dp)) {
                        Column(Modifier.padding(16.dp)) {
                            Text(if (message.role == "user") "YOU" else "ASSISTANT", style = MaterialTheme.typography.labelSmall, color = Indigo)
                            Spacer(Modifier.height(6.dp))
                            Text(message.text.ifEmpty { "Thinking…" }, style = MaterialTheme.typography.bodyLarge)
                        }
                    }
                }
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
@Composable private fun ModelsPage(vm: WorkbenchViewModel, onImport: () -> Unit) {
    var query by remember { mutableStateOf("") }
    val scope = rememberCoroutineScope()
    Column(Modifier.fillMaxSize().padding(24.dp)) {
        Text("Model library", style = MaterialTheme.typography.headlineMedium)
        Text("Search Hugging Face GGUF models. Only downloaded files are used for inference.", color = Color.LightGray)
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
        LazyColumn(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            item { Text("DOWNLOADED", style = MaterialTheme.typography.labelMedium, color = Indigo) }
            items(vm.installed, key = { it.file.absolutePath }) { model ->
                Surface(shape = RoundedCornerShape(12.dp), color = Panel) {
                    Row(Modifier.fillMaxWidth().padding(12.dp), verticalAlignment = Alignment.CenterVertically) {
                        Column(Modifier.weight(1f)) { Text(model.name, maxLines = 1); Text("${model.file.length() / 1048576} MiB · ${if (model.speech) "Speech" else "GGUF"}", style = MaterialTheme.typography.bodySmall) }
                        if (!model.speech) TextButton(onClick = { vm.chooseModel(model) }) { Text("Use") }
                        IconButton(onClick = { vm.deleteModel(model) }) { Icon(Icons.Default.DeleteOutline, "Delete ${model.name}") }
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
    var command by remember { mutableStateOf("") }
    Column(Modifier.fillMaxSize().padding(24.dp)) {
        Text("Workspace", style = MaterialTheme.typography.headlineMedium)
        Text("App-private Android shell · ${vm.workspace.absolutePath}", style = MaterialTheme.typography.bodySmall, color = Color.LightGray)
        Text("The assistant does not execute commands automatically. This is not a Linux container.", style = MaterialTheme.typography.bodySmall, color = Indigo)
        Spacer(Modifier.height(16.dp))
        Surface(color = Color(0xFF0A0F17), shape = RoundedCornerShape(12.dp), modifier = Modifier.weight(1f).fillMaxWidth()) {
            val state = rememberLazyListState()
            LaunchedEffect(vm.terminalOutput.length) { state.scrollToItem(0) }
            LazyColumn(state = state, modifier = Modifier.padding(16.dp)) { item { Text(vm.terminalOutput, style = MaterialTheme.typography.bodyMedium) } }
        }
        Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.padding(top = 12.dp)) {
            OutlinedTextField(command, { command = it }, label = { Text("Shell command") }, modifier = Modifier.weight(1f), singleLine = true)
            Spacer(Modifier.width(8.dp))
            Button(onClick = { vm.runShell(command); command = "" }, enabled = command.isNotBlank()) { Text("Run") }
        }
    }
}
