package com.pocketworkbench.app

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.animation.expandVertically
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.shrinkVertically
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.horizontalScroll
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
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.Send
import androidx.compose.material.icons.filled.Check
import androidx.compose.material.icons.filled.ContentCopy
import androidx.compose.material.icons.filled.ErrorOutline
import androidx.compose.material.icons.filled.ExpandMore
import androidx.compose.material.icons.filled.FolderOpen
import androidx.compose.material.icons.filled.Psychology
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TextField
import androidx.compose.material3.TextFieldDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.runtime.snapshotFlow
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.rotate
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import org.json.JSONObject
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.sin
import kotlin.random.Random

private val Space = Color(0xFF070C18)
private val Panel = Color(0xFF111B31)
private val PanelDeep = Color(0xFF0A1226)
private val Mint = Color(0xFF7CF5D3)
private val Indigo = Color(0xFF94B5FF)
private val Muted = Color(0xFF8EA2C8)
private val Paper = Color(0xFFDDE7FF)
private val Danger = Color(0xFFFF6B81)

/**
 * The native chat surface.
 *
 * Animations are driven by the runtime's real phase, never by a timer that
 * pretends work is happening: the orb speeds up only while a turn is actually
 * running. When the reduced-motion setting is on, the orbit stops and every
 * transition becomes instant.
 */
@Composable
fun AgentChatScreen(model: AgentViewModel, reducedMotion: Boolean) {
    val state by model.state.collectAsState()
    val clipboard = LocalClipboardManager.current
    val listState = rememberLazyListState()
    var followLatest by remember { mutableStateOf(true) }
    var copied by remember { mutableStateOf(false) }
    val copyAll: () -> Unit = {
        val transcript = transcriptOf(state.entries)
        if (transcript.isNotBlank()) {
            clipboard.setText(AnnotatedString(transcript))
        }
    }
    LaunchedEffect(copied) {
        if (copied) {
            kotlinx.coroutines.delay(1600)
            copied = false
        }
    }

    val fingerprint = state.entries.joinToString("|") { it.key + ":" + entryLength(it) }
    LaunchedEffect(state.entries.size, fingerprint) {
        if (followLatest && state.entries.isNotEmpty()) {
            listState.scrollToItem(state.entries.lastIndex)
        }
    }
    var preview by remember { mutableStateOf<CitationTarget?>(null) }
    val projectRoot = remember(state.projectId, model) {
        java.io.File(model.storeRef.workspaces, state.projectId.ifBlank { "default" })
    }

    Surface(Modifier.fillMaxSize(), color = Space) {
        Column(Modifier.fillMaxSize()) {
            AgentHeader(
                mood = state.mood,
                status = state.status,
                backend = state.backend,
                fallback = state.backendFallback,
                usage = state.lastUsage,
                reducedMotion = reducedMotion,
                canCopyAll = state.entries.isNotEmpty(),
                copied = copied,
                onCopyAll = { copyAll(); copied = true },
            )
            if (state.interrupted) {
                Surface(color = Color(0xFF3A2A14), shape = RoundedCornerShape(12.dp), modifier = Modifier.fillMaxWidth().padding(horizontal = 14.dp, vertical = 4.dp)) {
                    Row(Modifier.padding(12.dp), verticalAlignment = Alignment.CenterVertically) {
                        Icon(Icons.Default.Warning, null, tint = Color(0xFFFFB877), modifier = Modifier.size(18.dp))
                        Spacer(Modifier.width(8.dp))
                        Text(
                            "A previous turn was interrupted. Its effects on files are unknown: check them before retrying.",
                            color = Color(0xFFFFD9B0),
                            fontSize = 13.sp,
                            modifier = Modifier.weight(1f),
                        )
                        TextButton(onClick = model::dismissInterrupted) { Text("OK") }
                    }
                }
            }
            Box(Modifier.weight(1f).fillMaxWidth()) {
                if (state.entries.isEmpty()) {
                    AgentHero(
                        mood = state.mood,
                        busy = state.busy,
                        hasModel = state.modelName != null,
                        reducedMotion = reducedMotion,
                        onSuggest = model::send,
                    )
                } else {
                    LazyColumn(
                        state = listState,
                        modifier = Modifier
                            .fillMaxSize()
                            .padding(horizontal = 14.dp)
                            .semantics { contentDescription = "Conversazione" },
                        contentPadding = PaddingValues(vertical = 12.dp),
                        verticalArrangement = Arrangement.spacedBy(10.dp),
                    ) {
                        items(state.entries, key = { it.key }) { entry ->
                            AgentEntryRow(
                                entry,
                                state.mood,
                                projectRoot = projectRoot,
                                onPreview = { preview = it },
                                onCopy = { text ->
                                    clipboard.setText(AnnotatedString(text))
                                },
                            )
                        }
                        if (state.busy) item(key = "live") { LiveRow(state.mood, reducedMotion) }
                    }
                    if (!followLatest) {
                        Surface(
                            color = Panel,
                            shape = RoundedCornerShape(18.dp),
                            modifier = Modifier.align(Alignment.BottomEnd).padding(12.dp).clickable { followLatest = true },
                        ) {
                            Text("Latest ↓", color = Indigo, fontSize = 12.sp, modifier = Modifier.padding(horizontal = 12.dp, vertical = 6.dp))
                        }
                    }
                }
            }
            state.error?.let { failure ->
                Surface(color = Color(0xFF3A1420), shape = RoundedCornerShape(14.dp), modifier = Modifier.fillMaxWidth().padding(horizontal = 14.dp)) {
                    Column(Modifier.padding(12.dp)) {
                        Text(failure, color = Color(0xFFFFC9D4), fontSize = 13.sp)
                        Row {
                            TextButton(onClick = model::retry) { Text("Riprova") }
                            TextButton(onClick = model::clearError) { Text("Ignora") }
                        }
                    }
                }
            }
            AgentComposer(
                busy = state.busy,
                pending = state.pending,
                onSend = model::send,
                onStop = model::stop,
            )
            preview?.let { target ->
                CitationPreviewDialog(target = target, projectRoot = projectRoot, onClose = { preview = null })
            }
        }
    }
    LaunchedEffect(listState) {
        // Whether the user can still scroll forward is what tells us they moved
        // away from the end. The transcript must not yank them back while they
        // read history.
        snapshotFlow { listState.canScrollForward }.collect { followLatest = !it }
    }
}

private fun entryLength(entry: ChatEntry): Int = when (entry) {
    is ChatEntry.User -> entry.text.length
    is ChatEntry.Assistant -> entry.text.length
    is ChatEntry.Reasoning -> entry.text.length
    is ChatEntry.Notice -> entry.message.length
    is ChatEntry.Tool -> entry.activity.result.length
    is ChatEntry.Checkpoint -> entry.summary.length
}

@Composable
private fun AgentHeader(
    mood: AgentMood,
    status: String,
    backend: String?,
    fallback: Boolean,
    usage: String,
    reducedMotion: Boolean,
    canCopyAll: Boolean,
    copied: Boolean,
    onCopyAll: () -> Unit,
) {
    Column(Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 8.dp)) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            IconButton(
                onClick = onCopyAll,
                enabled = canCopyAll,
                modifier = Modifier.size(34.dp).semantics { contentDescription = "Copia tutta la chat" },
            ) {
                Icon(
                    if (copied) Icons.Default.Check else Icons.Default.ContentCopy,
                    null,
                    tint = if (copied) Mint else if (canCopyAll) Muted else Muted.copy(alpha = 0.4f),
                    modifier = Modifier.size(17.dp),
                )
            }
            AgentOrb(mood = mood, diameter = 34, reducedMotion = reducedMotion)
            Spacer(Modifier.width(10.dp))
            Column(Modifier.weight(1f)) {
                Text(
                    when (mood) {
                        AgentMood.IDLE -> "Pronto"
                        AgentMood.LOADING -> status.ifBlank { "Al lavoro" }
                        AgentMood.THINKING -> "Generazione"
                        AgentMood.WORKING -> "Uso strumento"
                        AgentMood.ERROR -> "Fermato per un errore"
                    },
                    color = Color.White,
                    fontWeight = FontWeight.SemiBold,
                    fontSize = 14.sp,
                )
                val detail = listOfNotNull(
                    status.takeIf { mood == AgentMood.LOADING },
                    backend,
                    if (fallback) "fallback" else null,
                ).joinToString(" · ")
                if (detail.isNotBlank()) {
                    Text(detail, color = Muted, fontSize = 11.sp, maxLines = 1, overflow = TextOverflow.Ellipsis)
                }
            }
        }
        if (usage.isNotBlank()) {
            Text(usage, color = Muted, fontSize = 11.sp, modifier = Modifier.padding(start = 44.dp, top = 2.dp))
        }
    }
}

/** The whole conversation as plain text, so it can leave the app in one tap. */
private fun transcriptOf(entries: List<ChatEntry>): String = buildString {
    for (entry in entries) {
        when (entry) {
            is ChatEntry.User -> appendLine("Tu: ${entry.text}")
            is ChatEntry.Assistant -> appendLine("Agente: ${entry.text.trim()}")
            is ChatEntry.Reasoning -> appendLine("(ragionamento) ${entry.text.trim()}")
            is ChatEntry.Tool -> {
                val a = entry.activity
                appendLine("[tool] ${a.name} ${a.arguments}")
                if (a.running) {
                    appendLine("in corso")
                } else {
                    appendLine("${if (a.ok) "ok" else "errore"}${if (a.durationMs > 0) " (${a.durationMs} ms)" else ""}:")
                    appendLine(a.result)
                }
            }
            is ChatEntry.Notice -> appendLine("[avviso ${entry.level}] ${entry.message}")
            is ChatEntry.Checkpoint ->
                appendLine("[contesto] ${entry.summary}${if (entry.dropped > 0) " (${entry.dropped} elementi compattati)" else ""}")
        }
        appendLine()
    }
}.trim()

@Composable
private fun AgentHero(mood: AgentMood, busy: Boolean, hasModel: Boolean, reducedMotion: Boolean, onSuggest: (String) -> Unit) {
    val suggestions = if (hasModel) {
        listOf(
            "Elenca i file del progetto",
            "Crea un file nota.txt con un riepilogo e poi rileggilo",
            "Cerca una stringa in tutto il progetto",
        )
    } else {
        listOf("Apri la pagina Modelli e carica un GGUF")
    }
    Column(
        Modifier.fillMaxSize().padding(24.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center,
    ) {
        AgentOrb(mood = mood, diameter = 200, reducedMotion = reducedMotion)
        Spacer(Modifier.height(18.dp))
        Text(
            if (hasModel) "Chiedi, e l'agente lavorerà nel progetto." else "Carica un modello per iniziare.",
            color = Color.White,
            fontSize = 20.sp,
            fontWeight = FontWeight.Bold,
        )
        Spacer(Modifier.height(8.dp))
        Text(
            "Tutto gira su questo dispositivo: modello, agente e strumenti.",
            color = Muted,
            fontSize = 13.sp,
        )
        Spacer(Modifier.height(18.dp))
        suggestions.forEach { suggestion ->
            Surface(
                shape = RoundedCornerShape(16.dp),
                color = Panel,
                modifier = Modifier.fillMaxWidth().padding(vertical = 4.dp).clickable(enabled = !busy) { onSuggest(suggestion) },
            ) {
                Text(suggestion, color = Paper, fontSize = 14.sp, modifier = Modifier.padding(14.dp))
            }
        }
    }
}

@Composable
private fun AgentEntryRow(
    entry: ChatEntry,
    mood: AgentMood,
    projectRoot: java.io.File,
    onPreview: (CitationTarget) -> Unit,
    onCopy: (String) -> Unit,
) {
    when (entry) {
        is ChatEntry.User -> Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.End) {
            Surface(color = Color(0xFF2D4B7A), shape = RoundedCornerShape(16.dp)) {
                Column(Modifier.padding(14.dp)) {
                    SelectionContainer { Text(entry.text, color = Color.White, fontSize = 15.sp) }
                    if (entry.queued) {
                        Text("in coda", color = Color(0xFFBBD0F5), fontSize = 11.sp)
                    }
                }
            }
        }
        is ChatEntry.Assistant -> Surface(color = Panel, shape = RoundedCornerShape(16.dp), modifier = Modifier.fillMaxWidth()) {
            Column(Modifier.padding(14.dp)) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text("Agente", color = Indigo, fontWeight = FontWeight.SemiBold, fontSize = 13.sp)
                    Spacer(Modifier.weight(1f))
                    IconButton(
                        onClick = { onCopy(entry.text) },
                        modifier = Modifier.size(28.dp).semantics { contentDescription = "Copy answer" },
                    ) {
                        Icon(Icons.Default.ContentCopy, null, tint = Muted, modifier = Modifier.size(15.dp))
                    }
                }
                if (entry.text.isBlank()) {
                    Text("…", color = Muted)
                } else {
                    RichText(entry.text, projectRoot, onPreview)
                }
            }
        }
        is ChatEntry.Reasoning -> ReasoningBlock(entry.text)
        is ChatEntry.Tool -> ToolRow(entry.activity, onCopy)
        is ChatEntry.Notice -> Surface(color = Color(0xFF2A1620), shape = RoundedCornerShape(14.dp), modifier = Modifier.fillMaxWidth()) {
            Row(Modifier.padding(12.dp), verticalAlignment = Alignment.Top) {
                Icon(Icons.Default.ErrorOutline, null, tint = Color(0xFFFFB4A0), modifier = Modifier.size(16.dp))
                Spacer(Modifier.width(8.dp))
                Column(Modifier.weight(1f)) {
                    Text(entry.level, color = Color(0xFFFFB4A0), fontSize = 11.sp)
                    SelectionContainer { Text(entry.message, color = Color(0xFFFFD9D0), fontSize = 13.sp) }
                }
                IconButton(
                    onClick = { onCopy("[${entry.level}] ${entry.message}") },
                    modifier = Modifier.size(28.dp).semantics { contentDescription = "Copy error" },
                ) {
                    Icon(Icons.Default.ContentCopy, null, tint = Muted, modifier = Modifier.size(15.dp))
                }
            }
        }
        is ChatEntry.Checkpoint -> Surface(color = PanelDeep, shape = RoundedCornerShape(12.dp), modifier = Modifier.fillMaxWidth()) {
            Row(Modifier.padding(12.dp), verticalAlignment = Alignment.CenterVertically) {
                Icon(Icons.Default.FolderOpen, null, tint = Muted, modifier = Modifier.size(16.dp))
                Spacer(Modifier.width(8.dp))
                Text(
                    "${entry.dropped} messaggi compattati. Il testo completo resta nel registro.",
                    color = Muted,
                    fontSize = 12.sp,
                )
            }
        }
    }
}

@Composable
private fun LiveRow(mood: AgentMood, reducedMotion: Boolean) {
    Surface(color = Color(0xFF0D1730), shape = RoundedCornerShape(16.dp), modifier = Modifier.fillMaxWidth()) {
        Row(Modifier.padding(14.dp), verticalAlignment = Alignment.CenterVertically) {
            if (reducedMotion) {
                Text("…", color = Indigo)
            } else {
                CircularProgressIndicator(Modifier.size(14.dp), strokeWidth = 2.dp, color = Mint)
            }
            Spacer(Modifier.width(10.dp))
            Text(
                when (mood) {
                    AgentMood.WORKING -> "Sto eseguendo uno strumento…"
                    AgentMood.LOADING -> "Preparo il contesto…"
                    else -> "Sto ragionando…"
                },
                color = Muted,
                fontSize = 13.sp,
            )
        }
    }
}

@Composable
private fun ReasoningBlock(text: String) {
    var open by remember { mutableStateOf(false) }
    val rotation by animateFloatAsState(if (open) 180f else 0f, label = "reasoning-chevron")
    Surface(color = PanelDeep, shape = RoundedCornerShape(14.dp), modifier = Modifier.fillMaxWidth()) {
        Column(Modifier.padding(12.dp)) {
            Row(
                Modifier.fillMaxWidth().clickable { open = !open },
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Icon(Icons.Default.Psychology, null, tint = Muted, modifier = Modifier.size(16.dp))
                Spacer(Modifier.width(6.dp))
                Text("Ragionamento", color = Muted, fontSize = 13.sp, modifier = Modifier.weight(1f))
                Icon(
                    Icons.Default.ExpandMore,
                    contentDescription = null,
                    tint = Muted,
                    modifier = Modifier.size(16.dp).rotate(rotation),
                )
            }
            AnimatedVisibility(visible = open, enter = expandVertically() + fadeIn(), exit = shrinkVertically() + fadeOut()) {
                SelectionContainer {
                    Text(text, color = Color(0xFFB9C7E8), fontSize = 12.sp, modifier = Modifier.padding(top = 8.dp))
                }
            }
        }
    }
}

@Composable
private fun ToolRow(activity: ToolActivity, onCopy: (String) -> Unit) {
    var open by remember(activity.callId) { mutableStateOf(activity.running) }
    LaunchedEffect(activity.running) { if (activity.running) open = true }
    val summary = remember(activity.callId, activity.arguments) { toolSummary(activity) }
    val preview = remember(activity.callId, activity.result, activity.ok) { toolPreview(activity) }
    Surface(color = PanelDeep, shape = RoundedCornerShape(14.dp), modifier = Modifier.fillMaxWidth()) {
        Column(Modifier.padding(horizontal = 12.dp, vertical = 10.dp)) {
            Row(
                Modifier.fillMaxWidth().clickable { open = !open },
                verticalAlignment = Alignment.CenterVertically,
            ) {
                if (activity.running) {
                    CircularProgressIndicator(Modifier.size(14.dp), strokeWidth = 2.dp, color = Mint)
                } else {
                    Text(if (activity.ok) "✓" else "✗", color = if (activity.ok) Mint else Danger, fontSize = 14.sp)
                }
                Spacer(Modifier.width(8.dp))
                Column(Modifier.weight(1f)) {
                    Text(
                        summary,
                        color = Color.White,
                        fontSize = 13.sp,
                        maxLines = 2,
                        overflow = TextOverflow.Ellipsis,
                    )
                    if (preview.isNotBlank()) {
                        Text(
                            preview,
                            color = Muted,
                            fontSize = 11.sp,
                            maxLines = 3,
                            overflow = TextOverflow.Ellipsis,
                            modifier = Modifier.padding(top = 2.dp),
                        )
                    }
                }
                Spacer(Modifier.width(6.dp))
                Text(
                    when {
                        activity.running -> "in corso"
                        activity.ok && activity.durationMs > 0 -> "${activity.durationMs} ms"
                        activity.ok -> "completato"
                        else -> "fallito"
                    },
                    color = Muted,
                    fontSize = 11.sp,
                )
            }
            AnimatedVisibility(visible = open, enter = expandVertically() + fadeIn(), exit = shrinkVertically() + fadeOut()) {
                Column {
                    if (activity.arguments.isNotBlank()) {
                        Spacer(Modifier.height(6.dp))
                        PrettyJson(activity.arguments, Color(0xFFB9C7E8))
                    }
                    if (activity.result.isNotBlank()) {
                        Spacer(Modifier.height(6.dp))
                        Row(verticalAlignment = Alignment.Top) {
                            SelectionContainer(Modifier.weight(1f)) {
                                Text(
                                    if (activity.truncated) "${activity.result}\n[output ridotto]" else activity.result,
                                    color = if (activity.ok) Paper else Color(0xFFFFB4A0),
                                    fontSize = 12.sp,
                                )
                            }
                            IconButton(
                                onClick = { onCopy("${activity.name}: ${activity.result}") },
                                modifier = Modifier.size(28.dp).semantics { contentDescription = "Copy tool result" },
                            ) {
                                Icon(Icons.Default.ContentCopy, null, tint = Muted, modifier = Modifier.size(15.dp))
                            }
                        }
                    }
                }
            }
        }
    }
}

/** One glanceable line of what the tool was asked to do: the command, the path, the query. */
private fun toolSummary(activity: ToolActivity): String {
    val args = runCatching { org.json.JSONObject(activity.arguments) }.getOrNull()
    return when (activity.name) {
        "shell" -> args?.optString("command").orEmpty().trim().ifBlank { "shell" }
        "fs_read", "fs_write", "fs_edit" -> {
            val path = args?.optString("path").orEmpty().trim()
            "${activity.name.removePrefix("fs_")} $path".trim()
        }
        "fs_list" -> "elenca ${args?.optString("path").orEmpty().trim().ifBlank { "progetto" }}"
        "fs_search" -> "cerca “${args?.optString("query").orEmpty().trim()}”"
        else -> activity.name
    }
}

/** Up to three lines of what came back, or the reason it failed — never blank on failure. */
private fun toolPreview(activity: ToolActivity): String {
    val result = runCatching { org.json.JSONObject(activity.result) }.getOrNull()
        ?: return if (activity.result.isBlank()) "" else activity.result.lineSequence().take(3).joinToString("\n")
    val output = result.optString("output").trim()
    if (output.isNotEmpty()) {
        return output.lineSequence().take(3).joinToString("\n")
    }
    if (!activity.ok || activity.running) {
        val exit = if (result.has("exit_code")) "exit ${result.optInt("exit_code")}" else null
        val note = result.optString("note").trim().ifBlank { null }
        val err = result.optString("error").trim().ifBlank { null }
        return listOfNotNull(exit, note, err, if (exit == null && note == null && err == null) "nessun output" else null)
            .joinToString(" · ")
    }
    return ""
}

@Composable
private fun PrettyJson(raw: String, color: Color) {
    val pretty = remember(raw) { prettify(raw) }
    SelectionContainer {
        Text(pretty, color = color, fontFamily = FontFamily.Monospace, fontSize = 11.sp)
    }
}

private fun prettify(raw: String): String = runCatching { JSONObject(raw).toString(2) }.getOrDefault(raw)

@Composable
private fun AgentComposer(busy: Boolean, pending: Int, onSend: (String) -> Unit, onStop: () -> Unit) {
    var draft by remember { mutableStateOf("") }
    fun submit() {
        val text = draft.trim()
        if (text.isEmpty() || busy) return
        draft = ""
        onSend(text)
    }
    Surface(color = Color(0xFF0D1730), shape = RoundedCornerShape(26.dp), modifier = Modifier.fillMaxWidth().padding(14.dp)) {
        Row(Modifier.padding(6.dp), verticalAlignment = Alignment.Bottom) {
            TextField(
                value = draft,
                onValueChange = { draft = it },
                modifier = Modifier.weight(1f),
                placeholder = { Text("Chiedi all'agente…", color = Color(0xFF5B6B8C)) },
                shape = RoundedCornerShape(20.dp),
                colors = TextFieldDefaults.colors(
                    focusedContainerColor = Color.Transparent,
                    unfocusedContainerColor = Color.Transparent,
                    focusedIndicatorColor = Color.Transparent,
                    unfocusedIndicatorColor = Color.Transparent,
                    focusedTextColor = Color.White,
                    unfocusedTextColor = Color.White,
                    cursorColor = Mint,
                ),
                keyboardOptions = KeyboardOptions(imeAction = ImeAction.Send),
                keyboardActions = KeyboardActions(onSend = { submit() }),
                maxLines = 5,
            )
            IconButton(
                onClick = { if (busy) onStop() else submit() },
                enabled = busy || draft.isNotBlank(),
                modifier = Modifier.size(48.dp).semantics { contentDescription = if (busy) "Interrompi" else "Invia" },
            ) {
                if (busy) Icon(Icons.Default.Stop, null, tint = Danger, modifier = Modifier.size(24.dp))
                else Icon(
                    Icons.AutoMirrored.Filled.Send,
                    null,
                    tint = if (draft.isNotBlank()) Mint else Muted.copy(alpha = 0.4f),
                    modifier = Modifier.size(24.dp),
                )
            }
        }
    }
    if (pending > 0) {
        Text(
            "$pending messaggi in coda",
            color = Muted,
            fontSize = 11.sp,
            modifier = Modifier.fillMaxWidth().padding(horizontal = 18.dp).padding(bottom = 6.dp),
        )
    }
}

private data class Star(val x: Float, val y: Float, val radius: Float, val phase: Float)

/**
 * The orbit. Its speed and colour come from [mood], which the runtime sets from
 * real events; with reduced motion it is a still image of the same states.
 */
@Composable
private fun AgentOrb(mood: AgentMood, diameter: Int, reducedMotion: Boolean) {
    val busy = mood == AgentMood.THINKING || mood == AgentMood.WORKING
    val transition = rememberInfiniteTransition(label = "orb")
    val rotation by transition.animateFloat(0f, 360f, infiniteRepeatable(tween(if (busy) 2600 else 8000, easing = LinearEasing)), label = "orb-rotation")
    val pulse by transition.animateFloat(0.94f, 1.06f, infiniteRepeatable(tween(2200, easing = LinearEasing), RepeatMode.Reverse), label = "orb-pulse")
    val stars = remember {
        List(64) { index ->
            val random = Random(2000 + index)
            Star(random.nextFloat(), random.nextFloat(), 0.7f + random.nextFloat() * 1.6f, random.nextFloat() * (PI.toFloat() * 2f))
        }
    }
    val (core, ring) = when (mood) {
        AgentMood.ERROR -> Color(0xFF8C2138) to Color(0xFFFF6B81)
        AgentMood.WORKING -> Color(0xFF0E7C66) to Mint
        AgentMood.LOADING -> Color(0xFF6B5A16) to Color(0xFFFFD166)
        AgentMood.THINKING -> Color(0xFF2B4B8A) to Indigo
        AgentMood.IDLE -> Color(0xFF1B2E5E) to Color(0xFF5E7BD9)
    }
    val angle = if (reducedMotion) 25f else rotation
    val scale = if (reducedMotion) 1f else pulse
    Canvas(Modifier.size(diameter.dp)) {
        val radius = size.minDimension / 2f
        val center = Offset(size.width / 2f, size.height / 2f)
        stars.forEach { star ->
            val twinkle = 0.2f + 0.4f * abs(sin(star.phase))
            drawCircle(Color.White.copy(alpha = 0.3f * twinkle), radius = star.radius * twinkle, center = Offset(star.x * size.width, star.y * size.height))
        }
        drawCircle(
            Brush.radialGradient(listOf(ring.copy(alpha = 0.26f), Color.Transparent), center, radius * scale),
            radius = radius * scale,
        )
        drawArc(
            color = ring.copy(alpha = 0.85f),
            startAngle = angle,
            sweepAngle = 250f,
            useCenter = false,
            topLeft = center - Offset(radius * 0.72f, radius * 0.72f),
            size = Size(radius * 1.44f, radius * 1.44f),
            style = Stroke(width = 4f),
        )
        drawCircle(
            Brush.radialGradient(listOf(Color.White.copy(alpha = 0.85f), ring, core), center, radius * 0.4f * scale),
            radius = radius * 0.4f * scale,
        )
    }
}

// ------------------------------------------------------------------- rich text

internal sealed interface Segment
internal data class Prose(val text: String) : Segment
internal data class Code(val code: String) : Segment

/** Fenced blocks become code surfaces; everything else is inline-formatted. */
internal fun splitMarkdown(source: String): List<Segment> {
    val segments = mutableListOf<Segment>()
    var cursor = 0
    while (true) {
        val start = source.indexOf("```", cursor)
        if (start < 0) {
            if (cursor < source.length) segments += Prose(source.substring(cursor))
            break
        }
        if (start > cursor) segments += Prose(source.substring(cursor, start))
        val end = source.indexOf("```", start + 3)
        if (end < 0) {
            segments += Code(source.substring(start + 3).trimStart('\n'))
            break
        }
        segments += Code(source.substring(start + 3, end).trimStart('\n'))
        cursor = end + 3
    }
    return segments.filter { segment ->
        when (segment) {
            is Prose -> segment.text.isNotBlank()
            is Code -> segment.code.isNotBlank()
        }
    }
}

@Composable
private fun RichText(
    text: String,
    projectRoot: java.io.File,
    onPreview: (CitationTarget) -> Unit,
) {
    Column {
        splitMarkdown(text).forEach { segment ->
            when (segment) {
                is Code -> Surface(color = Color(0xFF070D1D), shape = RoundedCornerShape(12.dp), modifier = Modifier.fillMaxWidth().padding(vertical = 4.dp)) {
                    Text(
                        segment.code,
                        color = Mint,
                        fontFamily = FontFamily.Monospace,
                        fontSize = 12.sp,
                        modifier = Modifier.horizontalScroll(rememberScrollState()).padding(12.dp),
                    )
                }
                is Prose -> ProseWithSources(segment.text, projectRoot, onPreview)
            }
        }
    }
}

/** A [path:line] pointer the model emitted. Tapping it opens the file, so a
 * factual claim is one tap away from its evidence — or visibly unverified. */
internal data class CitationTarget(val path: String, val line: Int)

private val CitationPattern = Regex("""\[([\w\-./]+):(\d+)\]""")

internal fun parseCitations(text: String): List<CitationTarget> =
    CitationPattern.findAll(text).mapNotNull { match ->
        val path = match.groupValues[1].trim().trimStart('/')
        val line = match.groupValues[2].toIntOrNull() ?: return@mapNotNull null
        if (path.isEmpty() || path.length > 160 || line < 1 || ".." in path) return@mapNotNull null
        CitationTarget(path, line)
    }.distinct().take(12).toList()

private sealed interface CitationCheck {
    data class Found(val totalLines: Int) : CitationCheck
    data class Missing(val reason: String) : CitationCheck
}

/** Resolves a citation against the project on disk. Cheap by design (512 KiB
 * cap): the dialog re-reads the file for display. */
private fun resolveCitation(target: CitationTarget, projectRoot: java.io.File): CitationCheck {
    return try {
        val root = projectRoot.canonicalFile
        val file = java.io.File(root, target.path).canonicalFile
        if (!file.canonicalPath.startsWith(root.canonicalPath + java.io.File.separator)) {
            return CitationCheck.Missing("fuori dal progetto")
        }
        if (!file.isFile) return CitationCheck.Missing("file assente")
        if (file.length() > 512 * 1024) return CitationCheck.Missing("troppo grande")
        val total = file.bufferedReader().useLines { it.count() }
        if (target.line > total) CitationCheck.Missing("riga $total max") else CitationCheck.Found(total)
    } catch (_: Exception) {
        CitationCheck.Missing("illeggibile")
    }
}

@Composable
private fun ProseWithSources(
    text: String,
    projectRoot: java.io.File,
    onPreview: (CitationTarget) -> Unit,
) {
    SelectionContainer { Text(formatInline(text), color = Paper, fontSize = 15.sp, lineHeight = 21.sp) }
    val citations = remember(text) { parseCitations(text) }
    if (citations.isNotEmpty()) {
        val verified = citations.count { resolveCitation(it, projectRoot) is CitationCheck.Found }
        Text(
            "Fonti: $verified/${citations.size} verificate",
            color = if (verified == citations.size) Mint else Color(0xFFFFB877),
            fontSize = 11.sp,
            modifier = Modifier.padding(top = 8.dp),
        )
        Row(
            modifier = Modifier.fillMaxWidth().horizontalScroll(rememberScrollState()).padding(top = 4.dp),
            horizontalArrangement = Arrangement.spacedBy(6.dp),
        ) {
            citations.forEach { citation ->
                val check = remember(citation, projectRoot) { resolveCitation(citation, projectRoot) }
                val ok = check is CitationCheck.Found
                Surface(
                    color = if (ok) Color(0xFF0E3A32) else Color(0xFF3A2A14),
                    shape = RoundedCornerShape(10.dp),
                    modifier = Modifier.clickable { onPreview(citation) },
                ) {
                    Text(
                        (if (ok) "✓ " else "∅ ") + citation.path + ":" + citation.line,
                        color = if (ok) Mint else Color(0xFFFFB877),
                        fontFamily = FontFamily.Monospace,
                        fontSize = 11.sp,
                        maxLines = 1,
                        modifier = Modifier.padding(horizontal = 10.dp, vertical = 6.dp),
                    )
                }
            }
        }
    }
}

@Composable
private fun CitationPreviewDialog(
    target: CitationTarget,
    projectRoot: java.io.File,
    onClose: () -> Unit,
) {
    val lines = remember(target, projectRoot) {
        runCatching {
            val root = projectRoot.canonicalFile
            val file = java.io.File(root, target.path).canonicalFile
            if (!file.canonicalPath.startsWith(root.canonicalPath + java.io.File.separator)) return@runCatching null
            if (!file.isFile || file.length() > 256_000) return@runCatching null
            file.readLines()
        }.getOrNull()
    }
    androidx.compose.material3.AlertDialog(
        onDismissRequest = onClose,
        title = { Text("${target.path}:${target.line}", fontSize = 14.sp) },
        text = {
            if (lines == null) {
                Text("File non apribile dal progetto.", color = Muted, fontSize = 13.sp)
            } else if (target.line > lines.size) {
                Text("Il file ha ${lines.size} righe: citazione oltre la fine.", color = Color(0xFFFFB877), fontSize = 13.sp)
            } else {
                val listState = rememberLazyListState()
                LaunchedEffect(target) { listState.scrollToItem((target.line - 1).coerceAtLeast(0)) }
                LazyColumn(state = listState, modifier = Modifier.height(320.dp)) {
                    items(lines.size) { index ->
                        val selected = index == target.line - 1
                        Surface(
                            color = if (selected) Color(0xFF0E3A32) else Color.Transparent,
                            shape = RoundedCornerShape(6.dp),
                            modifier = Modifier.fillMaxWidth(),
                        ) {
                            Row(Modifier.padding(horizontal = 6.dp, vertical = 1.dp)) {
                                Text(
                                    "${index + 1}",
                                    color = Muted,
                                    fontFamily = FontFamily.Monospace,
                                    fontSize = 11.sp,
                                    modifier = Modifier.width(36.dp),
                                )
                                Text(
                                    lines[index],
                                    color = if (selected) Color.White else Paper,
                                    fontFamily = FontFamily.Monospace,
                                    fontSize = 11.sp,
                                )
                            }
                        }
                    }
                }
            }
        },
        confirmButton = { TextButton(onClick = onClose) { Text("Chiudi") } },
    )
}

private fun formatInline(source: String): AnnotatedString = buildAnnotatedString {
    source.split("**").forEachIndexed { index, part ->
        if (index % 2 == 1) {
            withStyle(SpanStyle(fontWeight = FontWeight.Bold, color = Color.White)) { appendInlineCode(part) }
        } else {
            appendInlineCode(part)
        }
    }
}

private fun AnnotatedString.Builder.appendInlineCode(source: String) {
    source.split("`").forEachIndexed { index, part ->
        if (index % 2 == 1) {
            withStyle(SpanStyle(fontFamily = FontFamily.Monospace, color = Mint)) { append(part) }
        } else {
            append(part)
        }
    }
}