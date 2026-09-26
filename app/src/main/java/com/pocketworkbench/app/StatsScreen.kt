package com.pocketworkbench.app

import android.content.Intent
import android.widget.Toast
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ContentCopy
import androidx.compose.material.icons.filled.DeleteSweep
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Share
import androidx.compose.material.icons.filled.Speed
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

private val SIndigo = Color(0xFF94B5FF)
private val SPanel = Color(0xFF1B2535)
private val SConsole = Color(0xFF0A0F17)
private val SWarn = Color(0xFFFFB4A0)
private val SMuted = Color(0xFF9FB2CC)

@Composable
fun StatsScreen(vm: WorkbenchViewModel) {
    val clipboard = LocalClipboardManager.current
    val context = LocalContext.current
    val entries = vm.perfEntries
    val refresh = vm.engineRefresh
    val (caps, modelInfo) = remember(refresh) { EngineStatsParser.parseInfo(vm.engineInfoJson()) }
    val health = vm.healthReport
    val problems = entries.filter { it.problem.isNotBlank() || it.error.isNotBlank() }
    var confirmClear by remember { mutableStateOf(false) }

    Column(Modifier.fillMaxSize().padding(24.dp)) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text("Statistics & diagnostics", style = MaterialTheme.typography.headlineMedium, modifier = Modifier.weight(1f))
            IconButton(onClick = { vm.refreshEngineInfo() }) { Icon(Icons.Default.Refresh, "Refresh engine info") }
            IconButton(onClick = {
                clipboard.setText(AnnotatedString(vm.exportFullReport()))
                Toast.makeText(context, "Report copied", Toast.LENGTH_SHORT).show()
            }) { Icon(Icons.Default.ContentCopy, "Copy full report") }
            IconButton(onClick = {
                val intent = Intent(Intent.ACTION_SEND).apply { type = "text/plain"; putExtra(Intent.EXTRA_TEXT, vm.exportFullReport()) }
                context.startActivity(Intent.createChooser(intent, "Share report"))
            }) { Icon(Icons.Default.Share, "Share report") }
            IconButton(onClick = { confirmClear = true }) { Icon(Icons.Default.DeleteSweep, "Clear history") }
        }
        if (confirmClear) AlertDialog(
            onDismissRequest = { confirmClear = false },
            title = { Text("Clear run history?") },
            text = { Text("All recorded runs will be removed from this device. Diagnostics logs are kept.") },
            confirmButton = { TextButton(onClick = { vm.clearPerfLog(); confirmClear = false }) { Text("Clear") } },
            dismissButton = { TextButton(onClick = { confirmClear = false }) { Text("Cancel") } }
        )
        LazyColumn(verticalArrangement = Arrangement.spacedBy(10.dp), modifier = Modifier.padding(top = 8.dp)) {
            item { EngineCard(caps, modelInfo) }
            item { HealthCheckCard(vm, health) }
            item { LatestRunCard(entries.firstOrNull()) }
            if (problems.isNotEmpty()) item { ProblemsCard(problems) }
            item { DiagnosticsCard(vm) }
            item { DeviceCard(vm, caps) }
            item { Text("RUN HISTORY (newest first)", style = MaterialTheme.typography.labelMedium, color = SIndigo, modifier = Modifier.padding(top = 4.dp)) }
            if (entries.isEmpty()) item {
                Surface(color = SPanel, shape = RoundedCornerShape(12.dp), modifier = Modifier.fillMaxWidth()) {
                    Text("No runs recorded yet. Send a message in Chat and the technical metrics will appear here.", color = Color.LightGray, modifier = Modifier.padding(16.dp))
                }
            }
            items(entries.size) { i -> HistoryRow(entries[i]) }
        }
    }
}

@Composable
private fun Card(title: String, content: @Composable () -> Unit) {
    Surface(color = SPanel, shape = RoundedCornerShape(12.dp), modifier = Modifier.fillMaxWidth()) {
        Column(Modifier.padding(16.dp)) {
            Text(title.uppercase(), style = MaterialTheme.typography.labelMedium, color = SIndigo)
            Spacer(Modifier.height(6.dp))
            content()
        }
    }
}

@Composable
private fun EngineCard(caps: EngineCapabilities, model: ModelInfo?) {
    Card("Engine") {
        if (caps.loadError != null) {
            Text("FAILED TO LOAD libpocketinfer.so", style = MaterialTheme.typography.titleSmall, color = SWarn)
            Text(caps.loadError, style = MaterialTheme.typography.bodySmall, fontFamily = FontFamily.Monospace, color = SWarn)
            Text("The app keeps running, but chat generation is unavailable. Reinstall the APK or check the ABI/16 KB page compatibility.", style = MaterialTheme.typography.bodySmall, color = Color.LightGray)
        } else {
            Text("${caps.version} · ${caps.abi} · NEON ${yes(caps.neon)} · int8 dotprod ${yes(caps.dotprod)}", style = MaterialTheme.typography.titleSmall)
            Text("OpenCL: ${caps.opencl ?: "not available"} · QNN/NPU: ${caps.qnn.ifBlank { "not available" }}", style = MaterialTheme.typography.bodySmall, color = Color.LightGray)
        }
        Spacer(Modifier.height(8.dp))
        if (model == null) {
            Text("No model loaded yet. Send a message in Chat or run the health check.", style = MaterialTheme.typography.bodySmall, color = Color.LightGray)
        } else {
            Text(model.file, style = MaterialTheme.typography.titleSmall, maxLines = 1, overflow = TextOverflow.Ellipsis)
            Text("${model.arch} · ${model.layers} layers · hidden ${model.hidden} · ${model.heads}/${model.kvHeads} heads · head ${model.headDim} · vocab ${model.vocab}", style = MaterialTheme.typography.bodySmall)
            Text("File ${mb(model.fileBytes)} · KV cache ${mb(model.kvBytes)} · ctx ${model.ctx} · threads ${model.threads} · backend ${model.backend}", style = MaterialTheme.typography.bodySmall, color = Color.LightGray)
            Spacer(Modifier.height(6.dp))
            Text("WEIGHTS BY TYPE", style = MaterialTheme.typography.labelSmall, color = SIndigo)
            model.quant.forEach { q ->
                Text("${q.name}: ${q.tensors} tensors · ${mb(q.bytes)}", style = MaterialTheme.typography.bodySmall, fontFamily = FontFamily.Monospace, color = SMuted, fontSize = 10.sp)
            }
        }
    }
}

@Composable
private fun HealthCheckCard(vm: WorkbenchViewModel, report: DiagReport?) {
    val model = vm.selectedModel ?: vm.installed.firstOrNull()
    Card("Health check") {
        Text("Runs a step-by-step check on the selected model: file, GGUF metadata, tokenizer, engine load, prefill and sampling. Every step reports its own error.", style = MaterialTheme.typography.bodySmall, color = Color.LightGray)
        Spacer(Modifier.height(8.dp))
        Row(verticalAlignment = Alignment.CenterVertically) {
            Button(onClick = { vm.runHealthCheck() }, enabled = !vm.healthRunning) {
                Icon(Icons.Default.Speed, null, Modifier.size(16.dp)); Spacer(Modifier.width(6.dp))
                Text(if (vm.healthRunning) "Running…" else "Run health check")
            }
            Spacer(Modifier.width(10.dp))
            Text(model?.name ?: "no model selected", style = MaterialTheme.typography.bodySmall, color = Color.LightGray, maxLines = 1, overflow = TextOverflow.Ellipsis)
        }
        if (vm.lastEngineError.isNotBlank()) {
            Spacer(Modifier.height(8.dp))
            SqlBox("LAST ENGINE ERROR\n${vm.lastEngineError}", SWarn)
        }
        if (report != null) {
            Spacer(Modifier.height(8.dp))
            Text(if (report.ok) "PASSED" else "FAILED", style = MaterialTheme.typography.titleSmall, color = if (report.ok) Color(0xFF9BE7A8) else SWarn)
            report.steps.forEach { step ->
                val mark = if (step.ok) "OK" else "FAIL"
                val color = if (step.ok) Color(0xFF9BE7A8) else SWarn
                Text("[$mark] ${step.name} · ${String.format(Locale.US, "%.1f", step.ms)} ms", style = MaterialTheme.typography.bodySmall, fontFamily = FontFamily.Monospace, color = color)
                if (step.error.isNotBlank()) Text("       ${step.error}", style = MaterialTheme.typography.bodySmall, fontFamily = FontFamily.Monospace, color = SWarn)
            }
        }
    }
}

@Composable
private fun LatestRunCard(entry: PerfEntry?) {
    if (entry == null) return
    Card("Latest generation") {
        Text("${entry.backend} · ${entry.genTokens} tokens · ${String.format(Locale.US, "%.1f", entry.genTps)} tok/s", style = MaterialTheme.typography.titleMedium)
        Text("${if (entry.reasoning == "direct") "Direct answer" else "Automatic reasoning"} · ${entry.contextTokens / 1024}K context · ${entry.threads} threads · stop: ${entry.stop}", style = MaterialTheme.typography.bodySmall)
        Text("prefill ${String.format(Locale.US, "%.1f", entry.prefillTps)} tok/s · load ${if (entry.cached) "cached" else String.format(Locale.US, "%.1f s", entry.loadMs / 1000.0)} · TTFT ${String.format(Locale.US, "%.1f s", entry.ttftMs / 1000.0)}", style = MaterialTheme.typography.bodySmall, color = Color.LightGray)
        if (entry.error.isNotBlank()) {
            Spacer(Modifier.height(6.dp))
            SqlBox(entry.error, SWarn)
        }
        if (entry.problem.isNotBlank()) Text("⚠ ${entry.problem}", style = MaterialTheme.typography.bodySmall, color = SWarn)
    }
}

@Composable
private fun ProblemsCard(problems: List<PerfEntry>) {
    Card("Detected problems") {
        problems.take(8).forEach { entry ->
            Text("${date(entry.timestamp)} · ${entry.model.take(24)}: ${entry.error.ifBlank { entry.problem }}", style = MaterialTheme.typography.bodySmall, color = SWarn)
        }
    }
}

@Composable
private fun DiagnosticsCard(vm: WorkbenchViewModel) {
    Card("Diagnostics") {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text(vm.diagSessionInfo(), style = MaterialTheme.typography.bodySmall, color = Color.LightGray, modifier = Modifier.weight(1f))
            Text("App ${vm.appVersion()}", style = MaterialTheme.typography.labelSmall, color = Color.LightGray)
        }
        vm.diagPreviousEnd()?.let { note ->
            Spacer(Modifier.height(8.dp))
            SqlBox("⚠ $note", Color(0xFFFFD9A0))
        }
        vm.diagLastCrash()?.let { crash ->
            Spacer(Modifier.height(8.dp))
            SqlBox("LAST SESSION CRASHED\n$crash", SWarn)
        }
        Spacer(Modifier.height(10.dp))
        Text("RECENT EVENTS (latest 40)", style = MaterialTheme.typography.labelSmall, color = SIndigo)
        Surface(color = SConsole, shape = RoundedCornerShape(8.dp), modifier = Modifier.fillMaxWidth().padding(top = 4.dp)) {
            Column(Modifier.padding(8.dp)) {
                if (vm.diagRecent.isEmpty()) Text("No events yet.", style = MaterialTheme.typography.bodySmall, color = Color.LightGray)
                vm.diagRecent.forEach { line ->
                    Text(line, style = MaterialTheme.typography.bodySmall, fontFamily = FontFamily.Monospace, color = SMuted, fontSize = 9.sp)
                }
            }
        }
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.padding(top = 10.dp)) {
            OutlinedButton(onClick = { vm.clearDiag() }) { Icon(Icons.Default.DeleteSweep, null, Modifier.size(16.dp)); Spacer(Modifier.width(6.dp)); Text("Clear events") }
        }
    }
}

@Composable
private fun DeviceCard(vm: WorkbenchViewModel, caps: EngineCapabilities) {
    Card("Device") {
        Text(vm.deviceSummary(), style = MaterialTheme.typography.bodyMedium)
        Text("Engine: Rust pocketinfer (CPU NEON/int8${if (caps.opencl != null) " + OpenCL" else ""}); NPU detected: ${caps.qnn.ifBlank { "no" }}", style = MaterialTheme.typography.bodySmall, color = Color.LightGray)
    }
}

@Composable
private fun HistoryRow(entry: PerfEntry) {
    Surface(color = SPanel, shape = RoundedCornerShape(12.dp), modifier = Modifier.fillMaxWidth()) {
        Column(Modifier.padding(14.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text(entry.model.take(30), style = MaterialTheme.typography.titleSmall, modifier = Modifier.weight(1f), maxLines = 1, overflow = TextOverflow.Ellipsis)
                Text(date(entry.timestamp), style = MaterialTheme.typography.labelSmall, color = Color.LightGray)
            }
            Spacer(Modifier.height(4.dp))
            Text(String.format(Locale.US, "%s · %d threads · gen %.1f tok/s (%d tok in %.1fs) · prefill %.0f tok/s", entry.backend, entry.threads, entry.genTps, entry.genTokens, entry.genMs / 1000.0, entry.prefillTps), style = MaterialTheme.typography.bodySmall)
            Text(String.format(Locale.US, "Load: %s · RAM free %d MB · stop: %s · %s", if (entry.cached) "model cached" else String.format("%.1fs", entry.loadMs / 1000.0), entry.freeRamMb, entry.stop, entry.reasoning), style = MaterialTheme.typography.bodySmall, color = Color.LightGray)
            if (entry.error.isNotBlank()) Text("⚠ ${entry.error}", style = MaterialTheme.typography.bodySmall, color = SWarn)
            else if (entry.problem.isNotBlank()) Text("⚠ ${entry.problem}", style = MaterialTheme.typography.bodySmall, color = SWarn)
        }
    }
}

@Composable
private fun SqlBox(text: String, color: Color) {
    Surface(color = Color(0xFF3A2430), shape = RoundedCornerShape(8.dp), modifier = Modifier.fillMaxWidth()) {
        Text(text, style = MaterialTheme.typography.bodySmall, fontFamily = FontFamily.Monospace, color = color, modifier = Modifier.padding(10.dp))
    }
}

private fun yes(v: Boolean) = if (v) "yes" else "no"

private fun mb(bytes: Long): String = when {
    bytes >= 1024L * 1024L * 1024L -> String.format(Locale.US, "%.2f GB", bytes / 1073741824.0)
    bytes >= 1024L * 1024L -> String.format(Locale.US, "%.0f MB", bytes / 1048576.0)
    else -> String.format(Locale.US, "%.0f KB", bytes / 1024.0)
}

private fun date(ts: Long) = SimpleDateFormat("MM-dd HH:mm", Locale.US).format(Date(ts))
