#!/usr/bin/env python3
"""Small static project smoke check; Android Studio build and hardware QA remain required."""
from pathlib import Path
import re
import xml.etree.ElementTree as ET

root = Path(__file__).resolve().parent.parent
manifest = ET.parse(root / 'app/src/main/AndroidManifest.xml')
permissions = {x.attrib.get('{http://schemas.android.com/apk/res/android}name') for x in manifest.getroot().findall('uses-permission')}
assert {'android.permission.INTERNET', 'android.permission.RECORD_AUDIO'} <= permissions
src = root / 'app/src/main/java/com/pocketworkbench/app'

# Rust inference engine (LLM) + C++ speech bridge (whisper.cpp)
rust = (root / 'rust/pocketinfer/src/jni.rs').read_text()
engine = (src / 'NativeEngine.kt').read_text()
speech = (src / 'SpeechEngine.kt').read_text()
for method in ('generate', 'stop'):
    assert re.search(r'\bexternal fun ' + method + r'\b', engine)
    assert 'Java_com_pocketworkbench_app_NativeEngine_' + method in rust
assert re.search(r'\bexternal fun transcribe\b', speech)
assert 'Java_com_pocketworkbench_app_SpeechEngine_transcribe' in (root / 'app/src/main/cpp/speech_bridge.cpp').read_text()

ui = (src / 'MainActivity.kt').read_text()
for label in ('Stop generation', 'Model library', 'Conversations', 'Workspace', 'Message or edit voice transcript'):
    assert label in ui
model = (src / 'Data.kt').read_text()
assert 'Content-Range' in model and 'FileOutputStream(partial, append)' in model
assert 'saveHistory' in model and 'GGUF' in model
vm = (src / 'WorkbenchViewModel.kt').read_text()
assert 'native.stop()' in vm and 'speechEngine.transcribe' in vm
for new_file in ('PerfLog.kt', 'GitHubClient.kt', 'McpTools.kt'):
    assert (src / new_file).exists(), new_file
assert 'onStats' in engine and 'onStats' in rust, 'stats callback must exist on engine side'
assert 'device/code' in (src / 'GitHubClient.kt').read_text(), 'device flow endpoint missing'
assert 'systemPrompt' in (src / 'McpTools.kt').read_text() and 'dispatch_workflow' in (src / 'McpTools.kt').read_text()
assert 'McpTools.parse' in vm and 'McpTools.execute' in vm and 'perf.record' in vm
print('Static project smoke checks passed; Android build/device tests not run.')
