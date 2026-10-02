#!/usr/bin/env python3
"""Small static project smoke check; Android Studio build and hardware QA remain required."""
from pathlib import Path
import re
import xml.etree.ElementTree as ET

root = Path(__file__).resolve().parent.parent
manifest = ET.parse(root / 'app/src/main/AndroidManifest.xml')
permissions = {x.attrib.get('{http://schemas.android.com/apk/res/android}name') for x in manifest.getroot().findall('uses-permission')}
assert {'android.permission.INTERNET', 'android.permission.RECORD_AUDIO'} <= permissions
# The app must not reference a missing Application subclass (startup crash).
app = manifest.getroot().find('application')
assert app is not None
assert 'android:name' not in app.attrib or (root / 'app/src/main/java/com/pocketworkbench/app' / (app.attrib['{http://schemas.android.com/apk/res/android}name'].lstrip('.') + '.kt')).exists(), 'manifest application class missing'
# The agent runtime must live in its own process.
services = manifest.getroot().findall('.//{http://schemas.android.com/apk/res/android}service')
# fallback: plain tag search
services = manifest.getroot().iter('service')
found = False
for s in services:
    name = s.attrib.get('{http://schemas.android.com/apk/res/android}name', '')
    proc = s.attrib.get('{http://schemas.android.com/apk/res/android}process', '')
    if 'AgentService' in name:
        found = True
        assert proc == ':inference', 'AgentService must run in :inference'
assert found, 'AgentService missing from manifest'
src = root / 'app/src/main/java/com/pocketworkbench/app'

# Rust agent runtime JNI surface (replaces the old NativeEngine one-shot API)
rust = (root / 'rust/pocketinfer/src/agent_jni.rs').read_text()
runtime = (src / 'AgentRuntime.kt').read_text()
for method in ('nativeProtocolVersion', 'nativeOpenSession', 'nativeSubmit', 'nativePump', 'nativeCancel', 'nativeLoadModel', 'nativeDrainEvents', 'nativeTranscript'):
    assert re.search(r'\bexternal fun ' + method + r'\b', runtime), method
    assert ('Java_com_pocketworkbench_app_AgentRuntime_' + method) in rust, method
# Old one-shot engine surface must be gone from Kotlin (stale .so check is done via gradle inputs).
assert not (src / 'NativeEngine.kt').exists(), 'NativeEngine.kt should be removed'

# AIDL is the cross-process contract; callbacks must be registered through it,
# never by casting to a concrete binder class.
aidl = (root / 'app/src/main/aidl/com/pocketworkbench/app/IPocketAgent.aidl').read_text()
assert 'registerCallback' in aidl and 'unregisterCallback' in aidl, 'callbacks must be in AIDL'
client = (src / 'AgentClient.kt').read_text()
assert 'asInterface' in client and 'registerCallback' in client
assert 'as? AgentService.LocalBinder' not in client, 'cross-process cast forbidden'
assert 'Dispatchers.IO' in client, 'binder calls must leave the main thread'

# Speech bridge still present (optional, kept).
speech = (src / 'SpeechEngine.kt').read_text()
assert re.search(r'\bexternal fun transcribe\b', speech)
assert 'Java_com_pocketworkbench_app_SpeechEngine_transcribe' in (root / 'app/src/main/cpp/speech_bridge.cpp').read_text()

# Store layout + model hub
store = (src / 'WorkbenchStore.kt').read_text()
assert 'models/' in store or 'models' in store
hub = (src / 'ModelHub.kt').read_text()
assert 'Content-Range' in hub or 'Range' in hub
print('Static project smoke checks passed; Android build/device tests not run.')
