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
native = (root / 'app/src/main/cpp/bridge.cpp').read_text()
bridge = (src / 'NativeEngine.kt').read_text()
for method in ('generate', 'stop', 'transcribe'):
    assert re.search(r'\bexternal fun '+method+r'\b', bridge)
    assert 'Java_com_pocketworkbench_app_NativeEngine_'+method in native
ui = (src / 'MainActivity.kt').read_text()
for label in ('Stop generation', 'Model library', 'Conversations', 'Workspace', 'Message or edit voice transcript'):
    assert label in ui
model = (src / 'Data.kt').read_text()
assert 'Content-Range' in model and 'outputStream(append)' in model
assert 'saveHistory' in model and 'GGUF' in model
vm = (src / 'WorkbenchViewModel.kt').read_text()
assert 'native.stop()' in vm and 'native.transcribe' in vm
print('Static project smoke checks passed; Android build/device tests not run.')
