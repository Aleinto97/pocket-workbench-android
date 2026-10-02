# Pocket Workbench: piano per l’app agentica nativa

## 1. Obiettivo e decisioni

Realizzare una sola applicazione Android che abbia una chat originale, strumenti agentici reali, inferenza locale e gestione dei progetti. Il riferimento concettuale è DeepSeek Harness, ma il prodotto finale usa il nostro ciclo agente e la nostra interfaccia.

Decisioni concordate:

- **Kotlin + Compose** per chat, animazioni, navigazione, file, modelli e impostazioni.
- **Rust** per orchestrazione agentica, contratti degli strumenti, gestione del contesto e sessioni.
- **C++ con llama.cpp/GenieX** per il backend di inferenza Hexagon, dietro una piccola interfaccia sostituibile.
- Un solo APK, senza dipendere da Termux, NPU Bench o un server esterno per il funzionamento locale.
- Nessun passaggio manuale con porte HTTP, cookie o token di accesso tra chat e agente. La comunicazione interna passa attraverso JNI e, quando serve isolare l’inferenza, IPC Android.
- Pesi GGUF separati dall’APK, importabili o scaricabili dall’app. Il runtime d’inferenza è incluso nell’APK.
- UI ispirata alla pulizia e alle animazioni apprezzate in Grok, con identità propria: orbita animata, risposta progressiva e attività degli strumenti espandibili.

Non riscrivere da zero tokenizzazione, quantizzazione e kernel NPU se una libreria verificata li fornisce già. Rendere nostre soprattutto le parti che determinano l’esperienza: ciclo agente, strumenti, sessioni e chat.

## 2. Contesto del progetto

- Repository di lavoro: `/root/pocket-workbench-android`.
- Applicazione principale: `com.pocketworkbench.app`.
- Modulo sperimentale: `npubench`, utile per benchmark; non deve essere necessario per usare l’app principale.
- Dispositivo di riferimento: HONOR YLE-W09, SoC riportato SM8845P, Android 16, circa 16 GB RAM.
- ABI attualmente utilizzata: `arm64-v8a`.
- SDK di compilazione/target attuale: 35; minSdk: 29.
- Toolchain: Gradle 8.11.1, NDK 27.2.12479018, CMake 3.22.1.
- Il repository contiene modifiche locali precedenti: conservarle e distinguere gli interventi nuovi dai lavori già presenti. Non fare commit o push senza richiesta.
- La porta di ADB wireless cambia. Non codificare un indirizzo del tablet nel prodotto o negli script definitivi.

## 3. Cosa è stato verificato e cosa no

### Inferenza NPU

Il runtime prebuilt GenieX/llama.cpp raggiunge Hexagon tramite ggml-hexagon/FastRPC e lo skel `libggml-htp-v81.so`.

Modello locale di riferimento:

- Repository: `empero-ai/Qwen3.8-4B-Distill-GGUF`.
- File sorgente: `Qwen3.8-4B-Q4_K_M.gguf`.
- Dimensione osservata: 2.783.446.304 byte, circa 2,59 GiB.
- SHA-256 registrato: `dec96e8cf2e11b613bb46513dec485377f9ca5a351e71712ee0e244f287c6790`.
- Nome usato negli esperimenti: `qwen38-4b-distill-q4km.gguf`.
- Posizione di staging utilizzata: `/data/local/tmp/npb/qwen38-4b-distill-q4km.gguf`.

Misure precedenti, dipendenti da prompt, temperatura e configurazione:

| Misura | NPU | CPU |
|---|---:|---:|
| Prefill, benchmark 512/32 | circa 374,8 tok/s | circa 64,5 tok/s |
| Decode | circa 14,9 tok/s | circa 6,4 tok/s |

Allocazioni riportate: circa 2814 MiB di pesi su HTP0 e 497 MiB su CPU, oltre a cache e buffer di lavoro. Una breve prova ha prodotto JSON con una chiamata `multiply` e argomenti 17 e 23. Questo dimostra generazione di un formato tool-call su quel prompt, **non** affidabilità generale di un agente.

Il 27B GSQ-RCO IQ2_XS testato non è la scelta di default: quasi 8 GiB rimanevano su CPU e il decode osservato con percorso NPU era inferiore al CPU. La quantizzazione Q4_K_M può essere accelerata: non limitarne il supporto teorico ai soli Q4_0. Verificare sempre allocazioni e prestazioni reali.

### Runner attuale

Il runner basato su `geniex-bench` avvia un processo per richiesta. Alcune run shell brevi complete sono durate circa 7–15 secondi, ma il servizio Android ha mostrato anche latenze di minuti e blocchi. Non considerare questo percorso affidabile o pronto solo perché il servizio `/health` risponde.

Il benchmark in modalità `--accuracy --prompt-file` applica il chat template e stampa `[gen ]`. Parsare quelle righe dopo l’uscita non è vero streaming dei token. `--accuracy` forza un’unica run; `--system-prompt` è supportato con questa modalità.

**Da verificare prima di prometterli:** backend residente, callback token end-to-end, stop tempestivo, gestione degli errori FastRPC, riuso corretto della KV cache e più turni consecutivi con strumenti.

### Esperimento Harness incorporato

È stato realizzato un APK con Debian PRoot, Node 24 e `@deepseek-ai/dsh@0.2.0-rc.2`. Sono stati verificati shell Linux, I/O del workspace, PTY e apertura della UI originale. Il runtime compresso era circa 253 MiB; l’APK risultava nell’ordine delle centinaia di MiB.

Questo esperimento è una base di confronto, non l’architettura finale. Ha introdotto estrazione iniziale, gestione di un server locale, autenticazione via token/cookie e compatibilità WebView. L’utente ora richiede una soluzione nativa con chat propria.

## 4. Architettura proposta

```text
UI Kotlin / Compose
   │ invio, stop, scelta progetto; eventi e stato
   ▼
Runtime agente Rust
   ├── sessioni e registro eventi
   ├── contesto e ciclo dei turni
   ├── strumenti e contratti
   └── interfaccia di inferenza
           │
           ▼
Servizio d’inferenza interno allo stesso APK
   └── adattatore C++ → llama.cpp/GenieX → Hexagon o CPU
```

### Confini di responsabilità

- La UI non costruisce prompt, non esegue strumenti e non decide se un turno è terminato.
- Rust è l’autorità sullo stato del turno, sulle chiamate agli strumenti e sul registro della sessione.
- Il backend C++ gestisce modello, tokenizzazione/template, generazione, cache e cancellazione dell’inferenza.
- Kotlin gestisce lifecycle Android, import/download, document picker, notifiche e pubblicazione degli eventi nella UI.
- La chiamata a un backend alternativo deve mantenere invariato il contratto usato dall’agente.

Per l’inferenza residente preferire un servizio Android in processo separato, per esempio `android:process=":inference"`, con librerie C++ caricate tramite JNI. Lo stesso UID permette di leggere i modelli privati; il processo separato isola un eventuale crash nativo. AIDL/Binder o un canale interno esplicito trasporta richieste e batch di eventi. Non inviare i pesi o grandi cronologie come un unico payload Binder.

Il runner eseguibile da `nativeLibraryDir` resta un’alternativa sperimentale iniziale. La decisione JNI versus processo eseguibile va confermata con una prova minima di caricamento, streaming e stop sul tablet; non riscrivere l’intera app prima di questa prova.

## 5. Interfaccia d’inferenza

Definire un contratto piccolo e versionato, concettualmente:

```text
load(model_path, backend, context_size, options)
generate(request_id, conversation, tool_schemas, generation_options)
cancel(request_id)
unload()
capabilities()
```

Eventi richiesti:

- modello in caricamento/pronto;
- delta di testo;
- delta di reasoning, soltanto quando effettivamente fornito dal modello;
- chiamata strumento strutturata;
- utilizzo token e timing;
- fine generazione con motivo: EOS, limite token, annullamento, errore;
- errore classificato e backend realmente impiegato.

Non equiparare fine generazione a fine turno: se il modello richiede uno strumento, il ciclo agente continua.

Il backend residente deve caricare i pesi una volta, mantenere il modello fino a cambio esplicito o necessità di memoria e verificare il riuso del prefisso. La KV cache non va riutilizzata dopo modifiche incompatibili a prompt/template/backend. Misurare il guadagno: un processo residente non garantisce automaticamente cache corretta.

Ispezionare le firme e gli header della versione effettiva di GenieX prima di usare la sua API C. Non dedurre ABI, proprietà dei buffer o thread-safety dai soli simboli esportati. Valutare l’API diretta di llama.cpp se offre controllo più chiaro su contesto e cancellazione.

## 6. Modelli, memoria e workspace

### Tre concetti distinti

1. **Pesi GGUF:** file nell’archivio dei modelli dell’app.
2. **Modello caricato:** pesi/cache/buffer tenuti in memoria dal backend.
3. **Workspace:** cartella di progetto su cui lavora l’agente.

Il workspace non contiene necessariamente i pesi. Un modello può servire molti progetti e non deve finire automaticamente negli ZIP dei progetti.

Layout previsto, relativo alle directory ottenute da Android:

```text
filesDir/
  models/                  # GGUF importati o scaricati
  workspaces/<project-id>/ # file del progetto
  sessions/<session-id>/   # metadati, eventi e checkpoint
  settings/               # configurazione
  logs/                   # diagnostica limitata e ruotata
```

- Usare `Context.filesDir` e percorsi ricavati dall’app; non codificare il package path.
- `/data/local/tmp/npb` è staging per sviluppo, non un requisito di utilizzo.
- Nell’APK unico il backend può leggere `filesDir/models`: il problema `noexec` riguarda eseguibili/librerie, non la lettura di un GGUF.
- Import tramite Android document picker, con progresso, gestione spazio e verifica GGUF. Download riprendibile con `.part` e checksum se disponibile.
- Associare ogni sessione a un workspace esplicito.
- Mostrare subito l’interfaccia all’avvio e lo stato reale del modello. Non promettere inferenza istantanea al cold start.
- Rilasciare il modello su pressione di memoria secondo una politica esplicita. Non tenere contemporaneamente CPU e NPU duplicati senza verificarne il costo.

## 7. Nucleo agente: idee di Harness da adottare

### 7.1 Un ciclo agente unico

```text
messaggio utente
 → preparazione del contesto
 → inferenza
 → risposta finale, oppure chiamata strumento
 → validazione ed esecuzione dello strumento
 → inserimento del risultato nella conversazione
 → nuova inferenza
```

Limiti espliciti: passi per turno, tempo del turno, output dei tool, dimensione del contesto e numero di errori ripetuti. I limiti devono essere visibili come motivi di arresto, non risposte finali inventate.

Prima versione: un turno attivo per sessione, strumenti sequenziali e coda dei messaggi successivi. Lo *steering* del turno corrente viene aggiunto dopo aver definito bene cancellazione e ordinamento.

### 7.2 Registro degli strumenti e contratti

Ogni strumento dichiara:

- nome stabile e versione;
- descrizione per il modello;
- schema dei parametri supportato;
- limiti e capacità necessarie;
- esecutore e risultato strutturato.

Validare nomi, campi obbligatori, tipi, dimensioni e percorsi prima dell’esecuzione. Conservare la correlazione tra ID della chiamata e risultato. Un risultato non deve essere inventato dal modello o ricostruito dalla UI.

Il formato tool-call va scelto in base al chat template realmente supportato dal modello. Preferire generazione vincolata quando il backend lo permette e verificarne la qualità; non presumere che una singola istruzione di prompt garantisca JSON corretto. Prevedere recupero limitato dai formati malformati.

### 7.3 Strumenti iniziali

- lista file e cartelle;
- lettura con finestre e limiti;
- ricerca testuale;
- scrittura di nuovi file;
- edit/diff con verifica della versione letta per evitare modifiche su contenuti cambiati;
- comando shell Android con working directory, timeout e output limitato.

Non introdurre subito subagenti, plugin dinamici e scheduler. GitHub viene dopo la stabilizzazione del percorso locale.

**Importante:** impostare il `cwd` al workspace non crea una sandbox. Lettura/scrittura richiedono risoluzione canonica e controllo dei symlink; la shell può comunque raggiungere altre risorse accessibili all’UID. Definire esplicitamente capacità e limiti. Non presentare la shell Android come Linux Debian completo.

L’esecuzione di compilatori o tool Linux mancanti richiede un modulo aggiuntivo futuro. Se necessario, un ambiente Linux opzionale può essere valutato separatamente; non deve tornare a essere una dipendenza del ciclo agente di base.

### 7.4 Registro della sessione

Sessioni JSONL append-only con versione dello schema, numeri di sequenza monotoni e ID di turni, richieste e tool.

Eventi minimi:

- messaggio utente accettato;
- avvio turno e avvio passo;
- risposta dell’assistente e motivo di fine;
- chiamata/risultato strumento;
- errore o annullamento;
- utilizzo token e backend;
- checkpoint/compattazione;
- chiusura turno.

Le risposte complete e i risultati sono durabili. I delta temporanei possono essere raggruppati per non fare una scrittura sincrona per token. Gestire una riga finale tronca dopo un crash e rendere idempotente l’ammissione di una richiesta già accettata.

La UI deriva il transcript dal registro; non mantiene una seconda storia incompatibile. Nessun log diagnostico deve esporre chiavi API o registrare inutilmente contenuti completi.

### 7.5 Contesto e compattazione

- Contare i token con il tokenizer effettivo.
- Conservare istruzioni, messaggi recenti e coppie chiamata/risultato degli strumenti.
- Gestire output lunghi mediante estratti e riferimenti ai file.
- Compattare la parte vecchia con un checkpoint registrato; lasciare il log originale consultabile.
- Non spezzare coppie tool-call/tool-result o troncare silenziosamente istruzioni e JSON.
- Non inviare reasoning storico come nuove istruzioni per impostazione predefinita.

### 7.6 Stop, errori e ripresa

Lo Stop annulla inferenza e strumenti attivi; verifica sul device quanto rapidamente arriva all’effetto reale. Per la shell terminare l’albero dei processi, non soltanto il processo padre.

Dopo un crash non rieseguire automaticamente una scrittura o un comando dagli effetti sconosciuti. Mostrare lo stato interrotto, ripristinare i dati durabili e lasciare all’utente la ripresa del turno.

## 8. Chat e animazioni

Schermata principale nativa con:

- composer accessibile, invio e Stop;
- streaming di testo reale, senza effetti di digitazione che alterano o fingono la generazione;
- orbita animata con stati: inattivo, caricamento, generazione, esecuzione tool, errore;
- timeline degli strumenti con parametri, esito e output espandibili;
- reasoning espandibile quando presente, senza generare una spiegazione fittizia dei pensieri;
- Markdown, blocchi di codice, copia messaggi e azioni sui file;
- cronologia delle sessioni e selezione del progetto;
- pagine modelli, file, statistiche e impostazioni.

Collegare le animazioni agli eventi reali. Fermare o ridurre le animazioni quando l’app non è visibile e rispettare le preferenze di riduzione del movimento.

Lo scroll segue la risposta se l’utente è in fondo; non riportarlo forzatamente in fondo mentre legge la cronologia. Batch dei delta prima di aggiornare Compose, per evitare una ricomposizione costosa per ogni token.

## 9. Vincoli nativi emersi

### Esecuzione e librerie

Sul tablet l’esecuzione da directory dati dell’app ha incontrato restrizioni. Il benchmark è stato impacchettato come `libgeniexbench.so` per essere estratto in `nativeLibraryDir`, eseguibile dal package manager. Questa soluzione del runner non è un motivo per conservare HTTP o PRoot nel prodotto.

Whisper e GenieX contengono librerie ggml con gli stessi nomi/SONAME. Non risolvere con `pickFirst` arbitrario: ABI differenti possono rompersi a runtime. Lo script sperimentale `scripts/package-geniex-runtime.py` rinomina le copie GenieX e riscrive `DT_NEEDED`; verificarne anche discovery dinamico, allineamento e funzionamento reale. Un processo separato non risolve da solo la collisione dei file dentro l’APK.

Lo skel DSP `libggml-htp-v81.so` usa un URI hardcoded: conservarne il nome richiesto. Le dipendenze dello skel appartengono all’ambiente DSP e non vanno confuse con le librerie Android dell’app.

### FastRPC e OpenCL

Il plugin imposta `ADSP_LIBRARY_PATH` da `GENIEX_PLUGIN_PATH`. Il runner crea link al plugin e allo skel. La versione prebuilt richiede accesso al loader vendor OpenCL anche quando si usa HTP; il manifest di npubench contiene `uses-native-library` per `libOpenCL.so`. Controllare questo requisito anche nel modulo principale.

Una sessione HTP aperta o un numero di layer offload non dimostra da solo che tutti i pesi sono accelerati.

### Teardown

È stato osservato un crash nel deinit del runtime prebuilt. `npubench/src/main/c/bench_exit.c` fornisce uno shim LD_PRELOAD che termina il runner dopo il report. È un workaround per un processo one-shot, non una politica definitiva di gestione delle risorse per JNI residente. Verificare teardown della versione usata e, se necessario, isolare/terminare il processo d’inferenza in modo controllato.

### Distribuzione e licenze

I binari GenieX sono attualmente recuperati tramite `scripts/fetch-geniex.sh` da un dataset privato. La disponibilità sul computer di sviluppo non prova che possano essere redistribuiti senza ulteriori verifiche. Registrare versioni, origini, checksum, licenze e avvisi dei componenti effettivamente inclusi. Non dichiarare che l’intero runtime ha una sola licenza o che rimuovere PRoot elimina automaticamente ogni obbligo.

## 10. Stato del codice da cui partire

### Parti esistenti utili

- `rust/pocketinfer/`: loader GGUF, tokenizer, inferenza CPU e OpenCL sperimentale, JNI e diagnostica.
- `app/.../Data.kt`: archivio modelli e conversazioni; i modelli importati stanno in `filesDir/models`.
- `app/.../WorkbenchViewModel.kt`: vecchio ciclo agente a 12 passi e gestione modelli.
- `app/.../WorkspaceTools.kt`: strumenti workspace e shell Android.
- `app/.../McpTools.kt` e `GitHubClient.kt`: strumenti GitHub e integrazione esistente.
- `app/.../PerfLog.kt`, `EngineStats.kt`, `Diagnostics.kt`: diagnostica e misure.
- `npubench/`: benchmark da mantenere per confronti indipendenti.

### Esperimenti recenti da sostituire gradualmente

- `HarnessRuntimeService.kt`, `RootfsExtractor.kt`, `EmbeddedHarnessScreen.kt`: runtime Linux/Node e WebView incorporati.
- `HarnessAgentGateway.kt`: RPC HTTP con autenticazione del server Harness.
- `HarnessAgentTranscript.kt`: proiezione sperimentale dei record Harness.
- `AgentChatScreen.kt`: chat Compose animata; stile riutilizzabile, ma il ViewModel va collegato al nuovo runtime agente.
- `NpuModelService.kt`, `NpuBridge.kt`: endpoint HTTP modello e helper locale sperimentali.
- `scripts/package-harness-runtime.py`, `setup-deepseek-harness-termux.sh`, `start-deepseek-harness-termux.sh`: packaging/avvio del runtime precedente.

Non considerare questi esperimenti pronti end-to-end. Esempi di difetti già individuati o ancora da verificare:

- selezione del GGUF basata su “ultimo file modificato” può scegliere il 27B invece del modello selezionato;
- perdita delle precedenti chiamate tool durante la costruzione del prompt;
- SSE tool-call senza indice previsto dal protocollo;
- controllo `/health` che risponde prima di verificare effettiva disponibilità del modello;
- paging del gateway con cursore `-1`, che in Harness rappresenta un log vuoto, non “ultima pagina”;
- parser UI che presume un wrapper `message` anche dove il messaggio utente è direttamente in `data`;
- polling e cancellazione che possono aggiornare la UI con dati di una sessione già sostituita.

Il backend nativo deve eliminare questi problemi strutturali, non semplicemente nasconderne i messaggi.

## 11. Sequenza di implementazione

### Fase A — una fetta completa minima

1. Definire contratti del backend e degli eventi, con Rust come autorità del turno.
2. Collegare la chat nativa a un runtime Rust minimo.
3. Usare temporaneamente il percorso NPU verificato del runner soltanto come adattatore sperimentale.
4. Implementare un tool read/list e un tool write/edit nel workspace.
5. Registrare la sessione e proiettare messaggi/attività nella UI.

Accettazione: aprire un progetto, chiedere una modifica semplice, eseguire realmente lo strumento, vedere il file risultante e il transcript coerente. Nessuna dipendenza dall’APK npubench, da Termux o da Harness Node.

### Fase B — backend residente e streaming

1. Prova minima dell’API C++ realmente disponibile: load, callback token, stop, nuova richiesta.
2. Servizio d’inferenza interno isolato, protocollo versionato e riavvio dopo crash.
3. Modello selezionato da `filesDir/models`.
4. Streaming dei token e misure cold/warm.
5. Verifica del riuso del prefisso e invalidazione della cache.

Accettazione: almeno più turni consecutivi senza ricaricare i pesi; Stop efficace; nessun backend dichiarato NPU quando ha fatto fallback CPU; logits/output comparati quando si attiva il riuso della cache.

### Fase C — strumenti e persistenza affidabili

1. Contratti e validazione degli strumenti.
2. Ricerca, edit con versione letta, timeout e limiti output.
3. Coda dei messaggi; idempotenza delle richieste.
4. Ripristino di sessioni e handling del log tronco.
5. Politica esplicita per shell, percorsi e processi figli.

Accettazione: una piccola attività su file dall’inizio alla fine, con errori e annullamento verificati, senza duplicare comandi dopo un riavvio.

### Fase D — contesto e esperienza completa

1. Conteggio token effettivo, limiti, riduzione degli output e checkpoint.
2. Compattazione con conservazione delle coppie tool-call/risultato.
3. Cronologia, file, modelli e statistiche nella UI.
4. Animazioni collegate allo stato, accessibilità e comportamento tablet/finestre ridotte.
5. GitHub dopo stabilizzazione del percorso locale.

Accettazione: sessione lunga oltre la capacità del contesto, ancora continuabile con log consultabile e UI responsiva.

## 12. Verifica e consegna

- Test Rust mirati su ordinamento eventi, coda, cancellazione, contratti tool, contesto e recupero del log.
- Test della proiezione UI per messaggi, chiamate/risultati, errori e fine turno.
- Test device del backend: avvio freddo/caldo, streaming, Stop, memoria, termica, più richieste e crash del servizio.
- Confronti CPU/NPU a parità di modello, prompt, token, contesto e stato termico. Separare caricamento, prefill e decode.
- Test d’uso: crea e modifica un file, riapri la sessione, metti l’app in background e torna, prova una chiamata malformata e un comando lungo.
- Non confondere compilazione riuscita con funzionamento end-to-end.

Build standard: `gradle :app:assembleDebug` con la toolchain Android corretta. Sul computer ARM usato nella sessione i prebuilt x86_64 di CMake/NDK hanno dato errori di esecuzione. È stato usato un workaround locale con librerie speech estratte da un APK precedente e `-PusePrebuiltSpeech=true`; non assumerlo come build riproducibile per distribuzione. Anche `/tmp/opencode/aapt2` è un adattamento locale.

La build definitiva deve recuperare gli artefatti necessari con checksum, produrre un APK unico firmato come aggiornamento e documentare la toolchain. Preservare modelli, workspace e sessioni quando si installa l’aggiornamento.

L’utente ha chiesto anche di poter provare l’APK: dopo build e installazione, fornire solo istruzioni brevi di test e dichiarare precisamente quali percorsi sono stati verificati. Non disinstallare altre app o cancellare i dati dell’utente per “pulire” il sistema senza richiesta.

## 13. Risultato atteso

Un’app che apre immediatamente la propria interfaccia, rende visibile l’eventuale caricamento del modello e permette di lavorare su un progetto con un agente locale reale. L’esperienza e l’orchestrazione sono nostre; il lavoro di basso livello sui modelli resta affidato a componenti verificati. Nessuna promessa di inferenza istantanea o sandbox completa prima della relativa prova sul dispositivo.
