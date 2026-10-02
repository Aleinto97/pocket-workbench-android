package com.pocketworkbench.app;

import com.pocketworkbench.app.IPocketAgentCallback;

/**
 * Versioned control surface of the agent runtime, which lives in the
 * ":inference" process of this same APK.
 *
 * Everything is JSON text so the protocol version is one number and every
 * payload can grow without a signature change. Payloads stay small: model paths
 * and session ids go in, batches of events come out. Weights and long
 * transcripts are never sent over Binder — the runtime reads them from the same
 * filesDir, and the UI re-reads the session log.
 */
interface IPocketAgent {

    /** Protocol version this implementation speaks. */
    int protocolVersion();

    /** Protocol version, backend and tool catalogue as one JSON document. */
    String capabilities();

    /**
     * Opens (or reopens) a session. Safe to call again for the same id: the
     * durable log is the state, and a turn that was interrupted is reported
     * rather than resumed.
     */
    String openSession(String sessionId, String logPath, String workspaceRoot, String configJson);

    String closeSession(String sessionId);

    /** JSON array of the session ids this process currently holds. */
    String sessionIds();

    /**
     * Admits a message. Returns {"admitted":false} when the request id is
     * already in the log, so a retry after a crash cannot start a second turn.
     * Never blocks: running happens on pump().
     */
    String submit(String sessionId, String requestId, String text);

    /** Runs up to maxTurns queued turns. Blocks for the duration of a turn. */
    String pump(String sessionId, int maxTurns);

    /** Cancels inference and any running tool for this session. */
    String cancel(String sessionId);

    String loadModel(String sessionId, String modelPath, String backend,
                     int contextTokens, int threads, boolean useGpu);

    String unloadModel(String sessionId);

    /** Phase, running flag, pending count, last stop reason and model info. */
    String sessionState(String sessionId);

    /** A batch of events as JSONL, at most maxEvents records. */
    String drainEvents(int maxEvents);

    /** Transcript items projected from the durable log, as JSON. */
    String transcript(String logPath);

    /** Register/unregister a cross-process event callback. */
    void registerCallback(IPocketAgentCallback cb);
    void unregisterCallback(IPocketAgentCallback cb);
}
