package com.pocketworkbench.app;

/**
 * Events pushed from the agent runtime to the UI process. Registered callbacks
 * are held weakly: the runtime must not keep the UI alive, and a dead callback
 * is dropped rather than retried forever.
 */
oneway interface IPocketAgentCallback {
    /** A JSONL batch of session events. Never larger than the drain size. */
    void onEvents(String jsonl);

    /** A runtime-level state change: model loading, ready, unloaded, failed. */
    void onServiceState(int code, String detailJson);
}
