package com.pocketworkbench.app

import org.json.JSONObject

data class EngineCapabilities(
    val version: String,
    val abi: String,
    val busy: Boolean,
    val neon: Boolean,
    val dotprod: Boolean,
    val opencl: String?,
    val qnn: String,
    val loadError: String?
)

data class QuantSlice(val name: String, val bytes: Long, val tensors: Int)

data class ModelInfo(
    val file: String,
    val fileBytes: Long,
    val arch: String,
    val layers: Int,
    val hidden: Int,
    val ff: Int,
    val heads: Int,
    val kvHeads: Int,
    val headDim: Int,
    val vocab: Int,
    val ctx: Int,
    val threads: Int,
    val kvBytes: Long,
    val backend: String,
    val quant: List<QuantSlice>
)

data class DiagStep(val name: String, val ok: Boolean, val ms: Double, val error: String)

data class DiagReport(val ok: Boolean, val caps: EngineCapabilities, val steps: List<DiagStep>)

object EngineStatsParser {
    private fun caps(o: JSONObject): EngineCapabilities = EngineCapabilities(
        version = o.optString("engine", "?"),
        abi = o.optString("abi", "?"),
        busy = o.optBoolean("busy", false),
        neon = o.optBoolean("neon", false),
        dotprod = o.optBoolean("dotprod", false),
        opencl = if (o.isNull("opencl")) null else o.optString("opencl", null),
        qnn = o.optString("qnn", ""),
        loadError = if (o.optBoolean("loaded", false)) null else o.optString("error", "native library not loaded")
    )

    fun parseInfo(json: String): Pair<EngineCapabilities, ModelInfo?> {
        val o = JSONObject(json)
        val c = caps(o)
        val m = o.optJSONObject("model") ?: return c to null
        val quant = mutableListOf<QuantSlice>()
        m.optJSONArray("quant")?.let { arr ->
            for (i in 0 until arr.length()) {
                val e = arr.optJSONObject(i) ?: continue
                quant.add(QuantSlice(e.optString("name", "?"), e.optLong("bytes", 0), e.optInt("tensors", 0)))
            }
        }
        val info = ModelInfo(
            file = m.optString("file", "?"),
            fileBytes = m.optLong("file_bytes", 0),
            arch = m.optString("arch", "?"),
            layers = m.optInt("layers", 0),
            hidden = m.optInt("hidden", 0),
            ff = m.optInt("ff", 0),
            heads = m.optInt("heads", 0),
            kvHeads = m.optInt("kv_heads", 0),
            headDim = m.optInt("head_dim", 0),
            vocab = m.optInt("vocab", 0),
            ctx = m.optInt("ctx", 0),
            threads = m.optInt("threads", 0),
            kvBytes = m.optLong("kv_bytes", 0),
            backend = m.optString("backend", "?"),
            quant = quant
        )
        return c to info
    }

    fun parseDiag(json: String): DiagReport {
        val o = JSONObject(json)
        val steps = mutableListOf<DiagStep>()
        o.optJSONArray("steps")?.let { arr ->
            for (i in 0 until arr.length()) {
                val e = arr.optJSONObject(i) ?: continue
                steps.add(
                    DiagStep(
                        name = e.optString("name", "?"),
                        ok = e.optBoolean("ok", false),
                        ms = e.optDouble("ms", 0.0),
                        error = e.optString("error", "")
                    )
                )
            }
        }
        val co = o.optJSONObject("caps") ?: JSONObject()
        val c = EngineCapabilities(
            version = co.optString("engine", "?"),
            abi = "aarch64",
            busy = false,
            neon = true,
            dotprod = co.optBoolean("dotprod", false),
            opencl = if (co.isNull("opencl")) null else co.optString("opencl", null),
            qnn = "",
            loadError = null
        )
        return DiagReport(o.optBoolean("ok", false), c, steps)
    }
}
