package com.pocketworkbench.app

import android.content.Context
import android.security.keystore.KeyGenParameterSpec
import android.security.keystore.KeyProperties
import android.util.Base64
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.withContext
import okhttp3.FormBody
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import java.security.KeyStore
import java.util.concurrent.TimeUnit
import javax.crypto.Cipher
import javax.crypto.KeyGenerator
import javax.crypto.SecretKey
import javax.crypto.spec.GCMParameterSpec

/**
 * Stores the GitHub token in app-private SharedPreferences, encrypted at rest
 * with an AES/GCM key that lives inside the Android Keystore.
 */
object TokenVault {
    private const val ALIAS = "pocket_gh_key"
    private const val PREFS = "github_secure"

    private fun secretKey(): SecretKey {
        val store = KeyStore.getInstance("AndroidKeyStore").apply { load(null) }
        (store.getKey(ALIAS, null) as? SecretKey)?.let { return it }
        val generator = KeyGenerator.getInstance(KeyProperties.KEY_ALGORITHM_AES, "AndroidKeyStore")
        generator.init(KeyGenParameterSpec.Builder(ALIAS, KeyProperties.PURPOSE_ENCRYPT or KeyProperties.PURPOSE_DECRYPT)
            .setBlockModes(KeyProperties.BLOCK_MODE_GCM)
            .setEncryptionPaddings(KeyProperties.ENCRYPTION_PADDING_NONE)
            .setRandomizedEncryptionRequired(true)
            .build())
        return generator.generateKey()
    }

    fun save(context: Context, token: String) {
        val cipher = Cipher.getInstance("AES/GCM/NoPadding")
        cipher.init(Cipher.ENCRYPT_MODE, secretKey())
        val encrypted = cipher.doFinal(token.toByteArray(Charsets.UTF_8))
        val payload = Base64.encodeToString(cipher.iv, Base64.NO_WRAP) + ":" + Base64.encodeToString(encrypted, Base64.NO_WRAP)
        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit().putString("token", payload).apply()
    }

    fun load(context: Context): String? = try {
        val payload = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).getString("token", null) ?: return null
        val parts = payload.split(":", limit = 2)
        if (parts.size != 2) null else {
            val cipher = Cipher.getInstance("AES/GCM/NoPadding")
            cipher.init(Cipher.DECRYPT_MODE, secretKey(), GCMParameterSpec(128, Base64.decode(parts[0], Base64.NO_WRAP)))
            String(cipher.doFinal(Base64.decode(parts[1], Base64.NO_WRAP)), Charsets.UTF_8)
        }
    } catch (_: Exception) { null }

    fun clear(context: Context) {
        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit().remove("token").apply()
    }
}

data class GhResult(val code: Int, val body: String, val scopes: String) {
    val ok: Boolean get() = code in 200..299
    fun json(): JSONObject = JSONObject(body)
}

data class DeviceCode(
    val deviceCode: String, val userCode: String, val verificationUri: String,
    val interval: Int, val expiresIn: Int
)

class GitHubClient(private val context: Context) {
    private val config = context.getSharedPreferences("github_config", Context.MODE_PRIVATE)
    private val client = OkHttpClient.Builder()
        .connectTimeout(20, TimeUnit.SECONDS).readTimeout(40, TimeUnit.SECONDS).build()

    var clientId: String
        get() = config.getString("oauth_client_id", "") ?: ""
        set(value) = config.edit().putString("oauth_client_id", value.trim()).apply()

    fun token(): String? = TokenVault.load(context)
    fun saveToken(token: String) = TokenVault.save(context, token)
    fun clearToken() = TokenVault.clear(context)

    suspend fun deviceCodeStart(): DeviceCode = withContext(Dispatchers.IO) {
        val form = FormBody.Builder().add("client_id", clientId).add("scope", "repo workflow").build()
        val request = Request.Builder()
            .url("https://github.com/login/device/code")
            .header("Accept", "application/json")
            .header("User-Agent", "PocketWorkbench/0.2")
            .post(form).build()
        client.newCall(request).execute().use { response ->
            val text = response.body?.string() ?: throw IllegalStateException("Empty response from GitHub")
            if (!response.isSuccessful) throw IllegalStateException("GitHub returned HTTP ${response.code}: ${text.take(200)}")
            val json = JSONObject(text)
            DeviceCode(
                deviceCode = json.getString("device_code"), userCode = json.getString("user_code"),
                verificationUri = json.optString("verification_uri", "https://github.com/login/device"),
                interval = json.optInt("interval", 5), expiresIn = json.optInt("expires_in", 900)
            )
        }
    }

    suspend fun pollForToken(device: DeviceCode, onNote: (String) -> Unit): String = withContext(Dispatchers.IO) {
        val deadline = System.currentTimeMillis() + device.expiresIn * 1000L
        var interval = device.interval.toLong()
        while (System.currentTimeMillis() < deadline) {
            delay(interval * 1000)
            val form = FormBody.Builder()
                .add("client_id", clientId)
                .add("device_code", device.deviceCode)
                .add("grant_type", "urn:ietf:params:oauth:grant-type:device_code")
                .build()
            val request = Request.Builder()
                .url("https://github.com/login/oauth/access_token")
                .header("Accept", "application/json")
                .header("User-Agent", "PocketWorkbench/0.2")
                .post(form).build()
            val json = client.newCall(request).execute().use { JSONObject(it.body?.string() ?: "{}") }
            val token = json.optString("access_token", "")
            if (token.isNotBlank()) return@withContext token
            when (json.optString("error")) {
                "authorization_pending" -> onNote("Waiting for authorization in the browser…")
                "slow_down" -> interval += 5
                "expired_token" -> throw IllegalStateException("Device code expired; start sign-in again")
                "access_denied" -> throw IllegalStateException("Sign-in was denied in the browser")
                else -> if (json.has("error")) throw IllegalStateException("GitHub: ${json.optString("error_description", "unknown error")}")
            }
        }
        throw IllegalStateException("Sign-in timed out; try again")
    }

    suspend fun rest(method: String, path: String, body: JSONObject? = null): GhResult = withContext(Dispatchers.IO) {
        val builder = Request.Builder()
            .url("https://api.github.com$path")
            .header("Accept", "application/vnd.github+json")
            .header("User-Agent", "PocketWorkbench/0.2")
            .header("X-GitHub-Api-Version", "2022-11-28")
        token()?.let { builder.header("Authorization", "Bearer $it") }
        when (method.uppercase()) {
            "GET" -> builder.get()
            "POST" -> builder.post((body?.toString() ?: "{}").toRequestBody("application/json".toMediaType()))
            "PUT" -> builder.put((body?.toString() ?: "{}").toRequestBody("application/json".toMediaType()))
            "DELETE" -> builder.delete()
            else -> throw IllegalArgumentException("Unsupported method $method")
        }
        client.newCall(builder.build()).execute().use { response ->
            val text = response.body?.string() ?: ""
            GhResult(response.code, text, response.header("X-OAuth-Scopes") ?: "")
        }
    }

    suspend fun rawGet(url: String, maxBytes: Int): String = withContext(Dispatchers.IO) {
        val builder = Request.Builder().url(url).header("User-Agent", "PocketWorkbench/0.2")
        token()?.let { builder.header("Authorization", "Bearer $it") }
        client.newCall(builder.build()).execute().use { response ->
            if (!response.isSuccessful) throw IllegalStateException("HTTP ${response.code}")
            val stream = response.body?.byteStream() ?: throw IllegalStateException("Empty body")
            val buffer = ByteArray(maxBytes)
            var offset = 0
            while (offset < maxBytes) {
                val n = stream.read(buffer, offset, maxBytes - offset)
                if (n < 0) break
                offset += n
            }
            String(buffer, 0, offset, Charsets.UTF_8)
        }
    }

    suspend fun fetchUser(): Pair<String, String> {
        val result = rest("GET", "/user")
        if (!result.ok) throw IllegalStateException("GitHub HTTP ${result.code}: cannot load profile")
        return result.json().optString("login", "unknown") to result.scopes
    }
}
