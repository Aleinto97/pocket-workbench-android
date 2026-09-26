package com.pocketworkbench.app

import android.util.Base64
import org.json.JSONArray
import org.json.JSONObject
import org.xmlpull.v1.XmlPullParser
import org.xmlpull.v1.XmlPullParserFactory
import java.io.StringReader
import java.net.URLEncoder

data class McpToolDef(
    val name: String,
    val signature: String,
    val description: String,
    val handler: suspend (JSONObject, GitHubClient) -> String
)

data class McpToolCall(val name: String, val arguments: JSONObject)

data class McpParseResult(val call: McpToolCall?, val attempted: Boolean, val raw: String)

object McpTools {
    private fun enc(value: String): String = URLEncoder.encode(value, "UTF-8").replace("+", "%20")
    private fun encPath(path: String): String = path.split('/').joinToString("/") { enc(it) }

    private fun clip(text: String, max: Int = 5500): String =
        if (text.length <= max) text else text.take(max) + "\n…[truncated]"

    private fun jsonArray(result: GhResult, what: String): JSONArray {
        if (!result.ok) throw IllegalStateException("GitHub HTTP ${result.code}: ${result.body.take(300)}")
        return JSONArray(result.body)
    }

    private fun jsonObject(result: GhResult, what: String): JSONObject {
        if (!result.ok) throw IllegalStateException("GitHub HTTP ${result.code}: ${result.body.take(300)}")
        return result.json()
    }

    val tools: List<McpToolDef> = listOf(
        McpToolDef("github_whoami", "github_whoami()",
            "current GitHub user login and token scopes") { _, gh ->
            val result = gh.rest("GET", "/user")
            if (!result.ok) throw IllegalStateException("GitHub HTTP ${result.code}")
            val user = result.json()
            JSONObject().put("login", user.optString("login")).put("scopes", result.scopes).toString()
        },
        McpToolDef("github_list_repos", "github_list_repos()",
            "list the user's repositories (name, private, updated, default branch)") { _, gh ->
            val result = gh.rest("GET", "/user/repos?sort=updated&per_page=20&affiliation=owner,collaborator")
            val out = JSONArray()
            jsonArray(result, "repos").let { list ->
                for (i in 0 until list.length()) {
                    val repo = list.getJSONObject(i)
                    out.put(JSONObject().put("full_name", repo.getString("full_name"))
                        .put("private", repo.optBoolean("private")).put("updated", repo.optString("updated_at"))
                        .put("default_branch", repo.optString("default_branch")))
                }
            }
            clip(out.toString())
        },
        McpToolDef("github_get_repo", "github_get_repo(owner,repo)",
            "repository metadata, including default branch and description") { args, gh ->
            clip(jsonObject(gh.rest("GET", "/repos/${enc(args.getString("owner"))}/${enc(args.getString("repo"))}"), "repo")
                .let { JSONObject().put("full_name", it.optString("full_name")).put("default_branch", it.optString("default_branch"))
                    .put("description", it.optString("description")).put("language", it.optString("language")).toString() })
        },
        McpToolDef("github_list_files", "github_list_files(owner,repo,ref?)",
            "list up to 200 file paths of the repository tree") { args, gh ->
            val owner = args.getString("owner"); val repo = args.getString("repo")
            var ref = args.optString("ref", "")
            if (ref.isBlank()) ref = jsonObject(gh.rest("GET", "/repos/${enc(owner)}/${enc(repo)}"), "repo").optString("default_branch", "main")
            val tree = jsonObject(gh.rest("GET", "/repos/${enc(owner)}/${enc(repo)}/git/trees/${enc(ref)}?recursive=1"), "tree")
            val paths = tree.optJSONArray("tree") ?: JSONArray()
            val out = JSONArray()
            for (i in 0 until minOf(paths.length(), 200)) {
                val item = paths.getJSONObject(i)
                out.put(item.getString("path") + if (item.optString("type") == "tree") "/" else "")
            }
            clip(out.toString())
        },
        McpToolDef("github_read_file", "github_read_file(owner,repo,path,ref?)",
            "read a text file from the repository (base64 decoded, truncated)") { args, gh ->
            val owner = args.getString("owner"); val repo = args.getString("repo"); val path = args.getString("path")
            var url = "/repos/${enc(owner)}/${enc(repo)}/contents/${encPath(path)}"
            args.optString("ref", "").takeIf { it.isNotBlank() }?.let { url += "?ref=${enc(it)}" }
            val file = jsonObject(gh.rest("GET", url), "file")
            if (file.optString("encoding") == "base64") {
                val decoded = String(Base64.decode(file.getString("content"), Base64.NO_WRAP), Charsets.UTF_8)
                JSONObject().put("path", file.optString("path")).put("content", clip(decoded, 4500)).toString()
            } else "File is too large or has no base64 content; size ${file.optLong("size", -1)} bytes"
        },
        McpToolDef("github_search_code", "github_search_code(query,repo?)",
            "search code on GitHub; repo filters with owner/name") { args, gh ->
            val query = buildString {
                append(args.getString("query"))
                args.optString("repo", "").takeIf { it.isNotBlank() }?.let { append(" repo:$it") }
            }
            val result = gh.rest("GET", "/search/code?q=${enc(query)}&per_page=8")
            if (!result.ok) throw IllegalStateException("GitHub HTTP ${result.code}: code search may need a public repo")
            val items = result.json().optJSONArray("items") ?: JSONArray()
            val out = JSONArray()
            for (i in 0 until items.length()) {
                val item = items.getJSONObject(i)
                out.put(JSONObject().put("repo", item.optJSONObject("repository")?.optString("full_name") ?: "")
                    .put("path", item.optString("path")))
            }
            clip(out.toString())
        },
        McpToolDef("github_list_issues", "github_list_issues(owner,repo,state?)",
            "list issues (and PRs sharing the numbering) with numbers and titles") { args, gh ->
            val state = args.optString("state", "open")
            val result = gh.rest("GET", "/repos/${enc(args.getString("owner"))}/${enc(args.getString("repo"))}/issues?state=${enc(state)}&per_page=15")
            val out = JSONArray()
            jsonArray(result, "issues").let { list ->
                for (i in 0 until list.length()) {
                    val issue = list.getJSONObject(i)
                    if (issue.has("pull_request")) continue
                    out.put(JSONObject().put("number", issue.optInt("number")).put("title", issue.optString("title")).put("state", issue.optString("state")))
                }
            }
            clip(out.toString())
        },
        McpToolDef("github_create_issue", "github_create_issue(owner,repo,title,body?)",
            "open a new issue") { args, gh ->
            val body = JSONObject().put("title", args.getString("title"))
            args.optString("body", "").takeIf { it.isNotBlank() }?.let { body.put("body", it) }
            val created = jsonObject(gh.rest("POST", "/repos/${enc(args.getString("owner"))}/${enc(args.getString("repo"))}/issues", body), "issue")
            JSONObject().put("number", created.optInt("number")).put("html_url", created.optString("html_url")).toString()
        },
        McpToolDef("github_add_issue_comment", "github_add_issue_comment(owner,repo,number,body)",
            "comment on an existing issue or pull request") { args, gh ->
            val created = jsonObject(gh.rest("POST",
                "/repos/${enc(args.getString("owner"))}/${enc(args.getString("repo"))}/issues/${args.getInt("number")}/comments",
                JSONObject().put("body", args.getString("body"))), "comment")
            JSONObject().put("html_url", created.optString("html_url")).toString()
        },
        McpToolDef("github_create_branch", "github_create_branch(owner,repo,ref,from_branch?)",
            "create a new branch; from_branch defaults to the default branch") { args, gh ->
            val owner = args.getString("owner"); val repo = args.getString("repo")
            var from = args.optString("from_branch", "")
            if (from.isBlank()) from = jsonObject(gh.rest("GET", "/repos/${enc(owner)}/${enc(repo)}"), "repo").optString("default_branch", "main")
            val source = jsonObject(gh.rest("GET", "/repos/${enc(owner)}/${enc(repo)}/git/ref/heads/${enc(from)}"), "ref")
            val created = jsonObject(gh.rest("POST", "/repos/${enc(owner)}/${enc(repo)}/git/refs",
                JSONObject().put("ref", "refs/heads/${args.getString("ref")}").put("sha", source.optJSONObject("object")?.optString("sha") ?: "")), "branch")
            JSONObject().put("ref", created.optString("ref")).toString()
        },
        McpToolDef("github_create_or_update_file", "github_create_or_update_file(owner,repo,path,content,message,branch?)",
            "commit a file (creates or updates it); branch optional") { args, gh ->
            val owner = args.getString("owner"); val repo = args.getString("repo"); val path = args.getString("path")
            var url = "/repos/${enc(owner)}/${enc(repo)}/contents/${encPath(path)}"
            args.optString("branch", "").takeIf { it.isNotBlank() }?.let { url += "?ref=${enc(it)}" }
            val get = gh.rest("GET", url)
            val payload = JSONObject()
                .put("message", args.getString("message"))
                .put("content", Base64.encodeToString(args.getString("content").toByteArray(Charsets.UTF_8), Base64.NO_WRAP))
            args.optString("branch", "").takeIf { it.isNotBlank() }?.let { payload.put("branch", it) }
            if (get.ok) get.json().optString("sha", "").takeIf { it.isNotBlank() }?.let { payload.put("sha", it) }
            val saved = jsonObject(gh.rest("PUT", url.substringBefore('?'), payload), "commit")
            JSONObject().put("commit_url", saved.optJSONObject("commit")?.optString("html_url") ?: "").toString()
        },
        McpToolDef("github_create_pull_request", "github_create_pull_request(owner,repo,title,head,base,body?)",
            "open a pull request; head is the source branch, base the target") { args, gh ->
            val payload = JSONObject().put("title", args.getString("title"))
                .put("head", args.getString("head")).put("base", args.getString("base"))
            args.optString("body", "").takeIf { it.isNotBlank() }?.let { payload.put("body", it) }
            val created = jsonObject(gh.rest("POST", "/repos/${enc(args.getString("owner"))}/${enc(args.getString("repo"))}/pulls", payload), "pull")
            JSONObject().put("number", created.optInt("number")).put("html_url", created.optString("html_url")).toString()
        },
        McpToolDef("github_list_workflows", "github_list_workflows(owner,repo)",
            "list GitHub Actions workflow files (id, name, path)") { args, gh ->
            val result = gh.rest("GET", "/repos/${enc(args.getString("owner"))}/${enc(args.getString("repo"))}/actions/workflows")
            val list = result.json().optJSONArray("workflows") ?: JSONArray()
            val out = JSONArray()
            for (i in 0 until list.length()) {
                val wf = list.getJSONObject(i)
                out.put(JSONObject().put("id", wf.optLong("id")).put("name", wf.optString("name")).put("path", wf.optString("path")))
            }
            clip(out.toString())
        },
        McpToolDef("github_list_workflow_runs", "github_list_workflow_runs(owner,repo)",
            "recent CI runs with id, status, conclusion and branch") { args, gh ->
            val result = gh.rest("GET", "/repos/${enc(args.getString("owner"))}/${enc(args.getString("repo"))}/actions/runs?per_page=8")
            val list = result.json().optJSONArray("workflow_runs") ?: JSONArray()
            val out = JSONArray()
            for (i in 0 until list.length()) {
                val run = list.getJSONObject(i)
                out.put(JSONObject().put("id", run.optLong("id")).put("name", run.optString("name"))
                    .put("status", run.optString("status")).put("conclusion", run.optString("conclusion"))
                    .put("branch", run.optString("head_branch")).put("created", run.optString("created_at")))
            }
            clip(out.toString())
        },
        McpToolDef("github_get_run_jobs", "github_get_run_jobs(owner,repo,run_id)",
            "jobs of a CI run with id, name, status and conclusion") { args, gh ->
            val result = gh.rest("GET", "/repos/${enc(args.getString("owner"))}/${enc(args.getString("repo"))}/actions/runs/${args.getLong("run_id")}/jobs")
            val list = result.json().optJSONArray("jobs") ?: JSONArray()
            val out = JSONArray()
            for (i in 0 until list.length()) {
                val job = list.getJSONObject(i)
                out.put(JSONObject().put("id", job.optLong("id")).put("name", job.optString("name"))
                    .put("status", job.optString("status")).put("conclusion", job.optString("conclusion")))
            }
            clip(out.toString())
        },
        McpToolDef("github_get_job_log", "github_get_job_log(owner,repo,job_id)",
            "tail of a CI job log, useful to diagnose build failures") { args, gh ->
            val log = gh.rawGet("https://api.github.com/repos/${enc(args.getString("owner"))}/${enc(args.getString("repo"))}/actions/jobs/${args.getLong("job_id")}/logs", 200_000)
            clip(if (log.length > 4500) log.takeLast(4500) else log, 5000)
        },
        McpToolDef("github_dispatch_workflow", "github_dispatch_workflow(owner,repo,workflow_id,ref)",
            "trigger a workflow by file name (e.g. android-build.yml) on a ref") { args, gh ->
            val result = gh.rest("POST",
                "/repos/${enc(args.getString("owner"))}/${enc(args.getString("repo"))}/actions/workflows/${enc(args.getString("workflow_id"))}/dispatches",
                JSONObject().put("ref", args.optString("ref", "main")))
            if (result.code == 204) "Workflow dispatch accepted; check github_list_workflow_runs shortly."
            else throw IllegalStateException("GitHub HTTP ${result.code}: ${result.body.take(300)}")
        }
    )

    private val byName = tools.associateBy { it.name }

    fun systemPrompt(): String = buildString {
        appendLine("You are Pocket Workbench, an assistant running fully offline on an Android tablet, with GitHub tools.")
        appendLine("To call a tool, output exactly one tag with JSON, then stop and wait:")
        appendLine("<tool>{\"name\":\"github_read_file\",\"arguments\":{\"owner\":\"me\",\"repo\":\"demo\",\"path\":\"README.md\"}}</tool>")
        appendLine("You may also use MiniCPM5's native XML form: <function name=\"github_read_file\"><param name=\"owner\">me</param><param name=\"repo\">demo</param><param name=\"path\">README.md</param></function>.")
        appendLine("After your call you will receive a message starting with [TOOL RESULT]. Never invent tool results.")
        appendLine("Call another tool, or reply with plain text when you have enough information (that ends the turn).")
        appendLine("Rules: one tool call per reply; owner and repo are separate fields; file paths are repo-relative; dates are ISO.")
        appendLine("Available tools:")
        tools.forEach { appendLine("- ${it.signature}: ${it.description}") }
    }

    fun parse(text: String): McpParseResult {
        // Ignore examples inside the model's reasoning; only the final answer
        // may call a tool. Retain the existing JSON protocol for other models.
        val answer = text.replace(Regex("<think>[\\s\\S]*?</think>", RegexOption.IGNORE_CASE), "")
        val jsonTag = Regex("<tool>([\\s\\S]*?)</tool>", RegexOption.IGNORE_CASE).findAll(answer).lastOrNull()
        val xmlTag = Regex("<function\\b[^>]*>[\\s\\S]*?</function>", RegexOption.IGNORE_CASE).findAll(answer).lastOrNull()
        if (xmlTag != null && (jsonTag == null || xmlTag.range.first > jsonTag.range.first)) {
            val raw = xmlTag.value
            return try {
                val parser = XmlPullParserFactory.newInstance().newPullParser()
                parser.setInput(StringReader(raw))
                var name = ""
                val arguments = JSONObject()
                while (parser.eventType != XmlPullParser.END_DOCUMENT) {
                    if (parser.eventType == XmlPullParser.START_TAG) when (parser.name) {
                        "function" -> name = parser.getAttributeValue(null, "name") ?: ""
                        "param" -> {
                            val key = parser.getAttributeValue(null, "name") ?: ""
                            if (key.isNotBlank()) arguments.put(key, parser.nextText())
                        }
                    }
                    parser.next()
                }
                if (name !in byName) McpParseResult(null, true, raw)
                else McpParseResult(McpToolCall(name, arguments), true, raw)
            } catch (_: Exception) { McpParseResult(null, true, raw) }
        }
        if (jsonTag == null) return McpParseResult(null, false, "")
        val raw = jsonTag.groupValues[1].trim()
        return try {
            val start = raw.indexOf('{'); val end = raw.lastIndexOf('}')
            if (start < 0 || end <= start) return McpParseResult(null, true, raw)
            val json = JSONObject(raw.substring(start, end + 1))
            val name = json.optString("name", "")
            if (name.isBlank() || !byName.containsKey(name)) return McpParseResult(null, true, raw)
            McpParseResult(McpToolCall(name, json.optJSONObject("arguments") ?: JSONObject()), true, raw)
        } catch (_: Exception) { McpParseResult(null, true, raw) }
    }

    suspend fun execute(call: McpToolCall, gh: GitHubClient): String {
        val def = byName[call.name] ?: return JSONObject().put("ok", false).put("error", "Unknown tool ${call.name}").toString()
        return try {
            JSONObject().put("ok", true).put("tool", call.name).put("result", def.handler(call.arguments, gh)).toString()
        } catch (e: Exception) {
            JSONObject().put("ok", false).put("tool", call.name).put("error", e.message ?: "failed").toString()
        }
    }
}
