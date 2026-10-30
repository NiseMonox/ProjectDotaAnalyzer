# 在 Windows 上继续开发

这份指南说明怎样在 Windows 10 / 11 上把项目跑起来并继续开发。项目本身的说明见
[`README.md`](../README.md)，全部配置项见 [`.env.example`](../.env.example)，Linux 服务器部署见
[`deploy/README.md`](../deploy/README.md)。

约定：

- 命令都是 PowerShell（系统自带的 Windows PowerShell 5.1 和 PowerShell 7 都可以），在仓库根目录执行。
- `<尖括号>` 是占位符，换成你自己的值。
- 标了“未在 Windows 上实测”的内容是按代码和官方文档写的，这台开发机没有 Windows；
  测试套件和冒烟测试由 GitHub Actions 的 `windows-latest` 任务实际运行（见[测试](#5-测试)）。

## 1. 前置软件

| 软件 | 用途 | 安装 | 检查版本 |
|---|---|---|---|
| Git for Windows | 版本管理；Claude Code 用它带的 Git Bash 执行命令 | `winget install --id Git.Git -e` | `git --version` |
| Python 3.12（3.10 至 3.13 都支持） | 后端 | `winget install --id Python.Python.3.12 -e` | `py -3.12 --version` |
| Temurin JDK 17 | 解析上传的录像；不装也能按比赛 ID 获取比赛 | `winget install --id EclipseAdoptium.Temurin.17.JDK -e` | `java -version` |
| Node.js LTS（可选） | 只有一个测试用到（`backend\tests\test_frontend_logic.py`），没有就跳过 | `winget install --id OpenJS.NodeJS.LTS -e` | `node --version` |

- 装完后重新打开 PowerShell，新的 `PATH` 才生效。
- **换行符**：仓库的 `.gitattributes` 规定文本文件在仓库里和工作区里都是 LF（`*.bat` 原样保留
  CRLF，jar、向量文件、图片等按二进制处理），所以安装 Git 时“换行符转换”选哪一项、
  `core.autocrlf` 是什么值都不影响，不需要改。
- **Python**：`winget` 装的是 python.org 的安装包，自带 `py` 启动器。Microsoft Store 版也能用，
  但没有 `py`，把下文的 `py -3.12` 换成 `python`。`run.ps1` 两种都能找到。
- **Java**：winget 的安装包会把 `java` 加入 `PATH`，默认不设置 `JAVA_HOME`，对本项目没有影响
  （查找顺序见[首次运行](#3-首次运行)）。
- **可选的 Python 包**：`backend\requirements-rag.txt`（`onnxruntime`、`tokenizers`）只在需要本机
  生成向量时安装，见[第 4 节](#4-不在仓库里的本地数据)。解压 Zstandard 录像用的 `zstandard`
  已经在 `backend\requirements.txt` 里，不用单独装。
- **执行策略**：Windows 默认不允许运行 `.ps1` 脚本（报错 “running scripts is disabled on this
  system”）。两种办法任选：每次用 `powershell -ExecutionPolicy Bypass -File run.ps1` 启动，不改任何
  系统设置；或者给当前用户放开一次，以后直接写 `.\run.ps1`：

  ```powershell
  Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
  ```

## 2. 获取代码

把仓库放在一个不太深、路径里只有英文字符的目录下，例如 `C:\dev\ProjectDotaAnalyzer`
（原因见[第 7 节](#7-与-macos--linux-的差异和已知限制)）。

从 GitHub 克隆：

```powershell
git clone https://github.com/<owner>/ProjectDotaAnalyzer.git
cd ProjectDotaAnalyzer
```

或者从网络共享（NAS）上的私有裸仓库克隆，两种写法任选：

```powershell
git clone //<NAS主机>/<共享名>/<路径>/ProjectDotaAnalyzer.git      # UNC 路径
git clone Z:/<路径>/ProjectDotaAnalyzer.git                        # 已映射成盘符 Z: 的共享
```

Git for Windows 无法确认网络共享上仓库的所有者，第一次访问会报
`fatal: detected dubious ownership in repository`。按报错里给出的命令把这个仓库加入信任列表
（只加这一个路径，不要用 `*`）：

```powershell
git config --global --add safe.directory '%(prefix)///<NAS主机>/<共享名>/<路径>/ProjectDotaAnalyzer.git'   # UNC
git config --global --add safe.directory 'Z:/<路径>/ProjectDotaAnalyzer.git'                              # 盘符
```

**同时使用两个远端。** 假设 `origin` 是 GitHub，再把 NAS 加为 `nas`：

```powershell
git remote add nas //<NAS主机>/<共享名>/<路径>/ProjectDotaAnalyzer.git
git remote -v
git fetch --all                 # 两边都取
git push origin <分支>
git push nas <分支>
```

想让一次 `git push` 同时推到两边，给 `origin` 配两个推送地址（第一条必须是 GitHub 自己的地址，
否则它会被第二条取代）：

```powershell
git remote set-url --add --push origin https://github.com/<owner>/ProjectDotaAnalyzer.git
git remote set-url --add --push origin //<NAS主机>/<共享名>/<路径>/ProjectDotaAnalyzer.git
```

确认两边一致（两条命令输出的提交号相同）：

```powershell
git ls-remote origin refs/heads/<分支>
git ls-remote nas refs/heads/<分支>
```

注意：GitHub 上的仓库是公开的，密钥、个人信息和录像都不要提交；CI 只在推到 GitHub 时运行，
推到 NAS 不会触发。

## 3. 首次运行

```powershell
Copy-Item .env.example .env     # 可选：不建 .env 时全部用默认值
.\run.ps1                       # 创建 backend\venv、安装依赖、启动 http://localhost:8000
```

`run.ps1` 与 macOS / Linux 上的 `run.sh` 作用相同：

- 第一次运行时创建 `backend\venv`，依次尝试 `py -3.12`、`py -3.13`、`py -3.11`、`py -3.10`、`python`、
  `python3`；想指定解释器就先设 `$env:PYTHON = "C:\<路径>\python.exe"`。
- `backend\requirements.txt` 变了会自动重新安装依赖。
- 设置（`HOST`、`PORT`、`DEV`、`JAVA_PATH` 等）取自环境变量和 `.env`，环境变量优先：

  ```powershell
  $env:DEV = "1"; .\run.ps1                                # 改代码后自动重启
  $env:HOST = "127.0.0.1"; $env:PORT = "8001"; .\run.ps1    # 换地址和端口
  ```

  PowerShell 里这样设的变量在当前窗口一直有效，用完执行 `Remove-Item Env:DEV`，或者另开一个窗口。
  留着它不影响测试：测试会把 `DEV` 固定为 0，也不读取 `.env`（见[第 5 节](#5-测试)）。
- 默认监听 `0.0.0.0`（局域网内其他设备也能访问），第一次启动时 Windows 防火墙会询问是否放行
  Python；只在本机用可以拒绝，或者设 `HOST=127.0.0.1`。

等价的手动步骤：

```powershell
py -3.12 -m venv backend\venv
backend\venv\Scripts\python.exe -m pip install -r backend\requirements.txt
backend\venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

开发时在最后一条命令后加 `--reload --reload-dir backend/app`，改代码后自动重启。
下文所有命令都直接调用 `backend\venv\Scripts\python.exe`，不需要先“激活”虚拟环境。

**`.env` 里值得填的三项**（其余见 `.env.example`；真实的环境变量优先于 `.env`）：

| 变量 | 作用 | 不填时 |
|---|---|---|
| `DEEPSEEK_API_KEY` | AI 教练调用大模型 | 教练使用规则引擎，页面上会标明 |
| `ADMIN_TOKEN` | `/api/admin/*` 和 `/api/parse-local/*` 的口令（请求头 `X-Admin-Token`） | 这些接口一律返回 403 |
| `JAVA_PATH` | 指定 Java 可执行文件 | 自动查找，见下 |

生成一个随机的 `ADMIN_TOKEN`：`py -c "import secrets; print(secrets.token_hex(32))"`。
`.env` 用编辑器保存为 UTF-8（带 BOM 的 UTF-8 和 UTF-16 也能读）。

**`.env` 里的 Windows 路径不要加双引号**，路径里有空格也不用加：

```ini
JAVA_PATH=C:\Program Files\Eclipse Adoptium\jdk-17.0.13.11-hotspot\bin\java.exe
```

- 双引号里的 `\b`、`\t`、`\n`、`\a`、`\f`、`\r`、`\v` 会被读成控制字符，而 JDK 的路径里总有一个 `\bin`：
  结果是日志里的路径看上去没错，Java 却“找不到”。`CLARITY_JAR_PATH`、`RAG_MODEL_DIR`、`RAG_INDEX_DIR`、
  `BILIBILI_COOKIES_FILE` 同理（`\tools`、`\replays`、`\frontend`、`\venv`、`\app` 都会中招）。
- 不加引号、加单引号（`'C:\Program Files\...'`）或改用正斜杠（`C:/Program Files/...`）都能正确读取。
- 万一写成了双引号：程序在 Windows 上会把这几项设置里的控制字符还原成原来的写法，并在启动日志里
  给出一条警告（`... held a control character ...`），照着改掉即可。还原不了的只有一种：双引号里以 `\\`
  开头的网络路径会少一个反斜杠（未在 Windows 上实测）。

**程序在 Windows 上按这个顺序找 Java**（`backend\app\config.py`）：

1. `.env` 或环境变量里的 `JAVA_PATH`（相对路径相对于仓库根目录）；
2. 仓库内的 `tools\jdk17\bin\java.exe`（可选：把 JDK 17 的 zip 解压到 `tools\jdk17\`，该目录不入库）；
3. `%JAVA_HOME%\bin\java.exe`；
4. `PATH` 上的 `java`。

检查 Java 和解析器 jar（`tools\clarity-analyzer.jar`，已随仓库提交，各系统通用）：

```powershell
powershell -ExecutionPolicy Bypass -File tools\setup_parser.ps1
```

服务器启动后也可以看 <http://localhost:8000/api/capabilities>：`"replay_parser": true` 表示两者都可用。
找不到 Java 时服务器照常启动，只是不能上传录像。

**用起来：**

- 打开 <http://localhost:8000>。
- “打开演示比赛”：`replays\8713267082.dem` 存在时解析这个录像，否则显示生成的模拟数据
  （虚构的比赛，没有聊天和基准数据）。
- 输入比赛 ID（例如 `8713267082`）从 OpenDota 获取，或上传录像文件（`.dem`、`.dem.bz2`、`.dem.zst`）。
- 没有 `DEEPSEEK_API_KEY` 时 AI 教练页是规则引擎写的报告；填了密钥后是大模型智能体写的。

## 4. 不在仓库里的本地数据

下面这些都被 `.gitignore` 排除，换电脑后要么从旧电脑复制，要么重新生成。

| 路径 | 内容 | 没有时 | 怎样得到 |
|---|---|---|---|
| `.env` | 密钥和设置 | 全部用默认值：教练只有规则模式，管理接口 403 | 从 `.env.example` 复制后填写 |
| `replays\` | 上传过的录像（按内容哈希命名）和演示录像 `8713267082.dem` | 演示比赛是模拟数据 | 从旧电脑复制，或重新上传；目录由程序自动创建 |
| `backend\data\gamedata\` | 抓取的英雄、技能、物品数据和版本表 | 使用仓库里的后备数据（可能落后于当前游戏版本）；有两个测试跳过 | 服务器启动时自动下载（约 3 MB），或运行 `update_gamedata` |
| `backend\data\learned\` | 自学习得到的知识和按英雄的基准数据 | 教练只用内置的 777 条知识和内置的参考表 | 复制，或运行 `learn`（见下） |
| `backend\data\models\` | 嵌入模型 `bge-small-en-v1.5`（约 35 MB），可选 | 检索照常是混合模式（用仓库里已提交的向量文件，只需要 numpy）；只是新学到的知识没有向量，智能体的自由检索只按关键词 | 装 `requirements-rag.txt` 后运行 `rag download-model` |
| `cache\` | 解析过的比赛、教练报告、运行时向量、每日用量 | 没有影响，按需重建 | 不用复制 |
| `backend\venv\` | 虚拟环境 | — | 不能跨系统复制，由 `run.ps1` 重建 |
| `.claude\` | Claude Code 的本机配置 | — | 见[第 8 节](#8-用-claude-code-在-windows-上继续开发) |

关于自学习（`backend\data\learned\`）：

- 服务器启动时会在后台自动运行到期的层级，所以新电脑第一次启动后会有大约 20 分钟的后台请求；
  `DEV=1` 时不会自动运行，`.env` 里写 `AI_COACH_AUTO_LEARN=0` 可以关掉。手动运行：
  `backend\venv\Scripts\python.exe -m backend.scripts.learn`。
- 第 1、2 层只调用 OpenDota，免费、不需要任何密钥：2026-10-11 在 macOS 上实测第 1 层约 16 分钟
  （635 次请求），第 2 层约 3 分钟。
- 第 3 层（一次大模型请求，约 30 秒）和第 4 层（每个视频一次请求）需要 `DEEPSEEK_API_KEY`，
  会产生大模型费用，填了密钥后自动运行时也一样；没有密钥时这两层自动跳过。第 4 层还需要 `PATH`
  上有 `yt-dlp`（`winget install --id yt-dlp.yt-dlp -e`）。
- 第 4 层默认关闭：它会搜索 B 站和 YouTube、用 yt-dlp 下载字幕。确认这些网站的条款允许之后，在 `.env`
  里写 `AI_COACH_VIDEO_LEARN=1` 才会运行；关着时结果里是 `video_learning_off`，不发任何请求。
- `yt-dlp` 必须是 `yt-dlp.exe`。`PATH` 上排在前面的如果是 `yt-dlp.cmd` / `yt-dlp.bat` 这类包装脚本，
  程序不会运行它（批处理要经过 `cmd.exe`，后者会解释搜索词和网址里的字符），第 4 层按没有 yt-dlp
  跳过（结果里是 `no_yt_dlp`），日志里有一条说明。
- B 站的字幕接口通常需要登录后的 cookies：用浏览器扩展导出 Netscape 格式的 `cookies.txt`，放到
  `backend\data\learned\bilibili_cookies.txt`，或者用 `BILIBILI_COOKIES_FILE` 指定位置。文件按 UTF-8
  读取，带不带 BOM 都可以；它等同于登录凭据，不要提交也不要分享。
- 学到的文件是普通 JSON，可以直接从 macOS 复制过来。
- `backend\data\learned\_learn.lock` 和 `cache\rag\<模型名>.npz.lock` 是锁文件（保证学习任务、向量构建
  同一时间只有一个在跑），内容为空。服务器或命令行工具运行期间不要删除它们，也不要设成只读；
  停掉之后删不删都可以，下次会自动创建。

关于嵌入模型（`backend\data\models\`，可选）：

- `requirements-rag.txt` 里的 `onnxruntime` 在 Windows 上依赖 Microsoft Visual C++ 2015-2022 运行库
  （`msvcp140.dll`，Python 的安装包不带它）。没有它时 `onnxruntime` 导入失败，`rag status` /
  `rag doctor` 会把这两个包报告为未安装。安装：

  ```powershell
  winget install --id Microsoft.VCRedist.2015+.x64 -e
  ```

## 5. 测试

```powershell
backend\venv\Scripts\python.exe -m pip install -r backend\requirements-dev.txt    # pytest、httpx
$env:DEEPSEEK_API_KEY = ""
backend\venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

- 预期结果：没有失败。2026-10-11 在 macOS 上是 1731 项通过、6 项跳过；Windows 上的数字以 CI 的
  Windows 任务为准（未在 Windows 上实测）。加 `-rs` 会列出每一项跳过的原因，正常的有这几类：
  - 需要这台机器没有的东西：下载好的嵌入模型（`RAG_TEST_MODEL_DIR`，两项）、真实录像的目录
    （`CLARITY_TEST_REPLAYS`，一项）、Node.js（一项）；
  - 需要抓取过的游戏数据（两项）：新克隆的仓库上跳过，启动过一次服务器或运行过 `update_gamedata`
    之后就会运行；
  - 只属于另一类系统的测试：Windows 上跳过 POSIX 进程组和 `tools\setup_parser.sh` 的测试；
    macOS / Linux 上跳过需要 PowerShell 的三项（检查 `run.ps1` 和 `tools\setup_parser.ps1`，
    上面 macOS 的 6 项跳过里有它们）。
- 测试不联网、不需要密钥，也不读取 `.env`：`backend\tests\conftest.py` 拦截一切发往本机以外的连接，
  并在加载任何后端模块之前设置 `DOTA_ANALYZER_SKIP_DOTENV=1`（程序见到它就不加载 `.env`，测试启动的
  子进程也会继承它）和 `DEV=0`。所以 `.env` 里写了什么、当前窗口里是否还留着 `$env:DEV = "1"`，
  都不影响测试结果。
- `$env:DEEPSEEK_API_KEY = ""` 在 PowerShell 7.5 及以上把变量设为空，在更早的版本和 Windows
  PowerShell 5.1 里是删除变量。两种情况测试都不会用到 `.env` 里的真实密钥（原因见上一条）。
  在 7.5 及以上，这个空值会留在当前窗口里并且优先于 `.env`：之后要在同一个窗口启动服务器，先执行
  `Remove-Item Env:DEEPSEEK_API_KEY`，否则教练会是规则模式。
- 只跑一个文件：`backend\venv\Scripts\python.exe -m pytest -q backend\tests\test_geo.py`。
- 检查有没有漏写 `encoding="utf-8"` 的文本读写（中文、日文 Windows 的默认编码是 GBK、CP932，
  漏写就会读错文件）。CI 的 Windows 任务带着这两个参数运行：

  ```powershell
  backend\venv\Scripts\python.exe -X warn_default_encoding -W error::EncodingWarning -m pytest -q -p no:cacheprovider
  ```

**CI。** `.github\workflows\ci.yml` 在每次推送到 GitHub 和每个 pull request 时运行，结果在仓库页面的
Actions 标签下：

- 测试套件：Windows、Linux、macOS 上的 Python 3.12，另有 Linux 上的 Python 3.10，以及一个额外
  安装 `requirements-rag.txt` 的 Linux 任务；Windows 任务带上面的编码检查。
- 冒烟测试：`tools\setup_parser.ps1`；命令行工具（输出重定向到管道）；真实启动的服务器
  （uvicorn 不带和带 `--reload`，Windows 上还有用 `run.ps1` 从零创建虚拟环境再启动），每次都请求
  演示比赛、规则模式的中文教练报告，并上传一个损坏的录像，确认请求确实到了 Java 解析器。
- Java 解析器的单元测试和 jar 构建（Linux 用 `gradlew`，Windows 用 `gradlew.bat`）。

想在自己的页面上显示状态徽章：
`![CI](https://github.com/<owner>/ProjectDotaAnalyzer/actions/workflows/ci.yml/badge.svg)`。

## 6. 常用命令

```powershell
$py = "backend\venv\Scripts\python.exe"

& $py -m backend.scripts.update_gamedata --check-only     # 游戏版本、数据来源、是否需要更新
& $py -m backend.scripts.update_gamedata                  # 下载英雄 / 技能 / 物品数据
& $py -m backend.scripts.learn --status                   # 学到了什么、哪些层级到期
& $py -m backend.scripts.learn                            # 运行到期的层级
& $py -m backend.scripts.learn --levels "1,2" --force     # 只跑免费的两层（约 20 分钟）
& $py -m backend.scripts.rag status                       # 检索模式、向量数、覆盖率
& $py -m backend.scripts.rag eval                         # 检索基准测试
& $py -m backend.scripts.gen_static_catalogs --check      # 已提交的英雄 / 物品 / 技能目录是否最新（需要联网）

# 可选：本机的嵌入模型（给新学到的知识生成向量、智能体的语义检索）
& $py -m pip install -r backend\requirements-rag.txt
& $py -m backend.scripts.rag download-model               # 约 35 MB，下载到 backend\data\models\
& $py -m backend.scripts.rag doctor                       # 包、模型校验和、本机算出的向量是否与已提交的一致
& $py -m backend.scripts.rag build                        # 为缺少向量的文本生成向量，写入 cache\rag\
```

- PowerShell 把不加引号的 `1,2` 当成数组拆成两个参数，所以 `--levels "1,2"` 的引号不能省。
- 运行中的服务器会在约 30 秒内自己发现新的数据文件，这些命令之后都不用重启。

**把解析器的输出和 OpenDota 对比：**

```powershell
cmd /c "java -jar tools\clarity-analyzer.jar <录像.dem> > raw.json"
curl.exe -sS -o opendota.json "https://api.opendota.com/api/matches/<比赛ID>"
& $py tools\compare_with_opendota.py raw.json opendota.json
```

第一条特意通过 `cmd /c` 执行：Windows PowerShell 5.1 的 `>` 会把程序输出转存成 UTF-16，
得到的 `raw.json` 读不回来。第二条写 `curl.exe` 而不是 `curl`：后者在 5.1 里是另一个命令的别名。

**重新构建解析器的 jar**（只有改了 `tools\clarity-analyzer\` 下的 Java 源码才需要；需要 JDK 17，
第一次会从网络下载 Gradle 和依赖）：

```powershell
cd tools\clarity-analyzer
.\gradlew.bat test --no-daemon                                         # Java 单元测试
cd ..\..
powershell -ExecutionPolicy Bypass -File tools\setup_parser.ps1 -Rebuild   # 构建并替换 tools\clarity-analyzer.jar
```

已提交的 jar 是用 Eclipse Temurin 17.0.13+11 构建的，只有同一个 JDK 版本才能得到逐字节相同的文件；
用别的 JDK 17 构建出来功能相同，但 Git 会把 `tools\clarity-analyzer.jar` 显示为已修改
（撤销：`git checkout -- tools/clarity-analyzer.jar`）。没改源码就不要重建。

## 7. 与 macOS / Linux 的差异和已知限制

Windows 上实现方式不同的部分（都集中在 `backend\app\osutil.py`，POSIX 上的行为没有变）：

| 功能 | macOS / Linux | Windows |
|---|---|---|
| 文件锁（学习任务、向量构建同一时间只跑一个） | `fcntl.flock` | `msvcrt.locking` 锁住锁文件的第一个字节；等待锁时每 0.1 秒重试 |
| 结束超时的解析进程 | 向进程组发 `SIGKILL` | 每个 JVM 启动后放进它自己的作业对象（Job Object），结束时终止作业里的全部进程；建不了作业对象时退回 `taskkill /T /F` |
| 服务器进程被强行结束（`kill -9`；任务管理器、`Stop-Process`） | 正在解析的 JVM 会留下，直到这次解析结束 | 系统随服务器进程一起结束作业对象里的 JVM（未在 Windows 上实测） |
| 原子替换文件 | `os.replace` | `os.replace`，目标文件被占用时在 1 秒内重试 |
| `--reload` 下启动解析器 | asyncio 子进程 | uvicorn 在 Windows 的 `--reload` 模式使用不支持子进程的事件循环，改为 `subprocess` 加线程等待 |
| 重定向到文件或管道的输出 | UTF-8 | 系统代码页；代码页里没有的字符写成 `\uXXXX`，不会报错。想要 UTF-8 就先设 `$env:PYTHONIOENCODING = "utf-8"` |

已知限制：

- `run.sh`、`tools\setup_parser.sh` 和 `deploy\` 下的 systemd / nginx 文件只用于 macOS / Linux；
  Windows 上对应的是 `run.ps1` 和 `tools\setup_parser.ps1`，没有 Windows 服务的部署方案。
- CI 只覆盖 x64 的 Windows；ARM64 的 Windows 没有验证过（个别依赖可能没有 ARM64 的安装包，
  遇到时改装 x64 版的 Python）。
- **嵌入模型在 x64 上没有验证过**：已提交的向量文件是在 Apple 芯片上生成的。装了模型之后先运行
  `rag doctor`；如果它报告本机算出的向量与已提交的不一致，服务器会继续用已提交的向量排序，
  只是不给新文本生成向量。不装模型则完全不受影响。
- **路径里的非英文字符**：`java.exe` 按系统代码页读取命令行，仓库路径或录像路径里含有系统代码页
  没有的字符（例如日文 Windows 上的简体中文目录名）时录像无法解析。把仓库放在纯英文路径下。
- **长路径**：仓库自己的文件路径最长 77 个字符，虚拟环境里的文件相对仓库根目录最长约 170 个字符
  （macOS 上测得）。克隆位置很深时可能超过 Windows 的 260 字符限制，这时把仓库移到更短的路径，
  或者打开长路径支持（第二条需要管理员权限）：

  ```powershell
  git config --global core.longpaths true
  New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name LongPathsEnabled -Value 1 -PropertyType DWord -Force
  ```

- **占用中的文件**：Windows 不能删除或替换正被打开的文件。手动清理 `cache\`、`replays\` 或替换
  `tools\clarity-analyzer.jar` 时如果提示文件正在使用，先停掉服务器。
- **性能**（未在 Windows 上实测）：启动进程（Python、JVM）和大量小文件读写比 macOS / Linux 慢，
  实时杀毒扫描会再拖慢导入和测试。macOS 上整套测试约 45 秒，Windows 上的实际耗时看 CI 的
  Windows 任务。如果确认是 Microsoft Defender 的实时扫描造成的，可以自行决定把虚拟环境加入排除
  （管理员 PowerShell；这会降低该目录的防护）：

  ```powershell
  Add-MpPreference -ExclusionPath "<仓库路径>\backend\venv"
  ```

- **`.env` 的权限**：Linux 上的 `chmod 600 .env` 在 Windows 上没有对应效果；文件位于你自己的用户
  目录下即可，不要把它放进共享目录或网盘同步目录。

## 8. 用 Claude Code 在 Windows 上继续开发

- 项目级的说明 `CLAUDE.md` 在仓库里，克隆后直接生效，里面有 Windows 的运行和测试命令。
- Claude Code 在原生 Windows 上默认通过 Git for Windows 自带的 Git Bash 执行命令，所以它写的是
  bash 语法，虚拟环境的解释器是 `backend/venv/Scripts/python.exe`（不是 `backend/venv/bin/python`）。
  让它跑测试时用下面这条；如果你的 Claude Code 用的是 PowerShell，就用第 5 节的写法。

  ```bash
  DEEPSEEK_API_KEY="" backend/venv/Scripts/python.exe -m pytest -q -p no:cacheprovider
  ```

- `.claude\` 不入库，预览用的 `.claude\launch.json` 需要在新电脑上重新创建，内容如下：

  ```json
  {
    "version": "0.0.1",
    "configurations": [
      {
        "name": "dev",
        "runtimeExecutable": "backend\\venv\\Scripts\\python.exe",
        "runtimeArgs": ["-m", "uvicorn", "backend.app.main:app", "--host", "127.0.0.1", "--port", "8000", "--reload", "--reload-dir", "backend/app"],
        "port": 8000
      }
    ]
  }
  ```

  先运行一次 `.\run.ps1`（或第 3 节的手动步骤），让 `backend\venv` 存在。这样启动没有设 `DEV=1`，
  服务器仍会在后台运行到期的自学习，每次自动重启都会打断它；不想这样就在 `.env` 里写
  `AI_COACH_AUTO_LEARN=0`，需要时手动运行 `learn`。
- **对话记忆不随仓库走。** 旧电脑上 Claude Code 为这个项目记下的记忆保存在那台电脑的用户目录里
  （`~/.claude/projects/<项目目录标识>/memory/`），不在仓库中。想带过来，就把那个 `memory` 目录复制到
  新电脑的 `%USERPROFILE%\.claude\projects\<项目目录标识>\memory\`。`<项目目录标识>` 由项目的绝对路径
  生成，两台电脑不一样：先在新电脑的仓库目录里启动一次 Claude Code，再到
  `%USERPROFILE%\.claude\projects\` 下找对应这个仓库路径的目录（未在 Windows 上实测）。
  记忆文件里可能有个人信息，不要放进仓库。

## 9. 历史重写之后：更新已有的克隆

仓库的历史在 2026-10 被重写过一次，所有提交号都变了。**在那之前克隆的副本（部署服务器、
其他电脑）不能再用 `git pull`**：它要么报错，要么把旧历史和新历史合并在一起。
按下面的顺序做（示例是 bash，PowerShell 里命令相同）：

```bash
cd <安装目录>
git status --short                         # 1. 有没有未提交的改动？有就先备份到仓库之外
git log --oneline origin/<分支>..HEAD      # 2. 取新历史之前：有没有没推送过的本地提交？应当为空
git fetch --prune origin                   # 3. 取回新历史，会显示 (forced update)；远端已删的分支一并清掉
git reset --hard origin/<分支>             # 4. 让本地分支和工作区与新历史完全一致
```

- **第 4 步会丢弃本地分支上所有不在新历史里的提交，以及对已跟踪文件的未提交改动。** 第 1、2 步
  发现的东西要先处理：改动另存到仓库之外，本地提交记下提交号，重置之后用
  `git cherry-pick <提交号>` 重新应用。其他本地分支也各自重置，或者用 `git branch -D <分支>` 删掉。
- 不受影响：`.gitignore` 排除的文件，即 `.env`、`replays/`、`cache/`、`backend/data/gamedata/`、
  `backend/data/learned/`、`backend/data/models/`、`backend/venv/`。
- 旧历史的对象还留在这个克隆里。确认没有分支再指向旧历史之后，彻底清除：

  ```bash
  git reflog expire --expire=now --all
  git gc --prune=now
  ```

- 服务器上随后照常更新依赖并重启（见 `deploy/README.md` 的 “Redeploy”）：

  ```bash
  backend/venv/bin/pip install -r backend/requirements.txt
  sudo systemctl restart dota-analyzer
  ```

- 最省事的替代做法是重新克隆到新目录，再把上面列出的本地数据（至少 `.env`）搬过去。
