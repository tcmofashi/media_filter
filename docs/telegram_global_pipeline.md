# Telegram Global Pipeline

## 当前实现

本项目内置了一个本地 `tg_downloader/` 包，用来做 Telegram 全局游走门控下载。

它的工作方式是：

1. 用 `client.get_dialogs()` 发现当前账号可见的 dialogs。
2. 优先从本项目 `configs/config.yaml` 的 `telegram` 段读取默认 session 名、目录和路径配置。
3. 从原 `~/telegram_media_downloader` 读取 `config.yaml`、`data.yaml` 和 `.session`。
4. 把 legacy 里的 API 凭据、代理、`last_read_message_id`、`ids_to_retry` 迁移到本项目本地状态。
5. 调度器按频道广度优先轮转，每个频道每次只处理一个小批次，默认 `150` 个媒体消息。
6. 冷启动平铺若干轮后，会根据每个频道累计平均分优先回访高分频道做深挖。
7. 媒体先下载到本地缓存目录。
8. 用当前 Frozen CLIP 评分器打分，并把媒体结果和频道聚合统计写入 sqlite。
9. 只把达到阈值的文件物化到目标目录。
10. 目标目录文件名前会加上分数前缀，例如 `8.2445__10615_314_4.mp4`。
11. 低分文件默认不进入目标目录，但会保留在缓存目录，方便复查或重跑。
12. `cache_root` 默认最多保留 `100` 个媒体，超过后按分数从低到高淘汰。

## 默认目录

默认运行目录全部放在项目内，并且已经被 `.gitignore` 的 `data/*` 覆盖：

- 缓存目录：`data/tg_cache`
- 目标目录：`data/tg_target`
- 本地 session：`data/tg_session`
- 本地状态：`data/tg_downloader_state.json`
- 推理缓存库：`data/mediaflusher.db`
- 默认 session 配置：`configs/config.yaml -> telegram.session_name = mediaflusher_tg`

这就是两层文件缓存：

- 第一层 `cache_root` 保存 Telegram 原始下载结果。
- 第二层 `target_root` 只保存门控通过后的结果，默认用 `hardlink` 从缓存层物化。
- `target_root` 的文件名会带分数前缀，缓存层文件名保持原始稳定命名。
- 两层目录都按 `chat_type/频道名/文件` 组织，不再按日期拆目录。

## 直接运行

如果 `~/telegram_media_downloader/config.yaml`、`data.yaml`、`sessions/media_downloader.session`
都存在，默认不需要再手填 Telegram 参数：

```bash
cd /path/to/mediaflusher
python scripts/run_tg_gated_download.py --min-score 7.2
```

默认会：

- 自动读取原项目的 `api_id`、`api_hash`、代理配置。
- 自动使用 `configs/config.yaml` 里的默认 session 名 `mediaflusher_tg`。
- 只在本地默认 session 不存在时，才复制原 `.session` 到 `data/tg_session/`。
- 自动把原项目里的聊天进度和失败重试列表写入本地状态文件。
- 自动全局发现 `channel,supergroup,group,private`。
- 按频道 round-robin 小批量下载，默认每批 `150` 个媒体消息。
- 媒体先下到 `data/tg_cache/`，再按阈值物化到 `data/tg_target/`。
- 频道聚合统计会写入 sqlite 的 `telegram_chat_stats` 表。
- `data/tg_cache/` 默认只保留最高分的 `100` 个媒体文件。

如果做过一次新的交互式登录，只要把新的 session 固化为 `data/tg_session/mediaflusher_tg.session`，
后续脚本就会默认复用它，不需要再次输入验证码。

如果原 `.session` 已失效，但你仍然想复用原配置和代理做一次新的交互式登录：

```bash
python scripts/run_tg_gated_download.py \
  --min-score 7.2 \
  --session-name mediaflusher_tg_fresh \
  --skip-legacy-session-copy
```

这时脚本不会复制旧 session，而是直接让 Pyrogram 创建一个新 session。

常用参数：

```bash
python scripts/run_tg_gated_download.py \
  --min-score 7.2 \
  --chat-batch-size 150 \
  --cache-max-items 100 \
  --target-root /mnt/H/telegram_pass \
  --cache-root /mnt/H/telegram_cache \
  --target-mode hardlink
```

这里的 `--chat-batch-size` 现在按“媒体消息”计数；中间遇到的纯文本/不支持媒体消息仍会被顺序走过，并标记为 `skipped` 以推进断点。

如果要测试时保留低分结果也进入目标目录：

```bash
python scripts/run_tg_gated_download.py \
  --min-score 7.2 \
  --keep-below-threshold
```

## 小范围真实测试

建议先用很小的范围验证账号、代理、会话和模型都能工作：

```bash
cd /path/to/mediaflusher
python scripts/run_tg_gated_download.py \
  --min-score 7.2 \
  --max-chats 1 \
  --history-limit 5 \
  --log-every 1
```

## 一键编排

如果要跑完整 pipeline：

```bash
cd /path/to/mediaflusher
python scripts/run_telegram_global_pipeline.py \
  --min-score 7.2
```

它会按顺序做：

1. 全局游走门控下载到 `data/tg_cache` / `data/tg_target`
2. 可选地对 `target_root` 做全量补推理
3. 可选地把高分文件按阈值导出到 `target_root/score_links/`
4. 可选地清理 `target_root` 里低于阈值的文件

## 限制

- 当前评分器仍然需要先拿到完整媒体文件，所以只能做到“下载后门控”，做不到真正“下载前门控”。
- 现在只处理评分器支持的图片和视频。
- pipeline 脚本默认会把下载阶段按单轮执行；如果想让下载阶段常驻，需要显式传 `--continuous`。
- `--dry-run` 只影响 rebucket / prune，不影响 Telegram 下载阶段。
