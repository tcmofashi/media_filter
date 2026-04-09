# MediaFilter WebUI

This is the shipped frontend for the public baseline pipeline.

## Routes

- `/label`: 标注与导出（文件夹扫描 / 推理结果 / labels 导出）
- `/pipeline`: 任务面板（训练、下载、全链路执行与任务日志）

## Start

Use the unified Linux entrypoint:

```bash
./start.sh frontend
```

The frontend is available at `http://localhost:31212` by default.

## Notes

- Backend API is required for label/pipeline features (`src/main.py`).
- API endpoint docs (when API mode is enabled) at `http://localhost:31211/docs`.
