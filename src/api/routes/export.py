import csv
import io
import json
from fastapi import APIRouter, Query, Response

from src.storage.database import db

router = APIRouter(prefix="/export", tags=["export"])


@router.get("/labels")
async def export_labels(
    format: str = Query("json", pattern="^(json|csv)$", description="Output format"),
):
    labels = await db.get_all_labels_for_export()

    if format == "json":
        return Response(
            content=_to_json_lines(labels),
            media_type="application/json",
            headers={"Content-Disposition": "attachment; filename=labels.json"},
        )
    else:
        return Response(
            content=_to_csv(labels),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=labels.csv"},
        )


def _to_json_lines(labels: list) -> str:
    return "\n".join(json.dumps(label) for label in labels)


def _to_csv(labels: list) -> str:
    output = io.StringIO()
    if labels:
        writer = csv.DictWriter(output, fieldnames=["media_path", "score"])
        writer.writeheader()
        writer.writerows(labels)
    return output.getvalue()


@router.get("/stats")
async def get_stats():
    stats = await db.get_labeling_stats()
    return stats
