from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from langchain_core.documents import Document


def _normalize_cell(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def parse_excel_to_documents(file_path: str | Path) -> list[Document]:
    """Convert each Excel row into narrated text with source metadata."""
    path = Path(file_path)
    sheets = pd.read_excel(path, sheet_name=None, dtype=object)
    documents: list[Document] = []

    for sheet_name, frame in sheets.items():
        if frame.empty:
            continue

        frame = frame.copy()
        frame.columns = [str(col).strip() for col in frame.columns]

        for row_idx, row in frame.iterrows():
            parts: list[str] = []
            used_columns: list[str] = []
            for col_name in frame.columns:
                value = _normalize_cell(row[col_name])
                if not value:
                    continue
                parts.append(f"{col_name} is {value}")
                used_columns.append(col_name)

            if not parts:
                continue

            excel_row = int(row_idx) + 2
            narrated = f"Sheet '{sheet_name}' row {excel_row}: " + ", ".join(parts)
            documents.append(
                Document(
                    page_content=narrated,
                    metadata={
                        "source": str(path.name),
                        "sheet": sheet_name,
                        "row": excel_row,
                        "columns": used_columns,
                        "cell_range": f"A{excel_row}:XFD{excel_row}",
                        "doc_type": "excel",
                    },
                )
            )

    return documents
