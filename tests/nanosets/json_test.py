import os
from typing import Any, Dict, List, Optional

from nanorlhf.nanosets.table.record_batch import RecordBatch
from nanorlhf.nanosets.table.table import Table
from nanorlhf.nanosets.io.json_io import to_json, to_jsonl, from_json, from_jsonl


def generate_rows(n: int = 1000) -> List[Optional[Dict[str, Any]]]:
    """Create a list of row dicts with nested lists and structs (plus some nulls)."""
    rows: List[Optional[Dict[str, Any]]] = []
    for i in range(n):
        row: Dict[str, Any] = {
            "id": i,                                 # primitive int
            "name": f"user_{i}",                     # string
            "score": None if i % 7 == 0 else i / 10, # nullable float
            "tags": None if i % 10 == 0 else [f"t{i%3}", f"t{(i+1)%3}"],  # list<string> | null
            "metrics": None if i % 13 == 0 else {    # struct | null
                "clicks": i * 2,
                "rate": (i % 5) / 5.0,
            },
            "nested": None if i % 9 == 0 else [      # list<list<int>>
                [i, i + 1],
                []
            ],
        }
        # sprinkle in some completely-null rows for robustness
        if i % 111 == 0:
            rows.append(None)  # a null row (StructArray handles this)
        else:
            rows.append(row)
    return rows


def main(tmp_dir: str = "data/json_test"):
    os.makedirs(tmp_dir, exist_ok=True)
    json_path = os.path.join(tmp_dir, "data.json")
    jsonl_path = os.path.join(tmp_dir, "data.jsonl")

    # 1) Build a RecordBatch from Python rows (schema inferred inside)
    rows = generate_rows(1000)
    batch = RecordBatch.from_list(rows)
    table = Table([batch])

    # 2) Save as JSON (with schema) and JSONL (rows only)
    with open(json_path, "w", encoding="utf-8") as fp:
        to_json(fp, table, indent=2)

    with open(jsonl_path, "w", encoding="utf-8") as fp:
        to_jsonl(fp, table)

    # 3) Load back
    table_from_json = from_json(json_path)
    table_from_jsonl = from_jsonl(jsonl_path)

    # 4) Verify round-trip equality against original logical rows
    original_rows = table.to_list()
    rows_from_json = table_from_json.to_list()
    rows_from_jsonl = table_from_jsonl.to_list()

    assert rows_from_json == original_rows, "JSON round-trip mismatch"
    assert rows_from_jsonl == original_rows, "JSONL round-trip mismatch"

    print("✅ JSON / JSONL round-trip tests passed.")
    print(f"   - {json_path}")
    print(f"   - {jsonl_path}")


if __name__ == "__main__":
    main()
