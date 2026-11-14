import os

from nanorlhf.nanosets.io.ipc import write_table, read_table
from nanorlhf.nanosets.table.record_batch import RecordBatch
from nanorlhf.nanosets.table.table import Table


def gen_rows(n=1000):
    rows = []
    for i in range(n):
        rows.append({
            "id": i,
            "name": f"user_{i}",
            "score": None if i % 7 == 0 else i / 10,
            "tags": None if i % 10 == 0 else [f"t{i % 3}", f"t{(i + 1) % 3}"],
            "metrics": None if i % 13 == 0 else {"clicks": i * 2, "rate": (i % 5) / 5.0},
            "nested": None if i % 9 == 0 else [[i, i + 1], []],
        })
    return rows


def main(tmp_dir="data/ipc_test"):
    os.makedirs(tmp_dir, exist_ok=True)
    path = os.path.join(tmp_dir, "table.nano")

    # build
    rows = gen_rows(1000)
    batch = RecordBatch.from_list(rows)
    table = Table([batch])

    # write
    with open(path, "wb") as fp:
        write_table(fp, table)

    # read
    table2 = read_table(path)

    # verify
    assert table2.to_list() == table.to_list()
    print("✅ IPC round-trip passed:", path)


def test_round_trip_of_sliced_table_keeps_nulls(tmp_path):
    # Slicing from row 1 gives validity bitmaps that start mid-byte (bit_offset=1).
    rows = gen_rows(20)
    table = Table([RecordBatch.from_list(rows)])
    for offset, length in [(1, 8), (3, 13), (6, 9)]:
        sliced = table.slice(offset, length)
        assert any(col.validity is not None and col.validity.bit_offset for b in sliced.batches for col in b.columns)
        assert any(v is None for r in rows[offset:offset + length] for v in r.values())
        path = tmp_path / f"slice_{offset}_{length}.nano"
        with open(path, "wb") as fp:
            write_table(fp, sliced)
        assert read_table(str(path)).to_list() == rows[offset:offset + length]


if __name__ == "__main__":
    main()
