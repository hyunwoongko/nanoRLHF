import sys
from pathlib import Path


def main():
    try:
        from datasets import load_dataset as hf_load_dataset  # pip install datasets
    except Exception:
        print("This test requires the 'datasets' package. Install with: pip install datasets")
        sys.exit(1)

    from nanorlhf.nanosets.api import load_dataset as ns_load_dataset

    # 1) Download tiny slice from HF (GSM8K: main)
    print("Downloading tiny split from Hugging Face (openai/gsm8k:main, first 200 rows)...")
    ds_all = hf_load_dataset("openai/gsm8k", "main")
    ds_small = ds_all["train"].select(range(200))
    assert len(ds_small) == 200

    # 2) Write JSONL
    out_dir = Path("data/api_test")
    out_dir.mkdir(parents=True, exist_ok=True)
    src_jsonl = out_dir / "hf_sample.jsonl"
    ds_small.to_json(str(src_jsonl))

    size_mb = src_jsonl.stat().st_size / (1024 * 1024)
    print(f"Wrote HF sample JSONL → {src_jsonl} ({size_mb:.1f} MB)")

    # 3) Load via our Dataset API (auto by file extension)
    print("Loading JSONL via our Dataset API...")
    ds_ns = ns_load_dataset(data_files=[str(src_jsonl)])  # returns Table-like obj with to_dict()

    # 4) Validate
    rows_ref = [dict(r) for r in ds_small]
    rows_ns = ds_ns.to_dict()

    assert len(rows_ns) == len(rows_ref) == 200

    # Spot-check first 3 rows
    for i in range(3):
        assert set(rows_ns[i].keys()) == {"question", "answer"}
        assert rows_ns[i]["question"] == rows_ref[i]["question"]
        # GSM8K answers are strings; many include final '#### <number>'
        assert isinstance(rows_ns[i]["answer"], str)
        # optional soft check (don't fail if missing, just note)
        if "####" not in rows_ns[i]["answer"]:
            print(f"Note: row {i} answer doesn't contain '####' (format variations are possible).")

    # 5) Round-trip: write JSONL via our API and load again
    roundtrip_jsonl = out_dir / "ns_roundtrip.jsonl"
    ds_ns.to_json(str(roundtrip_jsonl))  # schema excluded by API design
    ds_ns2 = ns_load_dataset(data_files=[str(roundtrip_jsonl)])
    rows_ns2 = ds_ns2.to_dict()

    assert len(rows_ns2) == 200
    for i in (0, 50, 199):
        assert rows_ns2[i]["question"] == rows_ns[i]["question"]
        assert rows_ns2[i]["answer"] == rows_ns[i]["answer"]

    print("✓ HF → JSONL → nanosets load: OK")
    print("✓ nanosets → JSONL (lines=True) → nanosets load: OK")
    print(f"\nArtifacts:\n - Source JSONL: {src_jsonl}\n - Round-trip JSONL: {roundtrip_jsonl}")


if __name__ == "__main__":
    main()
