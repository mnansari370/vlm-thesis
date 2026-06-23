"""
v2/distill/docvqa_data.py — unified DocVQA loader.

The locally-cached `lmms-lab/DocVQA` builder only exposes validation+test (the repo declares
those splits even though train-*.parquet physically exist). So:
  - split in {validation, test} -> normal datasets.load_dataset (works, image decoded).
  - split == train               -> read the train parquet shards DIRECTLY (hf_hub_download,
                                     cached), decode image bytes -> PIL on access.

Both return a `ds[i] -> {"image": PIL, "question": str, "answers": [str], "docId": ...}` interface
so cache_teacher / train_student / eval_gate use one call regardless of split. Zero leakage:
train (real train split) and validation (gate) are entirely different datasets.
"""

import bisect
import io

from PIL import Image

REPO = "lmms-lab/DocVQA"
CONFIG = "DocVQA"


class _ParquetTrain:
    """Random-access view over the first `n_shards` DocVQA train parquet shards."""

    def __init__(self, n_shards=None):
        from huggingface_hub import HfApi, hf_hub_download
        import pyarrow.parquet as pq

        files = sorted(
            s.rfilename for s in HfApi().dataset_info(REPO).siblings
            if s.rfilename.startswith(f"{CONFIG}/train-") and s.rfilename.endswith(".parquet")
        )
        if not files:
            raise RuntimeError("no DocVQA train parquet shards found on the hub")
        if n_shards is not None:
            files = files[:n_shards]

        self.tables = []
        self.cum = [0]
        for f in files:
            fp = hf_hub_download(REPO, f, repo_type="dataset")
            t = pq.read_table(fp, columns=["question", "image", "answers", "docId"])
            self.tables.append(t)
            self.cum.append(self.cum[-1] + t.num_rows)
        self.n = self.cum[-1]
        print(f"[docvqa] train: {len(files)} shard(s), {self.n} rows", flush=True)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        if i < 0 or i >= self.n:
            raise IndexError(i)
        s = bisect.bisect_right(self.cum, i) - 1
        r = self.tables[s].slice(i - self.cum[s], 1).to_pylist()[0]
        return {
            "image": Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB"),
            "question": r["question"],
            "answers": [str(a) for a in (r["answers"] or [])],
            "docId": r.get("docId"),
        }


def load_docvqa(split, n_shards=None):
    if split in ("validation", "test"):
        from datasets import load_dataset
        return load_dataset(REPO, CONFIG, split=split)
    if split == "train":
        return _ParquetTrain(n_shards)
    raise ValueError(f"unknown split: {split}")
