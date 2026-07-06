"""Cache ChartQA (test) and DocVQA (validation) via HF datasets for the high-res
QC-selection confirmation. Images are embedded PIL; the HF cache is kept and the dataset
object is iterated directly in the eval (no manual image extraction)."""
import sys
from datasets import load_dataset

def main():
    print("=== ChartQA (test) ===", flush=True)
    c = load_dataset("lmms-lab/ChartQA", split="test")
    print("ChartQA test:", len(c), "examples", flush=True)
    print("=== DocVQA (validation) ===", flush=True)
    d = load_dataset("lmms-lab/DocVQA", "DocVQA", split="validation")
    print("DocVQA validation:", len(d), "examples", flush=True)
    print("DOWNLOAD_COMPLETE", flush=True)

if __name__ == "__main__":
    main()
