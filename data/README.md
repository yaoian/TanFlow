# TanFlow Benchmark Datasets

All six preprocessed benchmark datasets (`atc.npz`, `indoor.npz`, `sdd.npz`, `edinburgh.npz`, `zara1.npz`, `zara2.npz`) are hosted on Zenodo:

- **Zenodo DOI:** [https://doi.org/10.5281/zenodo.24163449](https://doi.org/10.5281/zenodo.24163449)
- **Direct Archive Download:** [tanflow_benchmarks.tar.gz](https://zenodo.org/records/24163449/files/tanflow_benchmarks.tar.gz?download=1) (10.75 MB)

## Quick Download

Run the download helper from the repository root to fetch and extract all benchmarks into this directory:

```bash
bash scripts/download_data.sh
```

Or download and extract manually:

```bash
curl -fL "https://zenodo.org/records/24163449/files/tanflow_benchmarks.tar.gz?download=1" -o tanflow_benchmarks.tar.gz
tar -xzvf tanflow_benchmarks.tar.gz -C data/
```
