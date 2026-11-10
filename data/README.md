# Local datasets

Dataset files are intentionally excluded from Git because the audio archives
and extracted recordings are large and have their own licensing terms.

From the repository root, place the data in this layout:

```text
data/
├── librispeech_finetuning/
│   ├── 1h/
│   └── 9h/
└── LibriSpeech/
    ├── dev-clean/
    ├── dev-other/
    ├── test-clean/
    └── test-other/
```

The loader expects `.flac` recordings and matching `.trans.txt` transcript
files. After preparing the dataset, validate it with:

```bash
python -m src.data --verify
```

The verification command can generate manifests and `data/vocab.json` when
`--write-manifests` is passed. Those generated files are local artifacts and
are also excluded from Git.
