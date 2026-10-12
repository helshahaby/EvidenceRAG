# EvidenceRAG — AMD AI League Match 3

**GPU-aware, extractive retrieval-augmented question answering for mixed enterprise documents, with exact source-file citations.**

EvidenceRAG is a Dockerized Python application built around the AMD AI League Match 3 contract. It indexes a corpus once, then accepts questions through separate command invocations and writes a small JSON result for each query. It is designed to prioritize evidence, precise citations, and abstention over fluent but unsupported answers.

> **Implementation note:** this project uses a pretrained *embedding model* (`sentence-transformers/all-MiniLM-L6-v2`) for semantic retrieval. It is an extractive RAG system, **not** a generative LLM. The answer and citation selection logic is implemented in Python. The Dockerfile downloads the model files during image build; model weights are not bundled in this source ZIP. A successful build therefore requires network access to Hugging Face. Evaluation-time inference is configured to use the local model files only.

## Contents

- `starter/app/app.py` — indexing, parsers, retrieval, answer extraction, and JSON writer.
- `starter/app/requirements.txt` — Python parser and model dependencies.
- `starter/Dockerfile` — mandated AMD ROCm base image, OCR setup, pinned Transformers stack, model download, and application packaging.
- `mc3-corpus/` — sample mixed-document corpus supplied with the challenge.
- `sample-questions.json` — sample questions and expected results supplied with the starter kit.
- `selfcheck.py` — checks the container contract against a chosen corpus.
- `CONTRACT.md` — challenge interface and constraints from the starter kit.
- `setup.sh` — recreates the empty-directory and unreadable-file cases that ZIP archives cannot preserve.

## How it works

1. **Walk the corpus defensively.** The indexer visits nested directories in a stable order, filters to supported extensions, and isolates read/parse errors so a single bad file does not abort the rest of the walk.
2. **Extract text from mixed formats.** Parsers handle PDF, DOCX paragraphs and tables, XLSX worksheets, CSV, TXT, LOG, and Python source. PNG/JPG/JPEG files are processed with Tesseract OCR when available.
3. **Build a reusable index.** Extracted text is recorded with corpus-relative paths. The pretrained MiniLM encoder embeds text lines during `--index`; cached embeddings are saved for subsequent query processes.
4. **Retrieve evidence.** A question is embedded and compared with the cached vectors. Semantic similarity is combined with lexical matching so exact identifiers such as part numbers, ticket IDs, and error codes remain discoverable.
5. **Extract a concise value.** The application applies targeted extraction and conservative rules to return the answer value rather than a conversational explanation. Some multi-hop cases need more than one evidence file.
6. **Write the contract JSON.** Every query writes `answer`, `citations`, and `confidence` to `/app/output/<query-id>_output.json`. If the evidence is insufficient, the intended behavior is an empty answer and an empty citation list.

### Supported inputs

| Extension | Handling |
|---|---|
| `.pdf` | Text extraction with `pypdf`; encrypted/unreadable PDFs are skipped if they cannot be read. |
| `.docx` | Paragraphs and table cells. |
| `.xlsx` | All worksheets, including non-empty cells. |
| `.csv` | Rows and fields. |
| `.txt`, `.log`, `.py` | UTF-8 text with tolerant decoding. |
| `.png`, `.jpg`, `.jpeg` | OCR using Tesseract. |
| Unknown extension | Skipped. |

Scanned or diagram-heavy images depend on OCR quality. This baseline does not include a separate vision-language model; OCR may not interpret complex diagrams as reliably as a purpose-built vision model.

## Requirements

- Linux host with Docker.
- Access to the challenge-mandated ROCm base image.
- Network access during image build to fetch pinned Python packages and the MiniLM model files from Hugging Face.
- For the competition GPU requirement: compatible AMD hardware and working ROCm device access at runtime.

The supplied Dockerfile starts from:

`rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0`

**Do not replace the mandated base image.** The challenge brief says the evaluator checks the base-image layers. The Dockerfile also installs Transformers and related packages with `--no-deps` to avoid pip replacing the ROCm-provided PyTorch with a CUDA build.

## Build

From the extracted project directory:

```bash
cd starter
docker build -t mc3-rag:latest .
```

The build downloads the model files into `/models/all-MiniLM-L6-v2` and sets `MC3_MODEL_PATH` accordingly. Verify the build logs complete successfully and the model weights exist in the resulting image. Do not submit an image that silently lacks the weights.

Check the model files in the built image:

```bash
docker run --rm --entrypoint bash mc3-rag:latest -lc \
  'ls -lh /models/all-MiniLM-L6-v2 && test -s /models/all-MiniLM-L6-v2/model.safetensors'
```

## Runtime interface

The evaluator invokes the program once to index, then starts a new process for each question. The cached index and embeddings must therefore be persisted to disk rather than held only in process memory.

### 1. Index the corpus

The evaluator mounts or supplies the corpus at `/app/corpus` and runs:

```bash
python3 /app/app.py --index /app/corpus
```

The index is stored under `/app/index` by default. Override this with `MC3_INDEX_DIR` if needed.

### 2. Answer a question

```bash
python3 /app/app.py \
  --corpus /app/corpus \
  --query-id query_01 \
  --query "What is the maximum junction temperature of the TQ-40?"
```

The output path must be `/app/output/query_01_output.json`. The query ID is supplied by the harness and must be preserved exactly.

Expected response shape:

```json
{
  "answer": "94",
  "citations": ["specs/tq40_datasheet_r2.pdf"],
  "confidence": 0.91
}
```

- `answer`: required string; return only the value. Use `""` if the corpus does not support an answer.
- `citations`: required list of corpus-relative file paths. Include only necessary evidence files, not every retrieved candidate.
- `confidence`: optional numeric estimate; the implementation writes it for diagnostics. It is not a substitute for evidence.

For an unanswerable question, the required shape is:

```json
{"answer":"","citations":[],"confidence":0.0}
```

The evaluator compares citations as an **exact set**. A correct answer with an irrelevant extra citation or a missing required source still fails that question. When an answer depends on a chain—such as a log identifying a ticket and a bug database mapping that ticket to a fixed firmware release—cite both files.

## Test locally

From the root `mc3-starter-kit` directory:

```bash
./setup.sh
python3 selfcheck.py mc3-rag:latest mc3-corpus
```

The self-check tests the container contract against the supplied corpus. It is not a replacement for testing the ten known sample questions and their exact citation sets. The hidden evaluation corpus is different and larger than the sample corpus.

### Check ROCm and actual GPU visibility

On an AMD ROCm host, run:

```bash
docker run --rm --network none \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  mc3-rag:latest python3 -c \
  "import torch; print('torch:', torch.__version__); print('GPU visible:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
```

ROCm exposes supported devices through PyTorch's `torch.cuda` API. A visible device is necessary but not sufficient: inspect runtime logs to confirm the embedding operations execute on the accelerator. CPU fallback is useful for local development, but it does **not** satisfy a requirement that the submitted run use a GPU.

### Verify offline operation

The evaluation environment may have no outbound network. After a successful build, run indexing and queries with `--network none` and confirm that no model download is attempted:

```bash
docker run --rm --network none mc3-rag:latest \
  python3 -c "from transformers import AutoTokenizer; print(AutoTokenizer.from_pretrained('/models/all-MiniLM-L6-v2', local_files_only=True))"
```

## Reliability and failure handling

- Empty corpus and empty subdirectories should not crash indexing.
- Unsupported extensions are skipped.
- Read and parse exceptions are caught per file so indexing can continue.
- Empty-password decryption is attempted for encrypted PDFs; files that remain unreadable contribute no evidence.
- Image OCR is bounded by a timeout and failures are treated as empty extraction.
- Output JSON must be written for every query, including refusals.

These safeguards are intended to satisfy the challenge's robustness cases. They should still be exercised in the built image with the same permissions and container capabilities used by the evaluator. In particular, test the unreadable-file case on Linux with the `DAC_OVERRIDE` capability dropped; running as root without that restriction can make a `chmod 000` file readable and invalidate the test.

## Performance considerations

The challenge brief specifies a 10-minute startup budget including model load and indexing, a 30-second budget per query, a 10-minute whole-run budget, and a 60 GiB uncompressed image-size ceiling. It also describes a VRAM ceiling of 48 GiB and a minimum of 1 GiB during the run. Precomputing embeddings during indexing avoids embedding the full corpus on each question. Nonetheless, actual timings and VRAM usage must be measured on the target hardware and the larger hidden corpus.

## Known limitations / pre-submission checklist

- [ ] Build successfully using the exact mandated ROCm base image.
- [ ] Confirm `model.safetensors` and all tokenizer/config files are present in the image.
- [ ] Confirm the installed `torch` version still contains the expected `+rocm` build identifier.
- [ ] Confirm the AMD GPU is visible and embedding inference actually uses it.
- [ ] Run the supplied self-check and all sample questions.
- [ ] Verify exact citation sets, especially multi-document questions and withdrawn/superseded revisions.
- [ ] Test unanswerable questions return an empty answer and no citations.
- [ ] Test empty corpus, empty directory, unreadable file, encrypted PDF, and unknown extension.
- [ ] Test repeated per-query processes and output-file naming.
- [ ] Test with network disabled after the image has been built.
- [ ] Measure startup, per-query latency, VRAM, and image size on the evaluation hardware.

**No hidden-set score or target-hardware GPU result is guaranteed by this repository.** The sample corpus is a development fixture, not the graded corpus. Validate the final image on the competition environment before submission.

## Challenge output example

Question: `What is the maximum junction temperature of the TQ-40?`

```json
{
  "answer": "94",
  "citations": ["specs/tq40_datasheet_r2.pdf"],
  "confidence": 0.91
}
```

Question: `What is the unit price of the TQ-40 at 10,000 unit volume?`

```json
{"answer":"","citations":[],"confidence":0.0}
```

The second example demonstrates the required abstention behavior when no accessible, readable source supports the requested value.

## License and attribution

This repository uses the starter kit and challenge contract supplied for AMD AI League Match 3. The embedding model is `sentence-transformers/all-MiniLM-L6-v2`; review its model card and license terms before redistributing a built image outside the challenge. Tesseract OCR and the Python parsing dependencies retain their respective licenses.
