# ProFAQLM

ProFAQLM is an instruction fine-tuning and export pipeline designed to train lightweight language models for academic document Q&A, structured exam answers, and citation-grounded retrieval.

- **Hugging Face Model & GGUF**: [park-bit/ProFAQLM-3B-Exam](https://huggingface.co/park-bit/ProFAQLM-3B-Exam)
- **Kaggle Training Notebook**: [parthbhuskade/profaqlm-ipynb](https://www.kaggle.com/code/parthbhuskade/profaqlm-ipynb)
- **GitHub Repository**: [park-bit/ProFAQLM](https://github.com/park-bit/ProFAQLM)

## Features

- **QLoRA Fine-Tuning**: 4-bit quantized parameter-efficient fine-tuning with PEFT, bitsandbytes, and TRL.
- **Structured Output Alignment**: Trained to generate clear Markdown hierarchy, formal definitions, inline citations (`[N]`), and comparative tables.
- **Dataset Schema**: Standard `(instruction, input, output)` JSONL schema compatible with academic datasets such as SciQ, HotpotQA, and MMLU.
- **GGUF Export Pipeline**: Merge LoRA adapters into base weights and convert to 4-bit quantized GGUF binaries for low-latency inference.
- **Ollama Integration**: Ready-to-use `Modelfile` for immediate deployment with the ProFAQ desktop application.

## Repository Structure

```text
ProFAQLM/
├── data/
│   └── example_dataset.jsonl      # Reference schema and sample records
├── notebooks/
│   └── ProFAQLM.ipynb             # Interactive training notebook (Kaggle / Colab)
├── Modelfile                      # Ollama runtime definition
├── requirements.txt               # Python package dependencies
├── LICENSE                        # MIT License
└── README.md
```

## Requirements

- Python 3.10 or higher
- NVIDIA GPU with CUDA support (recommended: 8GB+ VRAM for local training, or cloud environments such as Kaggle / Colab with 16GB VRAM)
- Optional: `llama.cpp` for GGUF compilation, `ollama` for local inference

## Installation

```bash
git clone https://github.com/park-bit/ProFAQLM.git
cd ProFAQLM

python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows PowerShell
.\.venv\Scripts\activate

pip install -r requirements.txt
```

## Dataset Schema

Datasets follow standard JSON Lines (`.jsonl`) format with `instruction`, `input`, and `output` keys:

```json
{
  "instruction": "Answer the question using structured academic format and cite sources using [N] notations.",
  "input": "Context: [1]: A system of n nodes tolerates up to f Byzantine failures when n >= 3f + 1.\n\nQuestion: What is the fault tolerance threshold?",
  "output": "## 1. Formal Definition\n\n**n >= 3f + 1** [1].\n\n## 2. Core Explanation\n\nA Byzantine fault tolerant system requires at least 3f + 1 nodes to reach consensus under f arbitrary failures [1]."
}
```

See `data/example_dataset.jsonl` for full examples.

## Training Workflow

Training is conducted interactively via Jupyter Notebook (in Kaggle or Google Colab):

1. **Dataset Ingestion**: Load and clean public academic question banks (e.g. `allenai/sciq`).
2. **Quantization & LoRA**: Load base model (`Qwen/Qwen2.5-3B-Instruct`) in 4-bit NF4 and attach LoRA adapter layers.
3. **Supervised Fine-Tuning**: Execute `SFTTrainer` with gradient checkpointing and sequence length of 2048 tokens.
4. **Adapter Merge**: Merge trained adapter weights into base model weights in FP16 precision.
5. **Quantization**: Convert merged model to 4-bit GGUF (`Q4_K_M`) using `llama.cpp`.

## Ollama Runtime Deployment

Register the exported 4-bit GGUF model in Ollama:

```bash
ollama create profaqlm -f Modelfile
```

Run test inference:

```bash
ollama run profaqlm "Explain consensus mechanisms in distributed systems with citations."
```

## License

MIT License. See LICENSE for details.
