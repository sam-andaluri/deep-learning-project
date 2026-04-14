# Deep Learning Systems

## Time-Series Demand Forecasting for LLM Inference Serving

This folder contains implementation of **Multi-Step Demand Forecasting using LSTM and Transformer Architectures** built on Microsoft's Azure LLM Inference Trace 2024 dataset (Stojkovic et al., HPCA '25).

GitHub repository can be found at `https://github.com/sam-andaluri/deep-learning-project`.

## Directory Structure


## 1. Install `uv`

Install `uv` with the standalone installer:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Or install it with Homebrew on macOS:

```bash
brew install uv
```

Confirm the installation:

```bash
uv --version
```

## 2. Prerequisites

Make sure the following tools are available on your system:

- Python 3.8 or higher
- Jupyter with `nbconvert`
- Pandoc for PDF generation

Quick checks:

```bash
python --version
python -m jupyter nbconvert --version
pandoc --version
```

## 3. Create and Activate a Virtual Environment

From the project folder:

```bash
cd deep-learning-project
uv venv .venv
source .venv/bin/activate
```

## 4. Install Dependencies

Install the pinned dependencies from `requirements.txt`:

```bash
uv pip install -r requirements.txt
```

## 5. Download the Dataset

Download the Azure LLM Inference Trace 2024 dataset:

```bash
mkdir -p data/azure_llm_2024
cd data/azure_llm_2024

# Code service trace (~660 MB)
curl -O https://azurepublicdatasettraces.blob.core.windows.net/azurellminfererencetrace/AzureLLMInferenceTrace_code_1week.csv

# Conversation service trace (~1.1 GB)
curl -O https://azurepublicdatasettraces.blob.core.windows.net/azurellminfererencetrace/AzureLLMInferenceTrace_conv_1week.csv

cd ../..
```

## 6. Run the Notebook

Open and run `deep_learning.ipynb` in Jupyter:

```bash
jupyter notebook deep_learning.ipynb
```

Or using JupyterLab:

```bash
jupyter lab deep_learning.ipynb
```

Run all cells from top to bottom (Cell > Run All).

## 7. Generate the Report PDF

If you update the report markdown, regenerate the PDF with:

```bash
pandoc Deep_Learning_Systems_Analysis_Report.md -o Deep_Learning_Systems_Analysis_Report.pdf
```

## 8. Generate Final requirements.txt

After running the notebook, generate the exact package versions:

```bash
pip freeze > requirements.txt
```

## References

Astral. (n.d.). *Installing uv*. uv documentation. Retrieved April 14, 2026, from https://docs.astral.sh/uv/getting-started/installation/

Astral. (n.d.). *Pip interface*. uv documentation. Retrieved April 14, 2026, from https://docs.astral.sh/uv/pip/

Astral. (n.d.). *Using environments*. uv documentation. Retrieved April 14, 2026, from https://docs.astral.sh/uv/pip/environments/

Microsoft Azure. (n.d.). *Azure LLM Inference Trace 2024* [Data set]. Retrieved April 14, 2026, from https://github.com/Azure/AzurePublicDataset

Stojkovic, U., et al. (2025). DynamoLLM: Designing LLM Inference Clusters for Performance and Energy Efficiency. *IEEE International Symposium on High-Performance Computer Architecture (HPCA '25)*.

Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735-1780.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. *Advances in Neural Information Processing Systems*, 30.
