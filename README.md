**Overview**
- **Repo**: instructions to train the multimodal EviVLM (vision + text) model using the dataset in `dataset/`.

**What I changed**
- **Dataset loader**: `code/Load_Dataset_val_SSL.py` now supports loading report text from an Excel file and attaches a `text` field to each sample.
- **New training script**: `code/train_evivlm.py` — trains `EviVLM` combining segmentation and cross-modal similarity loss. This script does not start training automatically; run it manually.
- **New inference script**: `code/infer_evivlm.py` — runs EviVLM inference and saves a panel with Image, GT, UNet, UNet+Text, and a saliency overlay for text interpretability.
- **Requirements**: added `openpyxl` to `requirements.txt` so pandas can read `.xlsx` files.

**Files to check**
- Dataset loader: [code/Load_Dataset_val_SSL.py](code/Load_Dataset_val_SSL.py)
- Multimodal model: [code/nets/EviVLM.py](code/nets/EviVLM.py)
- Text encoder: [code/backbones/bert_model/TextEncoder.py](code/backbones/bert_model/TextEncoder.py)
- New trainer: [code/train_evivlm.py](code/train_evivlm.py)
- Inference script: [code/infer_evivlm.py](code/infer_evivlm.py)

**Dataset (expected)**
- Place images and masks under:
  - `dataset/Train_Folder/img/` and `dataset/Train_Folder/labelcol/`
  - `dataset/Val_Folder/img/` and `dataset/Val_Folder/labelcol/`
- Provide a single Excel file mapping image filenames to text descriptions (one row per image). Example columns:
  - `image` (filename, e.g. `case001.png`)
  - `report` (text description)
- By default the trainer looks for `dataset/reports.xlsx` (absolute path `d:/VLM_Medical_Imaging/dataset/reports.xlsx`). You can pass a different path by editing `code/train_evivlm.py` or running it as a module with modifications.

**Text encoder assets**
- The text encoder now resolves paths relative to `code/backbones/bert_model/` instead of a hard-coded Linux path.
- For paper-style training, place the Bio-ClinicalBERT checkpoint files under `code/backbones/bert_model/Bio_ClinicalBERT/`, or set:
  - `BIO_CLINICAL_BERT_PATH`
  - `BIO_CLINICAL_BERT_NAME`
- If pretrained weights are not available, the code will fall back to a randomly initialized BERT-compatible model so the script can still run, but that is not equivalent to the paper setup.

**How to run (example)**
- Install dependencies (recommended inside your virtualenv `vlm_env`):

```bash
pip install -r requirements.txt
```

- Vision backbone selection lives in `code/Config_SSL.py`:
  - `vision_backbone = "segformer-b0"` uses `nvidia/segformer-b0-finetuned-ade-512-512` as a frozen SegFormer bottleneck projected into EviVLM.
  - `vision_backbone = "unet"` restores the original convolutional bottleneck.

- Run training (example):

```bash
python code/train_evivlm.py
```

- Run inference with saliency visualization (example):

```bash
python code/infer_evivlm.py --checkpoint ImageEncoder_Pretrain/EviVLM/Pretrain_EviVLM_05.25_14h42/models/best_model-EviVLM.pth.tar --dataset-path D:/VLM_Medical_Imaging/dataset/Test_Folder --report-excel D:/VLM_Medical_Imaging/dataset/reports.xlsx --output-dir D:/VLM_Medical_Imaging/inference_outputs
```

If the checkpoint was trained with the old visual path, add `--vision-backbone unet`.

Notes:
- `code/train_evivlm.py` will automatically use `d:/VLM_Medical_Imaging/dataset/reports.xlsx` if present. Otherwise it will run with empty texts (still trains segmentation only).
- `code/infer_evivlm.py` will use the text reports if the Excel file is available; otherwise it will still produce image-only saliency and prediction outputs.
- Tune `lambda_sim` in `train_evivlm.py` (default `0.1`) to balance segmentation and similarity loss.
- If you have a different Excel layout, open `code/Load_Dataset_val_SSL.py` and adjust the `report_sheet`, `report_col_image`, or `report_col_text` arguments when constructing the dataset.

**Next steps I can do for you**
- Add simple unit tests for dataset mapping and a small smoke test that runs one forward pass on CPU.

If you want me to proceed with any of those, tell me which one.

**Optional SCM reasoning branch**
- The model now has an optional structural-causal reasoning branch in `code/nets/causal_reasoning.py`.
- Existing calls still work unchanged:

```python
prob_V, prob_L, prob_VL, evi_V, evi_L, evi_VL, loss_sim = model(images, texts)
```

- To request reasoning outputs:

```python
prob_V, prob_L, prob_VL, evi_V, evi_L, evi_VL, loss_sim, reasoning = model(
    images,
    texts,
    return_reasoning=True,
)
```

- The reasoning output contains:
  - `clinical_graph`: concept nodes built from ClinicalBERT tokens.
  - `visual_graph`: region nodes built from the predicted mask and evidential uncertainty.
  - `causal_edges`: SCM-style concept-to-region causal strengths.
  - `attention_edges`: cross-attention-derived concept-to-region alignment.
  - `causal_consistency_loss`: optional training loss.
  - `reasoning_confidence`: per-sample reliability score.

- To train with the reasoning loss enabled:

```bash
python code/train_evivlm.py --enable-reasoning --lambda-reasoning 0.05
```

- To train with reasoning and save a persistent post-training knowledge graph:

```bash
python code/train_evivlm.py --enable-reasoning --lambda-reasoning 0.05 --save-kg
```

- This saves the aggregate KG under the training run folder:

```text
ImageEncoder_Pretrain/EviVLM/<run_name>/reasoning_kg/persistent_knowledge_graph.json
ImageEncoder_Pretrain/EviVLM/<run_name>/reasoning_kg/persistent_knowledge_graph.html
ImageEncoder_Pretrain/EviVLM/<run_name>/reasoning_kg/persistent_knowledge_graph.png
```

- To save reasoning summaries during inference:

```bash
python code/infer_evivlm.py --checkpoint path/to/checkpoint.pth.tar --save-reasoning
```

- With `--save-reasoning`, each inference case saves:

```text
<case>_reasoning.json
<case>_reasoning_graph.json
<case>_reasoning_graph.html
<case>_reasoning_graph.png
```

- To also save an aggregate KG over all processed inference cases:

```bash
python code/infer_evivlm.py --checkpoint path/to/checkpoint.pth.tar --save-reasoning --save-aggregate-kg
```

- This saves:

```text
aggregate_reasoning_kg.json
aggregate_reasoning_kg.html
aggregate_reasoning_kg.png
```
