"""Fine-tune vllm-sr/d3-edge decision readout model on Indonesian legal trajectories."""

from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
import tempfile
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from huggingface_hub import HfApi, snapshot_download
from peft import LoraConfig, get_peft_model
from safetensors.torch import save_file
from transformers import AutoModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = BASE_DIR / "data" / "d3_navigation_train.jsonl"
DEFAULT_OUTPUT_DIR = BASE_DIR / "models" / "d3-edge-indo-legal"
BASE_MODEL_NAME = "vllm-sr/d3-edge"
DEFAULT_HUB_MODEL_ID = "ersaasyhar/d3-edge-indo-legal"


def load_dataset_records(dataset_path: Path) -> list[dict[str, Any]]:
    """Load JSONL trajectory dataset from local disk or Hugging Face Hub."""
    if not dataset_path.exists():
        logger.info("Local dataset not found at %s. Downloading from Hub...", dataset_path)
        from huggingface_hub import hf_hub_download

        cached_file = hf_hub_download(
            repo_id="ersaasyhar/d3-indo-legal-navigation",
            filename="train.jsonl",
            repo_type="dataset",
        )
        dataset_path = Path(cached_file)

    with open(dataset_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    logger.info("Loaded %d trajectory samples from %s", len(records), dataset_path)
    return records


def split_records(
    records: list[dict[str, Any]], val_ratio: float = 0.1, seed: int = 42
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split records into training and validation sets."""
    rng = random.Random(seed)
    shuffled = records.copy()
    rng.shuffle(shuffled)
    val_size = max(1, int(len(shuffled) * val_ratio))
    val_set = shuffled[:val_size]
    train_set = shuffled[val_size:]
    logger.info("Split dataset: %d train, %d validation", len(train_set), len(val_set))
    return train_set, val_set


def prepare_batch(
    samples: list[dict[str, Any]], model: Any, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    """Render chat prompt, left-pad token sequence, and produce target option indices."""
    sequences: list[list[int]] = []
    target_indices: list[int] = []
    option_counts: list[int] = []

    for s in samples:
        q = {
            "type": "choice",
            "instructions": s["instructions"],
            "criteria": s["criteria"],
        }
        text = model.runtime.text(s["state"], q)
        seq = model.runtime.tokenizer([text], add_special_tokens=False)["input_ids"][0]
        opts = list(s["criteria"].keys())
        target_idx = opts.index(s["target_choice"])

        sequences.append(seq)
        target_indices.append(target_idx)
        option_counts.append(len(opts))

    # ponytail: left-padding manual loop fast enough for batch_size <= 16; use torch.nested if batch > 64
    max_len = max(len(seq) for seq in sequences)
    pad_id = model.runtime.pad_id
    batch_size = len(sequences)

    input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)

    for i, seq in enumerate(sequences):
        start = max_len - len(seq)
        input_ids[i, start:] = torch.as_tensor(seq, dtype=torch.long)
        attention_mask[i, start:] = 1

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    target_tensor = torch.tensor(target_indices, dtype=torch.long, device=device)

    return input_ids, attention_mask, target_tensor, option_counts


def compute_decision_loss(
    backbone: torch.nn.Module,
    readout: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    target_indices: torch.Tensor,
    option_counts: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward pass through backbone, linear readout projection, and masked cross-entropy."""
    out = backbone(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    hidden = out.last_hidden_state[:, -1]
    # ponytail: FP32 projection matches d3_runtime.py FP32 readout logic; use bf16 if VRAM is tight
    logits = hidden.float() @ readout.T

    # Mask options beyond criteria size for each sample
    for i, count in enumerate(option_counts):
        logits[i, count:] = -1e9

    loss = F.cross_entropy(logits, target_indices)
    preds = logits.argmax(dim=-1)
    acc = (preds == target_indices).float().mean()
    return loss, acc


def evaluate_model(
    val_set: list[dict[str, Any]],
    model: Any,
    backbone: torch.nn.Module,
    device: torch.device,
    batch_size: int = 4,
) -> tuple[float, float]:
    """Evaluate mean cross-entropy loss and choice accuracy on validation set."""
    backbone.eval()
    readout = model.runtime.readout.to(device)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for start_idx in range(0, len(val_set), batch_size):
            batch = val_set[start_idx : start_idx + batch_size]
            ids, mask, targets, counts = prepare_batch(batch, model, device)
            loss, acc = compute_decision_loss(backbone, readout, ids, mask, targets, counts)
            total_loss += loss.item() * len(batch)
            total_correct += int(round(acc.item() * len(batch)))
            total_samples += len(batch)

    mean_loss = total_loss / max(1, total_samples)
    mean_acc = total_correct / max(1, total_samples)
    return mean_loss, mean_acc


def generate_model_card(
    hub_model_id: str,
    base_model: str,
    val_acc: float,
    elbow_epoch: int = 1,
    val_loss: float = 0.0,
    has_curve_plot: bool = False,
) -> str:
    """Generate Hugging Face model card README.md with standard YAML metadata."""
    plot_md = (
        "\n## Elbow Method Convergence Curve\n![D3 Elbow Curve](d3_elbow_curve.png)\n"
        if has_curve_plot
        else ""
    )
    return f"""---
base_model: {base_model}
language:
- id
license: apache-2.0
tags:
- zero-shot-classification
- text-classification
- decision-model
- system-one
- indonesian
- legal
- rag
- d3-edge
pipeline_tag: zero-shot-classification
---

# {hub_model_id}

Fine-tuned decision model based on [{base_model}](https://huggingface.co/{base_model}) (0.6B) specialized for Indonesian statutory tree navigation trajectories (`root_to_bab` and `bab_to_pasal`).

## Model Details
- **Base Architecture**: Qwen3.5 0.6B with 255-way linear readout (`readout.safetensors`).
- **Domain**: Indonesian legal tree navigation (Undang-Undang, Peraturan Pemerintah).
- **Optimization**: 4-Epoch Elbow Method (Optimal Knee Point: Epoch {elbow_epoch}).
- **Validation Accuracy**: {val_acc * 100:.2f}% on held-out statutory navigation decisions.
- **Validation Loss**: {val_loss:.4f}.
- **Training Data**: [`ersaasyhar/d3-indo-legal-navigation`](https://huggingface.co/datasets/ersaasyhar/d3-indo-legal-navigation).
{plot_md}
## Usage with D3Navigator

```python
from src.tree_reasoning.d3_navigator import D3Navigator

navigator = D3Navigator(model_name="{hub_model_id}")
results = navigator.retrieve("Apa syarat pendirian perseroan terbatas?", top_k=5)
for chunk in results:
    print(f"{{chunk.doc_id}} - {{chunk.title}} (p. {{chunk.page_number}})")
```
"""


def save_elbow_plot(
    history: dict[str, list[float]],
    elbow_epoch: int,
    output_path: Path,
) -> Path | None:
    """Generate dual-panel elbow loss and accuracy curve visualization."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed, skipping elbow curve plot.")
        return None

    epochs = len(history["epochs"]) - 1
    elbow_val_loss = history["val_loss"][elbow_epoch]
    elbow_val_acc = history["val_acc"][elbow_epoch]
    zero_acc = history["val_acc"][0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Subplot 1: Loss trajectory & knee detection
    ax1.plot(
        history["epochs"][1:],
        history["train_loss"][1:],
        "b--o",
        label="Training Loss",
        linewidth=2,
        markersize=6,
    )
    ax1.plot(
        history["epochs"],
        history["val_loss"],
        "r-s",
        label="Validation Loss",
        linewidth=2,
        markersize=6,
    )
    ax1.axvline(
        x=elbow_epoch,
        color="gray",
        linestyle=":",
        linewidth=2,
        label=f"Elbow Epoch ({elbow_epoch})",
    )
    if elbow_epoch < epochs:
        ax1.axvspan(elbow_epoch, epochs, color="gray", alpha=0.12, label="Diminishing Returns Zone")
    ax1.scatter(
        [elbow_epoch],
        [elbow_val_loss],
        color="red",
        s=140,
        zorder=5,
        edgecolors="black",
        linewidth=1.5,
    )
    ax1.annotate(
        f"Elbow Knee Point\nEpoch {elbow_epoch} (Loss: {elbow_val_loss:.4f})",
        xy=(elbow_epoch, elbow_val_loss),
        xytext=(elbow_epoch + 0.15, elbow_val_loss + 0.25),
        arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff2cc", edgecolor="#d6b656"),
        fontsize=9,
        fontweight="bold",
    )
    ax1.set_xlabel("Epoch", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Cross-Entropy Loss", fontsize=11, fontweight="bold")
    ax1.set_title("D3 Decision Readout Loss Trajectory", fontsize=12, fontweight="bold")
    ax1.set_xticks(range(0, epochs + 1))
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="upper right", fontsize=9)

    # Subplot 2: Validation Accuracy Progression
    val_acc_pct = [acc * 100 for acc in history["val_acc"]]
    ax2.plot(
        history["epochs"],
        val_acc_pct,
        "g-^",
        label="Validation Accuracy",
        linewidth=2,
        markersize=7,
    )
    ax2.axhline(
        y=zero_acc * 100,
        color="gray",
        linestyle="--",
        alpha=0.7,
        label=f"Zero-Shot Baseline ({zero_acc * 100:.1f}%)",
    )
    ax2.axvline(x=elbow_epoch, color="gray", linestyle=":", linewidth=2)
    ax2.scatter(
        [elbow_epoch],
        [elbow_val_acc * 100],
        color="green",
        s=140,
        zorder=5,
        edgecolors="black",
        linewidth=1.5,
    )
    ax2.annotate(
        f"Elbow Peak Acc\n{elbow_val_acc * 100:.2f}%",
        xy=(elbow_epoch, elbow_val_acc * 100),
        xytext=(
            elbow_epoch - 0.7 if elbow_epoch > 1 else elbow_epoch + 0.2,
            elbow_val_acc * 100 - 6,
        ),
        arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#d5e8d4", edgecolor="#82b366"),
        fontsize=9,
        fontweight="bold",
    )
    ax2.set_xlabel("Epoch", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Validation Accuracy (%)", fontsize=11, fontweight="bold")
    ax2.set_title("D3 Navigation Validation Accuracy Progression", fontsize=12, fontweight="bold")
    ax2.set_xticks(range(0, epochs + 1))
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="lower right", fontsize=9)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info("Saved elbow curve visualization to %s", output_path)
    return output_path


def merge_and_upload_model(
    peft_backbone: Any,
    base_model_name: str,
    hub_model_id: str,
    val_acc: float,
    elbow_epoch: int = 1,
    val_loss: float = 0.0,
    elbow_curve_path: Path | None = None,
) -> None:
    """Merge LoRA weights into backbone and upload complete D3 model to Hugging Face Hub."""
    logger.info("Merging LoRA weights back into base backbone...")
    merged_backbone = peft_backbone.merge_and_unload()

    staging_dir = Path(tempfile.mkdtemp(prefix="d3_merge_"))
    try:
        logger.info("Downloading base repo scaffold from %s...", base_model_name)
        # Download scaffold without base backbone weights or video artifacts
        snapshot_download(
            base_model_name,
            local_dir=staging_dir,
            ignore_patterns=["model-*.safetensors", "*.mp4", "MODEL_MANIFEST.json"],
        )

        # Remove manifest or pycache if present
        manifest_file = staging_dir / "MODEL_MANIFEST.json"
        if manifest_file.exists():
            manifest_file.unlink()
        for pycache in staging_dir.rglob("__pycache__"):
            shutil.rmtree(pycache, ignore_errors=True)

        logger.info("Saving merged backbone weights to model-00001-of-00001.safetensors...")
        weight_file = staging_dir / "model-00001-of-00001.safetensors"
        save_file(merged_backbone.state_dict(), weight_file)

        has_curve = False
        if elbow_curve_path and elbow_curve_path.exists():
            shutil.copy2(elbow_curve_path, staging_dir / "d3_elbow_curve.png")
            has_curve = True
            logger.info(
                "Copied elbow curve plot to staging dir: %s", staging_dir / "d3_elbow_curve.png"
            )

        logger.info("Writing model card README.md...")
        readme_file = staging_dir / "README.md"
        readme_file.write_text(
            generate_model_card(
                hub_model_id,
                base_model_name,
                val_acc,
                elbow_epoch=elbow_epoch,
                val_loss=val_loss,
                has_curve_plot=has_curve,
            ),
            encoding="utf-8",
        )

        logger.info("Sanity-checking merged model loading from staging dir...")
        test_model = AutoModel.from_pretrained(str(staging_dir), trust_remote_code=True)
        assert test_model is not None, "Merged model failed to load"
        logger.info("Sanity check passed!")

        logger.info("Uploading merged model to Hugging Face Hub: %s...", hub_model_id)
        api = HfApi()
        api.create_repo(repo_id=hub_model_id, repo_type="model", exist_ok=True)
        api.upload_folder(
            folder_path=str(staging_dir),
            repo_id=hub_model_id,
            repo_type="model",
            commit_message=f"Fine-tune d3-edge on Indo legal navigation via elbow method (Epoch {elbow_epoch}, Val Acc: {val_acc * 100:.1f}%)",
        )
        logger.info("Successfully pushed model to: https://huggingface.co/%s", hub_model_id)
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)


def train_d3(
    dataset_path: Path = DEFAULT_DATASET,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    base_model_name: str = BASE_MODEL_NAME,
    hub_model_id: str = DEFAULT_HUB_MODEL_ID,
    epochs: int = 4,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    lr: float = 2e-4,
    weight_decay: float = 0.01,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    device_str: str | None = None,
    dry_run: bool = False,
    plot_elbow: bool = True,
    merge_and_upload: bool = False,
) -> None:
    if device_str is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    logger.info("Using device: %s", device)

    logger.info("Loading base D3 model from %s...", base_model_name)
    model = AutoModel.from_pretrained(base_model_name, trust_remote_code=True)
    backbone = model.runtime.backbone.to(device)
    readout = model.runtime.readout.to(device)

    logger.info("Applying LoRA to backbone...")
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=lora_dropout,
        bias="none",
    )
    peft_backbone = get_peft_model(backbone, lora_config)
    peft_backbone.print_trainable_parameters()

    records = load_dataset_records(dataset_path)
    train_set, val_set = split_records(records, val_ratio=0.1, seed=42)

    if dry_run:
        logger.info("Running dry-run smoke test (1 batch)...")
        sample_batch = train_set[:batch_size]
        ids, mask, targets, counts = prepare_batch(sample_batch, model, device)
        loss, acc = compute_decision_loss(peft_backbone, readout, ids, mask, targets, counts)
        logger.info("Dry run succeeded! Loss: %.4f, Acc: %.2f", loss.item(), acc.item())
        return

    logger.info("Computing zero-shot baseline on validation set...")
    val_loss, val_acc = evaluate_model(val_set, model, peft_backbone, device, batch_size=batch_size)
    logger.info("Zero-shot Validation - Loss: %.4f, Acc: %.2f%%", val_loss, val_acc * 100)

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    best_adapter_dir = output_dir / "lora_adapter"

    history: dict[str, list[float]] = {
        "epochs": [0],
        "train_loss": [float("nan")],
        "val_loss": [val_loss],
        "val_acc": [val_acc],
    }

    optimizer = torch.optim.AdamW(peft_backbone.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(1, epochs + 1):
        peft_backbone.train()
        random.shuffle(train_set)
        total_train_loss = 0.0
        step_count = 0
        optimizer.zero_grad()

        for start_idx in range(0, len(train_set), batch_size):
            batch = train_set[start_idx : start_idx + batch_size]
            ids, mask, targets, counts = prepare_batch(batch, model, device)
            loss, _ = compute_decision_loss(peft_backbone, readout, ids, mask, targets, counts)
            scaled_loss = loss / gradient_accumulation_steps
            scaled_loss.backward()

            total_train_loss += loss.item() * len(batch)
            step_count += 1

            if step_count % gradient_accumulation_steps == 0 or start_idx + batch_size >= len(
                train_set
            ):
                optimizer.step()
                optimizer.zero_grad()

        avg_train_loss = total_train_loss / len(train_set)
        val_loss, val_acc = evaluate_model(
            val_set, model, peft_backbone, device, batch_size=batch_size
        )
        history["epochs"].append(epoch)
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        epoch_adapter_dir = checkpoints_dir / f"adapter_epoch_{epoch}"
        epoch_adapter_dir.mkdir(parents=True, exist_ok=True)
        peft_backbone.save_pretrained(str(epoch_adapter_dir))

        delta_loss = val_loss - history["val_loss"][epoch - 1]
        delta_acc = (val_acc - history["val_acc"][epoch - 1]) * 100
        logger.info(
            "Epoch %d/%d - Train Loss: %.4f | Val Loss: %.4f (%+.4f) | Val Acc: %.2f%% (%+.2f%%)",
            epoch,
            epochs,
            avg_train_loss,
            val_loss,
            delta_loss,
            val_acc * 100,
            delta_acc,
        )

    # Detect elbow knee point (argmin val_loss over trained epochs, tie-break max val_acc)
    # ponytail: discrete argmin over trained epochs, add continuous knee curvature if step size changes
    cand_epochs = list(range(1, epochs + 1))
    elbow_epoch = min(cand_epochs, key=lambda e: (history["val_loss"][e], -history["val_acc"][e]))
    elbow_val_loss = history["val_loss"][elbow_epoch]
    elbow_val_acc = history["val_acc"][elbow_epoch]

    logger.info("=" * 68)
    logger.info(
        f"{'Epoch':<8}{'Train Loss':<14}{'Val Loss':<12}{'Val Acc (%)':<14}{'Delta Loss':<12}{'Status'}"
    )
    logger.info("-" * 68)
    for e in range(len(history["epochs"])):
        ep = history["epochs"][e]
        tr_l = f"{history['train_loss'][e]:.4f}" if e > 0 else "N/A"
        vl_l = f"{history['val_loss'][e]:.4f}"
        vl_a = f"{history['val_acc'][e] * 100:.2f}%"
        delta = f"{history['val_loss'][e] - history['val_loss'][e - 1]:+.4f}" if e > 0 else "-"
        marker = "<-- [ELBOW KNEE POINT]" if ep == elbow_epoch else ("Baseline" if ep == 0 else "")
        logger.info(f"{ep:<8}{tr_l:<14}{vl_l:<12}{vl_a:<14}{delta:<12}{marker}")
    logger.info("=" * 68)
    logger.info(
        "Selected Elbow Knee Point: Epoch %d (Val Loss: %.4f, Val Acc: %.2f%%)",
        elbow_epoch,
        elbow_val_loss,
        elbow_val_acc * 100,
    )

    shutil.rmtree(best_adapter_dir, ignore_errors=True)
    shutil.copytree(checkpoints_dir / f"adapter_epoch_{elbow_epoch}", best_adapter_dir)
    logger.info("Copied elbow adapter (epoch %d) to %s", elbow_epoch, best_adapter_dir)

    curve_path: Path | None = None
    if plot_elbow:
        curve_path = output_dir / "d3_elbow_curve.png"
        save_elbow_plot(history, elbow_epoch, curve_path)

    if merge_and_upload:
        merge_and_upload_model(
            peft_backbone=peft_backbone,
            base_model_name=base_model_name,
            hub_model_id=hub_model_id,
            val_acc=elbow_val_acc,
            elbow_epoch=elbow_epoch,
            val_loss=elbow_val_loss,
            elbow_curve_path=curve_path,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune vllm-sr/d3-edge decision model.")
    parser.add_argument(
        "--dataset", type=Path, default=DEFAULT_DATASET, help="Trajectory JSONL path"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Checkpoint output directory"
    )
    parser.add_argument("--base-model", default=BASE_MODEL_NAME, help="Base Hugging Face model ID")
    parser.add_argument(
        "--hub-model-id", default=DEFAULT_HUB_MODEL_ID, help="Target Hub model ID for upload"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=4,
        help="Number of training epochs (default: 4 for elbow method)",
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size per step")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--device", default=None, help="Device to use (cuda or cpu)")
    parser.add_argument("--dry-run", action="store_true", help="Run 1-batch smoke test")
    parser.add_argument(
        "--plot-elbow",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate and save elbow loss/accuracy curve plot",
    )
    parser.add_argument(
        "--merge-and-upload", action="store_true", help="Merge LoRA and push to Hub"
    )
    args = parser.parse_args()

    train_d3(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        base_model_name=args.base_model,
        hub_model_id=args.hub_model_id,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        lr=args.lr,
        device_str=args.device,
        dry_run=args.dry_run,
        plot_elbow=args.plot_elbow,
        merge_and_upload=args.merge_and_upload,
    )


if __name__ == "__main__":
    main()
