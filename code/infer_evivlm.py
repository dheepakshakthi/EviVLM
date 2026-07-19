# -*- coding: utf-8 -*-
"""Inference and saliency visualization for EviVLM.

This script loads a trained EviVLM checkpoint, runs segmentation inference on
the requested split, and saves a paper-style visualization panel containing:

- Image
- Ground truth
- UNet branch prediction
- UNet+Text fused prediction
- Text-conditioned saliency overlay

The saliency map is computed from the gradient of the text-conditioned gain,
defined as mean(prob_VL - prob_V), with respect to the input image.
"""

import argparse
import glob
import json
import os
import random
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

import Config_SSL as config
from Load_Dataset_val_SSL import ImageToImage2D_val, ValGenerator
from nets.EviVLM import EviVLM
from nets.causal_reasoning import (
    PersistentKnowledgeGraph,
    build_reasoning_graph,
    export_reasoning_graph,
    summarize_reasoning,
)


def parse_args():
    parser = argparse.ArgumentParser(description="EviVLM inference with saliency visualization")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a trained EviVLM checkpoint (.pth.tar). If omitted, the latest best_model-EviVLM.pth.tar under ImageEncoder_Pretrain/EviVLM is used.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Dataset split folder, for example D:/VLM_Medical_Imaging/dataset/Test_Folder/.",
    )
    parser.add_argument(
        "--report-excel",
        type=str,
        default=None,
        help="Excel file mapping image filenames to report text.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save predictions and visualization panels.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Inference batch size. Saliency is easiest to inspect with batch size 1.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional limit on the number of samples to process.",
    )
    parser.add_argument(
        "--vision-backbone",
        type=str,
        default=config.vision_backbone,
        choices=["unet", "segformer", "segformer-b0"],
        help="Vision backbone architecture used by the checkpoint.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold for turning probability maps into binary masks.",
    )
    parser.add_argument(
        "--no-text",
        action="store_true",
        help="Disable text conditioning and run the model with empty texts.",
    )
    parser.add_argument(
        "--save-reasoning",
        action="store_true",
        help="Save SCM reasoning summaries plus full JSON/HTML/PNG graph files.",
    )
    parser.add_argument(
        "--save-aggregate-kg",
        action="store_true",
        help="Save an aggregate knowledge graph over all processed inference samples.",
    )
    return parser.parse_args()


def find_latest_checkpoint() -> Optional[str]:
    candidates = glob.glob(
        os.path.join(
            "ImageEncoder_Pretrain",
            "EviVLM",
            "**",
            "models",
            "best_model-EviVLM.pth.tar",
        ),
        recursive=True,
    )
    if not candidates:
        return None
    candidates.sort(key=lambda path: os.path.getmtime(path), reverse=True)
    return candidates[0]


def logger_info(message: str) -> None:
    print(message)


def ensure_dir(path: str) -> None:
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def to_uint8_image(image_tensor: torch.Tensor) -> np.ndarray:
    image = image_tensor.detach().cpu().numpy()
    image = np.transpose(image, (1, 2, 0))
    image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    return image


def to_uint8_mask(mask_tensor: torch.Tensor, threshold: float = 0.5) -> np.ndarray:
    mask = mask_tensor.detach().cpu().numpy()
    if mask.ndim == 3:
        mask = mask[0]
    mask = (mask >= threshold).astype(np.uint8) * 255
    return mask


def squeeze_prediction(pred_tensor: torch.Tensor) -> np.ndarray:
    pred = pred_tensor.detach().cpu().numpy()
    pred = np.squeeze(pred)
    if pred.ndim != 2:
        raise ValueError(f"Expected a 2D prediction map after squeeze, got shape {pred.shape}")
    pred = np.clip(pred, 0.0, 1.0)
    return pred


def normalize_map(map_tensor: torch.Tensor) -> np.ndarray:
    saliency = map_tensor.detach().cpu().numpy()
    saliency = saliency - saliency.min()
    denom = saliency.max() - saliency.min()
    if denom > 1e-8:
        saliency = saliency / denom
    else:
        saliency = np.zeros_like(saliency)
    return saliency


def make_heatmap_overlay(image_bgr: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    heatmap_uint8 = np.clip(heatmap * 255.0, 0, 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image_bgr, 1.0 - alpha, heatmap_color, alpha, 0)
    return overlay


def make_pred_overlay(image_bgr: np.ndarray, pred_prob: torch.Tensor, threshold: float = 0.5) -> np.ndarray:
    pred = squeeze_prediction(pred_prob)
    pred_uint8 = np.clip(pred * 255.0, 0, 255).astype(np.uint8)
    pred_color = cv2.applyColorMap(pred_uint8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image_bgr, 0.45, pred_color, 0.55, 0)
    binary = (pred >= threshold).astype(np.uint8) * 255
    contour_mask = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(overlay, 0.9, contour_mask, 0.1, 0)
    return overlay


def add_caption(tile: np.ndarray, caption: str, pad: int = 28) -> np.ndarray:
    if tile.ndim == 2:
        tile = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)
    h, w = tile.shape[:2]
    canvas = np.zeros((h + pad, w, 3), dtype=np.uint8)
    canvas[:h] = tile
    cv2.putText(
        canvas,
        caption,
        (10, h + int(pad * 0.72)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return canvas


def compose_panel(
    image_bgr: np.ndarray,
    gt_mask: np.ndarray,
    unet_pred: torch.Tensor,
    fused_pred: torch.Tensor,
    saliency_map: np.ndarray,
    threshold: float,
) -> np.ndarray:
    image_tile = add_caption(image_bgr, "Image")
    gt_tile = add_caption(cv2.cvtColor(gt_mask, cv2.COLOR_GRAY2BGR), "GT")
    unet_tile = add_caption(make_pred_overlay(image_bgr, unet_pred, threshold), "UNet")
    fused_tile = add_caption(make_pred_overlay(image_bgr, fused_pred, threshold), "UNet+Text")
    saliency_tile = add_caption(make_heatmap_overlay(image_bgr, saliency_map), "Saliency")
    panel = np.concatenate([image_tile, gt_tile, unet_tile, fused_tile, saliency_tile], axis=1)
    return panel


def load_model(checkpoint_path: str, device: torch.device, vision_backbone: str) -> EviVLM:
    model = EviVLM(
        n_channels=config.n_channels,
        n_classes=config.n_labels,
        vision_backbone=vision_backbone,
        segformer_model_name=config.segformer_model_name,
        segformer_pretrained=config.segformer_pretrained,
        freeze_segformer=config.freeze_segformer,
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    logger_info(f"Loaded checkpoint: {checkpoint_path}")
    if missing:
        logger_info(f"Missing keys: {missing}")
    if unexpected:
        logger_info(f"Unexpected keys: {unexpected}")
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def compute_saliency(
    model: EviVLM,
    images: torch.Tensor,
    texts: Sequence[str],
    device: torch.device,
    return_reasoning: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[Sequence[dict]], Optional[dict]]:
    images = images.to(device).float().detach().requires_grad_(True)
    with torch.enable_grad():
        if return_reasoning:
            (
                prob_V,
                prob_L,
                prob_VL,
                evi_V,
                evi_L,
                evi_VL,
                loss_sim,
                reasoning_outputs,
            ) = model(images, list(texts), return_reasoning=True)
            reasoning_summaries = summarize_reasoning(reasoning_outputs)
        else:
            prob_V, prob_L, prob_VL, evi_V, evi_L, evi_VL, loss_sim = model(images, list(texts))
            reasoning_summaries = None
            reasoning_outputs = None
        score = (prob_VL - prob_V).mean()
        grads = torch.autograd.grad(score, images, retain_graph=False, create_graph=False)[0]
    saliency = grads.abs().max(dim=1)[0]
    saliency_maps = []
    for i in range(saliency.shape[0]):
        saliency_maps.append(torch.from_numpy(normalize_map(saliency[i])))
    saliency_tensor = torch.stack(saliency_maps, dim=0)
    return (
        prob_V.detach(),
        prob_VL.detach(),
        loss_sim.detach(),
        saliency_tensor,
        reasoning_summaries,
        reasoning_outputs,
    )


def run_inference(args):
    dataset_path = args.dataset_path or config.val_dataset
    if not os.path.isdir(dataset_path):
        fallback = os.path.join("D:/VLM_Medical_Imaging/dataset", "Test_Folder")
        dataset_path = fallback if os.path.isdir(fallback) else config.val_dataset

    report_excel = args.report_excel
    if report_excel is None:
        default_report = "d:/VLM_Medical_Imaging/dataset/reports.xlsx"
        if os.path.exists(default_report):
            report_excel = default_report

    if args.checkpoint is not None:
        checkpoint_path = args.checkpoint
    else:
        checkpoint_path = find_latest_checkpoint()
        if checkpoint_path is None:
            raise FileNotFoundError(
                "Could not find a checkpoint. Pass --checkpoint explicitly or save one under ImageEncoder_Pretrain/EviVLM/**/models/best_model-EviVLM.pth.tar"
            )

    output_dir = args.output_dir or os.path.join(os.path.dirname(checkpoint_path), "inference_outputs")
    ensure_dir(output_dir)

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    model = load_model(checkpoint_path, device, args.vision_backbone)

    dataset = ImageToImage2D_val(
        dataset_path,
        "Infer",
        ValGenerator(output_size=[config.img_size, config.img_size]),
        image_size=config.img_size,
        report_excel=report_excel,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    logger_info(f"Dataset: {dataset_path}")
    logger_info(f"Reports: {report_excel if report_excel else 'none'}")
    logger_info(f"Output dir: {output_dir}")

    aggregate_kg = PersistentKnowledgeGraph() if args.save_aggregate_kg else None
    processed = 0
    for batch_idx, (sampled_batch, names) in enumerate(loader, 1):
        images = sampled_batch["image"]
        masks = sampled_batch["label"]
        texts = sampled_batch.get("text", [""] * images.size(0))
        if args.no_text:
            texts = [""] * images.size(0)

        (
            prob_V,
            prob_VL,
            loss_sim,
            saliency,
            reasoning_summaries,
            reasoning_outputs,
        ) = compute_saliency(
            model,
            images,
            texts,
            device,
            return_reasoning=args.save_reasoning,
        )
        if aggregate_kg is not None and reasoning_outputs is not None:
            aggregate_kg.update_from_reasoning(reasoning_outputs, sample_names=names)

        for i in range(images.size(0)):
            image_bgr = to_uint8_image(images[i])
            gt_mask = to_uint8_mask(masks[i], threshold=args.threshold)
            panel = compose_panel(
                image_bgr,
                gt_mask,
                prob_V[i],
                prob_VL[i],
                saliency[i].numpy(),
                args.threshold,
            )

            base_name = os.path.splitext(names[i])[0]
            panel_path = os.path.join(output_dir, f"{base_name}_panel.png")
            pred_path = os.path.join(output_dir, f"{base_name}_pred.png")
            saliency_path = os.path.join(output_dir, f"{base_name}_saliency.png")
            reasoning_path = os.path.join(output_dir, f"{base_name}_reasoning.json")
            graph_prefix = os.path.join(output_dir, f"{base_name}_reasoning_graph")

            cv2.imwrite(panel_path, panel)
            cv2.imwrite(pred_path, (squeeze_prediction(prob_VL[i]) * 255.0).astype(np.uint8))
            saliency_overlay = make_heatmap_overlay(image_bgr, saliency[i].numpy())
            cv2.imwrite(saliency_path, saliency_overlay)
            if reasoning_summaries is not None:
                with open(reasoning_path, "w", encoding="utf-8") as f:
                    json.dump(reasoning_summaries[i], f, indent=2)
            if reasoning_outputs is not None:
                graph = build_reasoning_graph(
                    reasoning_outputs,
                    sample_idx=i,
                    sample_name=names[i],
                )
                export_reasoning_graph(graph, graph_prefix)

            logger_info(
                f"Saved {base_name}: pred={pred_path}, panel={panel_path}, saliency={saliency_path}, sim_loss={loss_sim.item():.4f}"
            )

            processed += 1
            if args.max_samples is not None and processed >= args.max_samples:
                if aggregate_kg is not None:
                    aggregate_kg.save(os.path.join(output_dir, "aggregate_reasoning_kg"))
                return

    if aggregate_kg is not None:
        aggregate_kg.save(os.path.join(output_dir, "aggregate_reasoning_kg"))


def main():
    args = parse_args()
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    run_inference(args)


if __name__ == "__main__":
    main()
