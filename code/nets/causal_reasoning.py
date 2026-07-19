import math
import json
import html as html_lib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ReasoningConfig:
    embedding_dim: int = 512
    hidden_dim: int = 256
    region_grid: int = 3
    min_region_mass: float = 0.03


class ClinicalConceptGraphBuilder:
    """Build compact clinical concept nodes from ClinicalBERT word tokens."""

    CONCEPT_TERMS: Dict[str, Tuple[str, ...]] = {
        "opacity": ("opacity", "opacities", "opaque", "consolidation"),
        "infection": ("infection", "infectious", "pneumonia", "inflammatory"),
        "lesion": ("lesion", "nodule", "mass", "tumor", "tumour"),
        "edema": ("edema", "oedema", "fluid"),
        "effusion": ("effusion", "pleural"),
        "atelectasis": ("atelectasis", "collapse"),
        "left": ("left",),
        "right": ("right",),
        "upper": ("upper", "apical"),
        "middle": ("middle", "central"),
        "lower": ("lower", "basal", "base"),
        "severity": ("mild", "moderate", "severe", "large", "small"),
    }

    def __init__(self, concept_names: Optional[Sequence[str]] = None):
        if concept_names is None:
            concept_names = tuple(self.CONCEPT_TERMS.keys())
        self.concept_names = tuple(concept_names)

    @property
    def num_concepts(self) -> int:
        return len(self.concept_names)

    def build(
        self,
        word_emb: torch.Tensor,
        token_words: Sequence[Sequence[str]],
    ) -> Dict[str, object]:
        device = word_emb.device
        batch_size, token_count, emb_dim = word_emb.shape
        concept_mask = torch.zeros(
            batch_size,
            self.num_concepts,
            token_count,
            dtype=word_emb.dtype,
            device=device,
        )

        token_concepts: List[List[List[str]]] = []
        stored_token_words: List[List[str]] = []
        for batch_idx, words in enumerate(token_words):
            sample_concepts: List[List[str]] = []
            aligned_words = list(words)[:token_count]
            aligned_words += ["[PAD]"] * max(0, token_count - len(aligned_words))
            stored_token_words.append(aligned_words)
            for token_idx, word in enumerate(aligned_words):
                normalized = self._normalize_token(word)
                matched_names: List[str] = []
                for concept_idx, concept_name in enumerate(self.concept_names):
                    if self._matches_concept(normalized, concept_name):
                        concept_mask[batch_idx, concept_idx, token_idx] = 1.0
                        matched_names.append(concept_name)
                sample_concepts.append(matched_names)
            token_concepts.append(sample_concepts)

        counts = concept_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        concept_nodes = torch.bmm(concept_mask, word_emb) / counts
        concept_presence = (concept_mask.sum(dim=-1) > 0).to(word_emb.dtype)

        return {
            "concept_names": self.concept_names,
            "concept_nodes": concept_nodes,
            "concept_presence": concept_presence,
            "token_concept_mask": concept_mask,
            "token_concepts": token_concepts,
            "token_words": stored_token_words,
        }

    def _normalize_token(self, token: str) -> str:
        return token.lower().replace("##", "").strip(".,;:()[]{}")

    def _matches_concept(self, token: str, concept_name: str) -> bool:
        if token in {"", "[pad]", "[sep]", "[cls]"}:
            return False
        terms = self.CONCEPT_TERMS.get(concept_name, (concept_name,))
        return any(token == term or token.startswith(term) for term in terms)


class VisualRegionGraphBuilder(nn.Module):
    """Convert mask/evidence/image features into a small visual region graph."""

    def __init__(self, config: ReasoningConfig):
        super().__init__()
        self.config = config

    @property
    def num_regions(self) -> int:
        return self.config.region_grid * self.config.region_grid

    @property
    def region_names(self) -> Tuple[str, ...]:
        if self.config.region_grid == 3:
            rows = ("upper", "middle", "lower")
            cols = ("left", "center", "right")
            return tuple(f"{row}_{col}" for row in rows for col in cols)
        return tuple(f"region_{idx}" for idx in range(self.num_regions))

    def forward(
        self,
        patch_emb: torch.Tensor,
        seg_prob: torch.Tensor,
        evidence: Optional[torch.Tensor],
    ) -> Dict[str, object]:
        batch_size, patch_count, emb_dim = patch_emb.shape
        patch_side = int(math.sqrt(patch_count))
        if patch_side * patch_side != patch_count:
            raise ValueError("patch_emb must contain a square patch grid.")

        region_grid = self.config.region_grid
        region_patch = patch_emb.permute(0, 2, 1).view(
            batch_size, emb_dim, patch_side, patch_side
        )
        region_nodes = F.adaptive_avg_pool2d(
            region_patch,
            output_size=(region_grid, region_grid),
        )
        region_nodes = region_nodes.flatten(2).permute(0, 2, 1)

        region_mass_map = F.adaptive_avg_pool2d(
            seg_prob.float(),
            output_size=(region_grid, region_grid),
        )
        region_mass = region_mass_map.flatten(1)
        region_presence = (
            region_mass > self.config.min_region_mass
        ).to(seg_prob.dtype)

        uncertainty_map = self._evidence_to_uncertainty(evidence, seg_prob)
        region_uncertainty = F.adaptive_avg_pool2d(
            uncertainty_map,
            output_size=(region_grid, region_grid),
        ).flatten(1)
        region_reliability = 1.0 - region_uncertainty.clamp(0.0, 1.0)

        return {
            "region_names": self.region_names,
            "region_nodes": region_nodes,
            "region_mass": region_mass,
            "region_presence": region_presence,
            "region_uncertainty": region_uncertainty,
            "region_reliability": region_reliability,
        }

    def _evidence_to_uncertainty(
        self,
        evidence: Optional[torch.Tensor],
        seg_prob: torch.Tensor,
    ) -> torch.Tensor:
        if evidence is None:
            return 1.0 - (seg_prob.float() - 0.5).abs() * 2.0

        batch_size, _, height, width = seg_prob.shape
        expected_pixels = batch_size * height * width
        if evidence.ndim != 2 or evidence.shape[0] != expected_pixels:
            return 1.0 - (seg_prob.float() - 0.5).abs() * 2.0

        evidence_strength = evidence.float().clamp_min(1e-6)
        uncertainty = 2.0 / evidence_strength.sum(dim=-1)
        return uncertainty.view(batch_size, 1, height, width).clamp(0.0, 1.0)


class SCMReasoningHead(nn.Module):
    """Lightweight structural-causal reasoning head for EviVLM outputs."""

    def __init__(
        self,
        config: Optional[ReasoningConfig] = None,
        concept_names: Optional[Sequence[str]] = None,
    ):
        super().__init__()
        self.config = config or ReasoningConfig()
        self.clinical_graph_builder = ClinicalConceptGraphBuilder(concept_names)
        self.visual_graph_builder = VisualRegionGraphBuilder(self.config)

        emb_dim = self.config.embedding_dim
        hidden_dim = self.config.hidden_dim
        self.concept_projection = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.region_projection = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.edge_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        word_emb: torch.Tensor,
        token_words: Sequence[Sequence[str]],
        patch_emb: torch.Tensor,
        atten_scores: Optional[torch.Tensor],
        seg_prob: torch.Tensor,
        evidence: Optional[torch.Tensor] = None,
    ) -> Dict[str, object]:
        clinical_graph = self.clinical_graph_builder.build(word_emb, token_words)
        visual_graph = self.visual_graph_builder(patch_emb, seg_prob, evidence)

        concept_nodes = clinical_graph["concept_nodes"]
        concept_presence = clinical_graph["concept_presence"]
        region_nodes = visual_graph["region_nodes"]
        region_presence = visual_graph["region_presence"]
        region_reliability = visual_graph["region_reliability"]
        region_mass = visual_graph["region_mass"]

        concept_latent = self.concept_projection(concept_nodes)
        region_latent = self.region_projection(region_nodes)

        concept_count = concept_latent.shape[1]
        region_count = region_latent.shape[1]
        concept_pair = concept_latent.unsqueeze(2).expand(
            -1, -1, region_count, -1
        )
        region_pair = region_latent.unsqueeze(1).expand(
            -1, concept_count, -1, -1
        )
        region_meta = torch.stack(
            (region_mass, region_reliability),
            dim=-1,
        ).unsqueeze(1).expand(-1, concept_count, -1, -1)
        pair_features = torch.cat((concept_pair, region_pair, region_meta), dim=-1)

        causal_edges = torch.sigmoid(self.edge_scorer(pair_features).squeeze(-1))
        causal_edges = causal_edges * concept_presence.unsqueeze(-1)
        causal_edges = causal_edges * region_presence.unsqueeze(1)

        attention_edges = self._attention_to_region_edges(
            atten_scores=atten_scores,
            token_concept_mask=clinical_graph["token_concept_mask"],
            region_grid=self.config.region_grid,
        )
        causal_consistency_loss = self._causal_consistency_loss(
            causal_edges,
            attention_edges,
            concept_presence,
            region_presence,
        )
        reasoning_confidence = self._reasoning_confidence(
            causal_edges,
            concept_presence,
            region_presence,
            region_reliability,
        )

        return {
            "clinical_graph": clinical_graph,
            "visual_graph": visual_graph,
            "causal_edges": causal_edges,
            "attention_edges": attention_edges,
            "causal_consistency_loss": causal_consistency_loss,
            "reasoning_confidence": reasoning_confidence,
        }

    def _attention_to_region_edges(
        self,
        atten_scores: Optional[torch.Tensor],
        token_concept_mask: torch.Tensor,
        region_grid: int,
    ) -> Optional[torch.Tensor]:
        if atten_scores is None:
            return None

        batch_size, patch_count, token_count = atten_scores.shape
        patch_side = int(math.sqrt(patch_count))
        if patch_side * patch_side != patch_count:
            return None

        token_region_attention = atten_scores.permute(0, 2, 1).view(
            batch_size,
            token_count,
            patch_side,
            patch_side,
        )
        token_region_attention = F.adaptive_avg_pool2d(
            token_region_attention,
            output_size=(region_grid, region_grid),
        ).flatten(2)

        concept_token_counts = token_concept_mask.sum(dim=-1).clamp_min(1.0)
        concept_region_attention = torch.bmm(
            token_concept_mask,
            token_region_attention,
        )
        concept_region_attention = (
            concept_region_attention / concept_token_counts.unsqueeze(-1)
        )
        return concept_region_attention

    def _causal_consistency_loss(
        self,
        causal_edges: torch.Tensor,
        attention_edges: Optional[torch.Tensor],
        concept_presence: torch.Tensor,
        region_presence: torch.Tensor,
    ) -> torch.Tensor:
        if attention_edges is None:
            return causal_edges.new_tensor(0.0)

        mask = concept_presence.unsqueeze(-1) * region_presence.unsqueeze(1)
        denom = mask.sum().clamp_min(1.0)
        diff = (causal_edges - attention_edges.detach()) ** 2
        return (diff * mask).sum() / denom

    def _reasoning_confidence(
        self,
        causal_edges: torch.Tensor,
        concept_presence: torch.Tensor,
        region_presence: torch.Tensor,
        region_reliability: torch.Tensor,
    ) -> torch.Tensor:
        mask = concept_presence.unsqueeze(-1) * region_presence.unsqueeze(1)
        reliability = region_reliability.unsqueeze(1)
        weighted_edges = causal_edges * reliability * mask
        denom = mask.sum(dim=(1, 2)).clamp_min(1.0)
        return weighted_edges.sum(dim=(1, 2)) / denom


def summarize_reasoning(
    reasoning_outputs: Dict[str, object],
    top_k: int = 5,
) -> List[Dict[str, object]]:
    """Convert reasoning tensors into lightweight per-sample summaries."""

    causal_edges = reasoning_outputs["causal_edges"].detach().cpu()
    reasoning_confidence = reasoning_outputs["reasoning_confidence"].detach().cpu()
    clinical_graph = reasoning_outputs["clinical_graph"]
    visual_graph = reasoning_outputs["visual_graph"]
    concept_names = tuple(clinical_graph["concept_names"])
    region_names = tuple(visual_graph["region_names"])

    summaries: List[Dict[str, object]] = []
    for sample_idx in range(causal_edges.shape[0]):
        sample_edges = causal_edges[sample_idx]
        flat_scores = sample_edges.flatten()
        k = min(top_k, flat_scores.numel())
        scores, indices = torch.topk(flat_scores, k=k)

        links = []
        for score, flat_idx in zip(scores.tolist(), indices.tolist()):
            concept_idx = flat_idx // len(region_names)
            region_idx = flat_idx % len(region_names)
            if score <= 0.0:
                continue
            links.append(
                {
                    "concept": concept_names[concept_idx],
                    "region": region_names[region_idx],
                    "causal_strength": float(score),
                }
            )

        summaries.append(
            {
                "reasoning_confidence": float(reasoning_confidence[sample_idx]),
                "top_causal_links": links,
            }
        )

    return summaries


def build_reasoning_graph(
    reasoning_outputs: Dict[str, object],
    sample_idx: int = 0,
    sample_name: Optional[str] = None,
    min_edge_strength: float = 0.05,
) -> Dict[str, object]:
    """Build a full per-sample KG/SCM graph suitable for JSON/HTML/PNG export."""

    clinical_graph = reasoning_outputs["clinical_graph"]
    visual_graph = reasoning_outputs["visual_graph"]
    causal_edges = reasoning_outputs["causal_edges"].detach().cpu()
    reasoning_confidence = reasoning_outputs["reasoning_confidence"].detach().cpu()
    attention_edges = reasoning_outputs.get("attention_edges")
    if attention_edges is not None:
        attention_edges = attention_edges.detach().cpu()

    concept_names = tuple(clinical_graph["concept_names"])
    region_names = tuple(visual_graph["region_names"])
    concept_presence = clinical_graph["concept_presence"].detach().cpu()
    region_presence = visual_graph["region_presence"].detach().cpu()
    region_mass = visual_graph["region_mass"].detach().cpu()
    region_uncertainty = visual_graph["region_uncertainty"].detach().cpu()
    region_reliability = visual_graph["region_reliability"].detach().cpu()
    token_concepts = clinical_graph["token_concepts"][sample_idx]
    token_words = clinical_graph["token_words"][sample_idx]

    nodes: List[Dict[str, object]] = []
    edges: List[Dict[str, object]] = []

    for concept_idx, concept_name in enumerate(concept_names):
        presence = float(concept_presence[sample_idx, concept_idx])
        nodes.append(
            {
                "id": f"concept:{concept_name}",
                "label": concept_name,
                "type": "clinical_concept",
                "present": bool(presence > 0.0),
                "presence": presence,
            }
        )

    for region_idx, region_name in enumerate(region_names):
        nodes.append(
            {
                "id": f"region:{region_name}",
                "label": region_name,
                "type": "visual_region",
                "present": bool(region_presence[sample_idx, region_idx] > 0.0),
                "region_mass": float(region_mass[sample_idx, region_idx]),
                "uncertainty": float(region_uncertainty[sample_idx, region_idx]),
                "reliability": float(region_reliability[sample_idx, region_idx]),
            }
        )

    for token_idx, matched_concepts in enumerate(token_concepts):
        if matched_concepts:
            token_label = str(token_words[token_idx])
            nodes.append(
                {
                    "id": f"token:{token_idx}",
                    "label": token_label,
                    "type": "clinical_token",
                    "present": True,
                }
            )
        for concept_name in matched_concepts:
            edges.append(
                {
                    "source": f"token:{token_idx}",
                    "target": f"concept:{concept_name}",
                    "type": "token_supports_concept",
                    "weight": 1.0,
                }
            )

    edges.extend(_clinical_prior_edges(concept_names, concept_presence[sample_idx]))

    sample_causal_edges = causal_edges[sample_idx]
    for concept_idx, concept_name in enumerate(concept_names):
        for region_idx, region_name in enumerate(region_names):
            causal_strength = float(sample_causal_edges[concept_idx, region_idx])
            if causal_strength < min_edge_strength:
                continue
            edge = {
                "source": f"concept:{concept_name}",
                "target": f"region:{region_name}",
                "type": "causal_support",
                "weight": causal_strength,
                "causal_strength": causal_strength,
            }
            if attention_edges is not None:
                edge["attention_alignment"] = float(
                    attention_edges[sample_idx, concept_idx, region_idx]
                )
            edges.append(edge)

    return {
        "sample_name": sample_name or f"sample_{sample_idx}",
        "graph_type": "clinical_visual_scm_reasoning_graph",
        "reasoning_confidence": float(reasoning_confidence[sample_idx]),
        "nodes": nodes,
        "edges": edges,
    }


def export_reasoning_graph(
    graph: Dict[str, object],
    output_prefix: str,
    export_html: bool = True,
    export_png: bool = True,
) -> Dict[str, str]:
    """Save a reasoning graph as JSON and optional HTML/PNG files."""

    output_path = Path(output_prefix)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    saved_paths: Dict[str, str] = {}

    json_path = output_path.with_suffix(".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(graph, f, indent=2)
    saved_paths["json"] = str(json_path)

    if export_html:
        html_path = output_path.with_suffix(".html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(render_reasoning_graph_html(graph))
        saved_paths["html"] = str(html_path)

    if export_png:
        png_path = output_path.with_suffix(".png")
        if render_reasoning_graph_png(graph, str(png_path)):
            saved_paths["png"] = str(png_path)

    return saved_paths


class PersistentKnowledgeGraph:
    """Aggregate per-sample reasoning graphs into a dataset-level KG."""

    def __init__(self, min_edge_strength: float = 0.05):
        self.min_edge_strength = min_edge_strength
        self.nodes: Dict[str, Dict[str, object]] = {}
        self.edges: Dict[Tuple[str, str, str], Dict[str, object]] = {}
        self.sample_count = 0

    def update_from_reasoning(
        self,
        reasoning_outputs: Dict[str, object],
        sample_names: Optional[Sequence[str]] = None,
    ) -> None:
        batch_size = reasoning_outputs["causal_edges"].shape[0]
        for sample_idx in range(batch_size):
            sample_name = None
            if sample_names is not None and sample_idx < len(sample_names):
                sample_name = str(sample_names[sample_idx])
            graph = build_reasoning_graph(
                reasoning_outputs,
                sample_idx=sample_idx,
                sample_name=sample_name,
                min_edge_strength=self.min_edge_strength,
            )
            self.update_from_graph(graph)

    def update_from_graph(self, graph: Dict[str, object]) -> None:
        self.sample_count += 1

        for node in graph["nodes"]:
            if node.get("type") == "clinical_token":
                continue
            node_id = str(node["id"])
            stored = self.nodes.setdefault(
                node_id,
                {
                    "id": node_id,
                    "label": node.get("label", node_id),
                    "type": node.get("type", "unknown"),
                    "count": 0,
                    "present_count": 0,
                },
            )
            stored["count"] = int(stored["count"]) + 1
            if node.get("present"):
                stored["present_count"] = int(stored["present_count"]) + 1

        for edge in graph["edges"]:
            if edge.get("type") == "token_supports_concept":
                continue
            key = (
                str(edge["source"]),
                str(edge["target"]),
                str(edge.get("type", "edge")),
            )
            weight = float(edge.get("weight", 1.0))
            stored_edge = self.edges.setdefault(
                key,
                {
                    "source": key[0],
                    "target": key[1],
                    "type": key[2],
                    "count": 0,
                    "weight_sum": 0.0,
                    "max_weight": 0.0,
                },
            )
            stored_edge["count"] = int(stored_edge["count"]) + 1
            stored_edge["weight_sum"] = float(stored_edge["weight_sum"]) + weight
            stored_edge["max_weight"] = max(float(stored_edge["max_weight"]), weight)

    def to_graph(self) -> Dict[str, object]:
        nodes = []
        for node in self.nodes.values():
            count = max(int(node["count"]), 1)
            nodes.append(
                {
                    **node,
                    "presence_rate": float(node["present_count"]) / count,
                }
            )

        edges = []
        for edge in self.edges.values():
            count = max(int(edge["count"]), 1)
            edges.append(
                {
                    "source": edge["source"],
                    "target": edge["target"],
                    "type": edge["type"],
                    "count": edge["count"],
                    "mean_weight": float(edge["weight_sum"]) / count,
                    "max_weight": edge["max_weight"],
                    "weight": float(edge["weight_sum"]) / count,
                }
            )

        return {
            "sample_name": "aggregate_knowledge_graph",
            "graph_type": "persistent_clinical_visual_knowledge_graph",
            "sample_count": self.sample_count,
            "nodes": nodes,
            "edges": edges,
        }

    def save(
        self,
        output_prefix: str,
        export_html: bool = True,
        export_png: bool = True,
    ) -> Dict[str, str]:
        return export_reasoning_graph(
            self.to_graph(),
            output_prefix,
            export_html=export_html,
            export_png=export_png,
        )


def render_reasoning_graph_html(graph: Dict[str, object]) -> str:
    """Render a dependency-free HTML/SVG graph visualization."""

    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    positions = _layout_nodes(nodes)
    width = 1180
    height = max(620, 90 + 56 * max(len(nodes), 6))
    title = html_lib.escape(str(graph.get("sample_name", "reasoning graph")))
    graph_type = html_lib.escape(str(graph.get("graph_type", "graph")))

    svg_edges = []
    for edge in edges:
        source = edge.get("source")
        target = edge.get("target")
        if source not in positions or target not in positions:
            continue
        x1, y1 = positions[source]
        x2, y2 = positions[target]
        weight = float(edge.get("weight", edge.get("mean_weight", 1.0)))
        stroke_width = 1.0 + min(max(weight, 0.0), 1.0) * 5.0
        label = html_lib.escape(_edge_label(edge))
        color = "#2b6cb0" if edge.get("type") == "causal_support" else "#718096"
        svg_edges.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            f'stroke="{color}" stroke-width="{stroke_width:.2f}" '
            f'stroke-opacity="0.72" marker-end="url(#arrow)" />'
        )
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        svg_edges.append(
            f'<text x="{mid_x}" y="{mid_y - 4}" class="edge-label">{label}</text>'
        )

    svg_nodes = []
    for node in nodes:
        node_id = str(node.get("id"))
        if node_id not in positions:
            continue
        x, y = positions[node_id]
        label = html_lib.escape(str(node.get("label", node_id)))
        node_type = node.get("type")
        fill = "#dbeafe" if node_type == "clinical_concept" else "#dcfce7"
        stroke = "#1d4ed8" if node_type == "clinical_concept" else "#15803d"
        if node_type not in {"clinical_concept", "visual_region"}:
            fill = "#f8fafc"
            stroke = "#64748b"
        opacity = 1.0 if node.get("present", True) else 0.45
        svg_nodes.append(
            f'<g opacity="{opacity:.2f}">'
            f'<rect x="{x - 88}" y="{y - 20}" width="176" height="40" rx="6" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="1.5" />'
            f'<text x="{x}" y="{y + 5}" class="node-label">{label}</text>'
            f'</g>'
        )

    details_rows = []
    for edge in sorted(edges, key=lambda item: float(item.get("weight", item.get("mean_weight", 0.0))), reverse=True):
        details_rows.append(
            "<tr>"
            f"<td>{html_lib.escape(str(edge.get('source', '')))}</td>"
            f"<td>{html_lib.escape(str(edge.get('target', '')))}</td>"
            f"<td>{html_lib.escape(str(edge.get('type', '')))}</td>"
            f"<td>{float(edge.get('weight', edge.get('mean_weight', 0.0))):.4f}</td>"
            "</tr>"
        )

    confidence = graph.get("reasoning_confidence")
    confidence_html = ""
    if confidence is not None:
        confidence_html = f"<p>Reasoning confidence: {float(confidence):.4f}</p>"

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>
    body {{
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      color: #111827;
      background: #f8fafc;
    }}
    main {{
      max-width: 1240px;
      margin: 0 auto;
      padding: 24px;
    }}
    h1 {{
      margin: 0 0 6px;
      font-size: 24px;
      font-weight: 700;
    }}
    p {{
      margin: 4px 0 16px;
      color: #475569;
    }}
    svg {{
      width: 100%;
      height: auto;
      background: #ffffff;
      border: 1px solid #dbe3ef;
    }}
    .node-label {{
      dominant-baseline: middle;
      text-anchor: middle;
      font-size: 13px;
      font-weight: 700;
      fill: #0f172a;
    }}
    .edge-label {{
      dominant-baseline: middle;
      text-anchor: middle;
      font-size: 11px;
      fill: #334155;
      paint-order: stroke;
      stroke: #ffffff;
      stroke-width: 4px;
    }}
    table {{
      width: 100%;
      margin-top: 18px;
      border-collapse: collapse;
      background: #ffffff;
      border: 1px solid #dbe3ef;
    }}
    th, td {{
      padding: 8px 10px;
      border-bottom: 1px solid #e5edf7;
      text-align: left;
      font-size: 13px;
    }}
    th {{
      background: #eef4fb;
      font-weight: 700;
    }}
  </style>
</head>
<body>
<main>
  <h1>{title}</h1>
  <p>{graph_type}</p>
  {confidence_html}
  <svg viewBox="0 0 {width} {height}" role="img">
    <defs>
      <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5"
        markerWidth="6" markerHeight="6" orient="auto-start-reverse">
        <path d="M 0 0 L 10 5 L 0 10 z" fill="#475569" />
      </marker>
    </defs>
    {''.join(svg_edges)}
    {''.join(svg_nodes)}
  </svg>
  <table>
    <thead>
      <tr><th>Source</th><th>Target</th><th>Type</th><th>Weight</th></tr>
    </thead>
    <tbody>
      {''.join(details_rows)}
    </tbody>
  </table>
</main>
</body>
</html>
"""


def render_reasoning_graph_png(graph: Dict[str, object], output_path: str) -> bool:
    """Render a simple PNG graph using OpenCV if it is available."""

    try:
        import cv2
        import numpy as np
    except Exception:
        return False

    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    positions = _layout_nodes(nodes)
    width = 1180
    height = max(620, 90 + 56 * max(len(nodes), 6))
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)

    for edge in edges:
        source = edge.get("source")
        target = edge.get("target")
        if source not in positions or target not in positions:
            continue
        x1, y1 = positions[source]
        x2, y2 = positions[target]
        weight = float(edge.get("weight", edge.get("mean_weight", 1.0)))
        thickness = int(1 + min(max(weight, 0.0), 1.0) * 4)
        color = (176, 108, 43) if edge.get("type") == "causal_support" else (150, 150, 150)
        cv2.line(canvas, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness, cv2.LINE_AA)

    for node in nodes:
        node_id = str(node.get("id"))
        if node_id not in positions:
            continue
        x, y = positions[node_id]
        node_type = node.get("type")
        fill = (254, 234, 219) if node_type == "clinical_concept" else (231, 252, 220)
        stroke = (216, 78, 29) if node_type == "clinical_concept" else (61, 128, 21)
        if node_type not in {"clinical_concept", "visual_region"}:
            fill = (248, 250, 252)
            stroke = (132, 146, 166)
        top_left = (int(x - 88), int(y - 20))
        bottom_right = (int(x + 88), int(y + 20))
        cv2.rectangle(canvas, top_left, bottom_right, fill, -1)
        cv2.rectangle(canvas, top_left, bottom_right, stroke, 2)
        label = str(node.get("label", node_id))[:22]
        cv2.putText(
            canvas,
            label,
            (int(x - 72), int(y + 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (20, 30, 45),
            1,
            cv2.LINE_AA,
        )

    title = str(graph.get("sample_name", "reasoning graph"))[:80]
    cv2.putText(canvas, title, (28, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (15, 23, 42), 2, cv2.LINE_AA)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(output_path, canvas))


def _clinical_prior_edges(
    concept_names: Sequence[str],
    concept_presence: torch.Tensor,
) -> List[Dict[str, object]]:
    priors = (
        ("infection", "opacity", "causes_or_supports"),
        ("infection", "effusion", "may_cooccur_with"),
        ("lesion", "opacity", "may_appear_as"),
        ("edema", "opacity", "may_appear_as"),
        ("effusion", "lower", "often_localized_toward"),
        ("atelectasis", "lower", "often_localized_toward"),
        ("left", "opacity", "localizes"),
        ("right", "opacity", "localizes"),
        ("upper", "opacity", "localizes"),
        ("middle", "opacity", "localizes"),
        ("lower", "opacity", "localizes"),
        ("severity", "lesion", "modifies"),
        ("severity", "opacity", "modifies"),
    )
    concept_set = set(concept_names)
    concept_index = {name: idx for idx, name in enumerate(concept_names)}
    edges: List[Dict[str, object]] = []
    for source, target, edge_type in priors:
        if source not in concept_set or target not in concept_set:
            continue
        if concept_presence[concept_index[source]] <= 0.0:
            continue
        if concept_presence[concept_index[target]] <= 0.0:
            continue
        edges.append(
            {
                "source": f"concept:{source}",
                "target": f"concept:{target}",
                "type": edge_type,
                "weight": 1.0,
            }
        )
    return edges


def _layout_nodes(nodes: Sequence[Dict[str, object]]) -> Dict[str, Tuple[int, int]]:
    clinical_nodes = [node for node in nodes if node.get("type") == "clinical_concept"]
    region_nodes = [node for node in nodes if node.get("type") == "visual_region"]
    other_nodes = [
        node
        for node in nodes
        if node.get("type") not in {"clinical_concept", "visual_region"}
    ]
    row_count = max(len(clinical_nodes), len(region_nodes), len(other_nodes), 1)
    y_start = 90
    y_gap = 56
    positions: Dict[str, Tuple[int, int]] = {}

    for idx, node in enumerate(clinical_nodes):
        positions[str(node["id"])] = (230, y_start + idx * y_gap)
    for idx, node in enumerate(region_nodes):
        positions[str(node["id"])] = (920, y_start + idx * y_gap)
    for idx, node in enumerate(other_nodes):
        positions[str(node["id"])] = (575, y_start + idx * y_gap)

    return positions


def _edge_label(edge: Dict[str, object]) -> str:
    edge_type = str(edge.get("type", "edge"))
    if "mean_weight" in edge:
        return f"{edge_type} {float(edge['mean_weight']):.2f}"
    if "weight" in edge:
        return f"{edge_type} {float(edge['weight']):.2f}"
    return edge_type
