from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import tempfile
import time
import traceback
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch_geometric.data import Batch, Data
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from model_ast_gat import ASTGATClassifier
from model_fusion import FusionClassifier
from model_unixcoder import UnixCoderClassifier


# =========================
# CONFIG: sua o day la chay
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"

LANGUAGE = "java"
SMELL = "all"
MODEL_NAME = "fusion"  # unixcoder | ast_gat | fusion | all
MEMBER_NAME = "Giang"
TASK_NAME = "Code smell detection via fine-tuned pretrained models"
ALL_SMELLS = ["ComplexMethod", "ComplexConditional", "FeatureEnvy", "MultifacetedAbstraction"]
ALL_MODELS = ["unixcoder", "ast_gat", "fusion"]
MODEL_DISPLAY_NAMES = {
    "unixcoder": "UniXCoder",
    "ast_gat": "AST-GAT",
    "fusion": "UniXCoder + AST-GAT Fusion",
}

SEED = 42
POSITIVE_SAMPLES = 2000
NEGATIVE_SAMPLES = 8000
DEV_POSITIVE_SAMPLES = 200
DEV_NEGATIVE_SAMPLES = 800
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

UNIXCODER_NAME = "microsoft/unixcoder-base"
MAX_TOKEN_LENGTH = 512
MAX_AST_NODES = 512
SOURCE_EXTENSIONS = (".code", ".java", ".cs", ".txt")

BATCH_SIZE = 8
MAX_EPOCHS = 30
PATIENCE = 5
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
FOCAL_GAMMA = 2.0
WARMUP_RATIO = 0.10
NUM_WORKERS = 0
DEVICE = "auto"

GAT_HIDDEN_DIM = 128
GAT_HEADS = 4
GAT_LAYERS = 2
DROPOUT = 0.2

SAVE_BEST_MODEL = True
FORCE_REBUILD_INDEX = False
FORCE_RESAMPLE = False
FORCE_REBUILD_AST = False
FORCE_REBUILD_TOKENS = False

NODE_TYPE_HASH_BUCKETS = 128
NODE_NUMERIC_FEATURES = 5  # depth, child_count, sibling_index, named flag, terminal flag
NODE_FEATURE_DIM = NODE_TYPE_HASH_BUCKETS + NODE_NUMERIC_FEATURES
EDGE_TYPES = {
    "parent_to_child": 0,
    "child_to_parent": 1,
    "next_sibling": 2,
    "prev_sibling": 3,
    "same_identifier": 4,
    "control_flow_hint": 5,
    "member_receiver": 6,
}
EDGE_FEATURE_DIM = len(EDGE_TYPES)
INDEX_VERSION = 4
TOKEN_SCHEMA_VERSION = 2
AST_SCHEMA_VERSION = 2
MIN_SAME_IDENTIFIER_LENGTH = 3
MAX_SAME_IDENTIFIER_OCCURRENCES = 10


@dataclass(frozen=True)
class Paths:
    artifact_dir: Path
    checkpoint_dir: Path
    result_csv: Path
    dataset_index: Path
    sampled_dataset: Path
    splits: Path
    sampled_sources: Path
    source_cache_ids: Path
    token_cache: Path
    token_cache_ids: Path
    token_cache_meta: Path
    token_stats: Path
    ast_parse_result: Path
    ast_graphs: Path
    ast_cache_ids: Path
    ast_cache_meta: Path
    sample_audit: Path
    split_summary: Path
    source_extract_audit: Path
    log_file: Path
    best_checkpoint: Path


class RunLogger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def info(self, message: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] INFO {message}"
        print(line)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def error(self, message: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ERROR {message}"
        print(line)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def warning(self, message: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] WARNING {message}"
        print(line)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def exception(self, message: str) -> None:
        self.error(message)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(traceback.format_exc() + "\n")


class FocalLoss(nn.Module):
    def __init__(self, alpha: torch.Tensor | None = None, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma
        if alpha is not None:
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)[targets]
        else:
            alpha_t = 1.0
        return (alpha_t * ((1.0 - pt) ** self.gamma) * ce_loss).mean()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", default=LANGUAGE)
    parser.add_argument("--smell", default=SMELL, help="Use 'all' to run all four smells in one command.")
    parser.add_argument("--model", choices=["unixcoder", "ast_gat", "fusion", "all"], default=MODEL_NAME)
    parser.add_argument("--member", default=MEMBER_NAME)
    parser.add_argument("--hardware", default=None, help="Example: 'Kaggle T4 x2'. Defaults to detected hardware.")
    parser.add_argument("--dev", action="store_true", help="Demo mode: 200 positive + 800 negative per smell.")
    parser.add_argument("--positive", type=int, default=None)
    parser.add_argument("--negative", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--force-rebuild-index", action="store_true")
    parser.add_argument("--force-resample", action="store_true")
    parser.add_argument("--force-rebuild-ast", action="store_true")
    parser.add_argument("--force-rebuild-tokens", action="store_true")
    parser.add_argument("--no-save-model", action="store_true")
    return parser.parse_args()


def resolve_smells(smell_arg: str) -> list[str]:
    if smell_arg.lower() == "all":
        return ALL_SMELLS
    return [smell_arg]


def resolve_models(model_arg: str) -> list[str]:
    if model_arg.lower() == "all":
        return ALL_MODELS
    return [model_arg]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if DEVICE == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(DEVICE)


def load_trusted_artifact(path: Path, map_location):
    # These pickle-based files are generated locally by this experiment.
    return torch.load(path, map_location=map_location, weights_only=False)


def get_paths(language: str, smell: str, model_name: str) -> Paths:
    artifact_dir = PROJECT_ROOT / "artifacts" / language / smell
    checkpoint_dir = PROJECT_ROOT / "checkpoints" / language / smell
    result_csv = PROJECT_ROOT / "results" / "results.csv"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    result_csv.parent.mkdir(parents=True, exist_ok=True)
    return Paths(
        artifact_dir=artifact_dir,
        checkpoint_dir=checkpoint_dir,
        result_csv=result_csv,
        dataset_index=artifact_dir / "dataset_index.csv",
        sampled_dataset=artifact_dir / "sampled_dataset.csv",
        splits=artifact_dir / "splits.csv",
        sampled_sources=artifact_dir / "sampled_sources.jsonl",
        source_cache_ids=artifact_dir / "source_cache_ids.txt",
        token_cache=artifact_dir / "token_cache.pt",
        token_cache_ids=artifact_dir / "token_cache_ids.txt",
        token_cache_meta=artifact_dir / "token_cache_meta.json",
        token_stats=artifact_dir / "token_stats.csv",
        ast_parse_result=artifact_dir / "ast_parse_result.csv",
        ast_graphs=artifact_dir / "ast_graphs.pt",
        ast_cache_ids=artifact_dir / "ast_cache_ids.txt",
        ast_cache_meta=artifact_dir / "ast_cache_meta.json",
        sample_audit=artifact_dir / "sample_audit.csv",
        split_summary=artifact_dir / "split_summary.csv",
        source_extract_audit=artifact_dir / "source_extract_audit.csv",
        log_file=artifact_dir / f"{model_name}_run.log",
        best_checkpoint=checkpoint_dir / f"{model_name}_best.pt",
    )


def smell_sample_size(smell: str, positive: int | None, negative: int | None, dev: bool) -> tuple[int, int]:
    if positive is not None:
        negative = negative if negative is not None else positive * 4
        return positive, negative
    if dev:
        return DEV_POSITIVE_SAMPLES, DEV_NEGATIVE_SAMPLES
    return POSITIVE_SAMPLES, POSITIVE_SAMPLES * 4


def infer_label(member_name: str) -> int:
    lower = member_name.replace("\\", "/").lower()
    positive_keys = ["/positive/", "_positive_", "/pos/", "_pos_", "/true/", "_true_"]
    negative_keys = ["/negative/", "_negative_", "/neg/", "_neg_", "/false/", "_false_"]
    if any(key in lower for key in positive_keys):
        return 1
    if any(key in lower for key in negative_keys):
        return 0
    raise ValueError(f"Cannot infer label from archive member: {member_name}")


def infer_repo(member_name: str) -> str:
    parts = [part for part in member_name.replace("\\", "/").split("/") if part]
    if parts:
        stem = Path(parts[-1]).stem
        match = re.match(r"^(\d+)_", stem)
        if match:
            return match.group(1)
    return "unknown_repo"


def infer_class_name(member_name: str) -> tuple[str, str]:
    stem = Path(member_name.replace("\\", "/")).stem
    parts = stem.split("_")
    pattern = re.compile(
        r"^(?P<class_name>[A-Z$][A-Za-z0-9_$]*)(?P<ordinal>\d+)(?P<method_name>[A-Za-z_$][A-Za-z0-9_$]*)$"
    )
    for start in range(1, len(parts)):
        candidate = "_".join(parts[start:])
        match = pattern.match(candidate)
        if match:
            return match.group("class_name"), "parsed"
    return "unknown_class", "unparsed"


def find_7z_cli() -> str | None:
    from_path = shutil.which("7z") or shutil.which("7za") or shutil.which("7zr")
    if from_path:
        return from_path
    for candidate in (Path("C:/Program Files/7-Zip/7z.exe"), Path("C:/Program Files (x86)/7-Zip/7z.exe")):
        if candidate.exists():
            return str(candidate)
    return None


def require_py7zr():
    try:
        import py7zr
    except ImportError as exc:
        raise RuntimeError(
            "Neither 7z CLI nor py7zr is available. Install 7-Zip or run: pip install py7zr"
        ) from exc
    return py7zr


def iter_archive_members(archive_path: Path) -> list[str]:
    seven_zip = find_7z_cli()
    if seven_zip:
        completed = subprocess.run(
            [seven_zip, "l", "-slt", str(archive_path)],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
        )
        members = []
        for line in completed.stdout.splitlines():
            if line.startswith("Path = "):
                member = line.removeprefix("Path = ").strip()
                if member and member != str(archive_path):
                    members.append(member)
        return members

    py7zr = require_py7zr()
    with py7zr.SevenZipFile(archive_path, mode="r") as archive:
        return archive.getnames()


def source_archive_dir(language: str) -> Path:
    return DATA_ROOT / f"training_data_{language}"


def find_smell_archives(language: str, smell: str, logger: RunLogger) -> list[Path]:
    root = source_archive_dir(language)
    archive = root / f"{smell}.7z"
    if not archive.exists():
        raise FileNotFoundError(f"Missing archive: {archive}")
    logger.info(f"Use archive only (ignore extensionless _1/_2 files): {archive}")
    return [archive]


def build_dataset_index(language: str, smell: str, paths: Paths, force: bool, logger: RunLogger) -> pd.DataFrame:
    archives = find_smell_archives(language, smell, logger)
    if paths.dataset_index.exists() and not force:
        cached = pd.read_csv(paths.dataset_index)
        archive_mtime = max(path.stat().st_mtime for path in archives)
        if (
            "index_version" in cached
            and set(cached["index_version"].unique()) == {INDEX_VERSION}
            and paths.dataset_index.stat().st_mtime >= archive_mtime
        ):
            logger.info(f"Reuse dataset index: {paths.dataset_index}")
            return cached
        logger.info("Dataset index schema/archive handling changed; rebuilding index.")

    if not archives:
        raise FileNotFoundError(f"No archive found for {language}/{smell} under {source_archive_dir(language)}")

    rows = []
    for archive_path in archives:
        logger.info(f"Index archive: {archive_path}")
        for member_name in iter_archive_members(archive_path):
            if not member_name.lower().endswith(SOURCE_EXTENSIONS):
                continue
            try:
                label = infer_label(member_name)
            except ValueError:
                continue
            sample_id = hashlib.sha1(f"{archive_path}|{member_name}".encode("utf-8")).hexdigest()
            class_name, class_parse_status = infer_class_name(member_name)
            rows.append(
                {
                    "sample_id": sample_id,
                    "archive_path": str(archive_path),
                    "member_name": member_name,
                    "repo": infer_repo(member_name),
                    "label": label,
                    "index_version": INDEX_VERSION,
                    "class_name": class_name,
                    "class_parse_status": class_parse_status,
                }
            )

    if not rows:
        raise RuntimeError(
            "Cannot build dataset index because no labelled members were found. "
            "Open src/run_experiment.py and update infer_label() to match your archive structure."
        )
    df = pd.DataFrame(rows)
    df.to_csv(paths.dataset_index, index=False)
    counts = df["label"].value_counts().to_dict()
    repo_count = df["repo"].nunique()
    class_status = df["class_parse_status"].value_counts().to_dict()
    logger.info(
        f"Wrote dataset_index rows={len(df)} positives={counts.get(1, 0)} "
        f"negatives={counts.get(0, 0)} repos={repo_count} class_parse={class_status}"
    )
    return df


def sample_diverse_by_repo(records: list[dict[str, object]], target_n: int, rng: random.Random) -> list[dict[str, object]]:
    groups: dict[str, list[dict[str, object]]] = {}
    for record in records:
        groups.setdefault(str(record.get("repo", "unknown_repo")), []).append(record)
    for group in groups.values():
        rng.shuffle(group)
    repo_order = list(groups.keys())
    rng.shuffle(repo_order)

    selected = []
    while len(selected) < target_n:
        progressed = False
        for repo in repo_order:
            if groups[repo]:
                selected.append(groups[repo].pop())
                progressed = True
                if len(selected) >= target_n:
                    break
        if not progressed:
            break
    return selected


def write_sample_audit(
    paths: Paths,
    requested_positive: int,
    requested_negative: int,
    selected_positive: int,
    selected_negative: int,
    available_positive: int,
    available_negative: int,
    sampled_df: pd.DataFrame,
    index_rows: int,
) -> None:
    rows = [
        {
            "requested_positive": requested_positive,
            "requested_negative": requested_negative,
            "available_positive": available_positive,
            "available_negative": available_negative,
            "selected_positive": selected_positive,
            "selected_negative": selected_negative,
            "selected_total": len(sampled_df),
            "repo_count": sampled_df["repo"].nunique() if "repo" in sampled_df else 0,
            "seed": SEED,
            "train_ratio": TRAIN_RATIO,
            "validation_ratio": VAL_RATIO,
            "test_ratio": TEST_RATIO,
            "index_rows": index_rows,
            "index_version": INDEX_VERSION,
        }
    ]
    pd.DataFrame(rows).to_csv(paths.sample_audit, index=False)


def write_split_summary(paths: Paths, split_df: pd.DataFrame) -> None:
    summary = (
        split_df.groupby(["split", "label"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["split", "label"])
    )
    repo_summary = (
        split_df.groupby("split")
        .agg(total=("sample_id", "count"), repo_count=("repo", "nunique"))
        .reset_index()
    )
    summary["section"] = "label_count"
    repo_summary["section"] = "split_total"
    combined = pd.concat([summary, repo_summary], ignore_index=True, sort=False)
    combined.to_csv(paths.split_summary, index=False)


def expected_split_sizes(total: int) -> dict[str, int]:
    train_size = round(total * TRAIN_RATIO)
    val_size = round(total * VAL_RATIO)
    return {"train": train_size, "val": val_size, "test": total - train_size - val_size}


def has_expected_split_sizes(split_df: pd.DataFrame) -> bool:
    actual = split_df["split"].value_counts().to_dict()
    expected = expected_split_sizes(len(split_df))
    return all(actual.get(name, 0) == size for name, size in expected.items())


def sample_cache_matches(
    index_df: pd.DataFrame,
    cached: pd.DataFrame,
    paths: Paths,
    requested_positive: int,
    requested_negative: int,
) -> bool:
    if not paths.sample_audit.exists() or "split_protocol" not in cached or not has_expected_split_sizes(cached):
        return False
    audit = pd.read_csv(paths.sample_audit)
    if audit.empty:
        return False
    row = audit.iloc[0]
    required = {
        "requested_positive",
        "requested_negative",
        "selected_positive",
        "selected_negative",
        "seed",
        "train_ratio",
        "validation_ratio",
        "test_ratio",
        "index_rows",
        "index_version",
    }
    if not required.issubset(audit.columns):
        return False
    cached_counts = cached["label"].value_counts().to_dict()
    return (
        int(row["requested_positive"]) == requested_positive
        and int(row["requested_negative"]) == requested_negative
        and int(row["selected_positive"]) == cached_counts.get(1, 0)
        and int(row["selected_negative"]) == cached_counts.get(0, 0)
        and int(row["seed"]) == SEED
        and math.isclose(float(row["train_ratio"]), TRAIN_RATIO, rel_tol=0.0, abs_tol=1e-9)
        and math.isclose(float(row["validation_ratio"]), VAL_RATIO, rel_tol=0.0, abs_tol=1e-9)
        and math.isclose(float(row["test_ratio"]), TEST_RATIO, rel_tol=0.0, abs_tol=1e-9)
        and int(row["index_rows"]) == len(index_df)
        and int(row["index_version"]) == INDEX_VERSION
        and "class_name" in cached.columns
        and set(cached["sample_id"].astype(str)).issubset(set(index_df["sample_id"].astype(str)))
    )


def sample_and_split(
    index_df: pd.DataFrame,
    paths: Paths,
    positive_n: int,
    negative_n: int,
    force: bool,
    logger: RunLogger,
) -> tuple[pd.DataFrame, bool]:
    if paths.splits.exists() and paths.sampled_dataset.exists() and not force:
        cached = pd.read_csv(paths.splits)
        if sample_cache_matches(index_df, cached, paths, positive_n, negative_n):
            logger.info(f"Reuse sampled_dataset/splits: {paths.sampled_dataset}, {paths.splits}")
            return cached, False
        logger.info("Sampling config/index changed or cache is incomplete; rebuilding sample/split.")

    rng = random.Random(SEED)
    positives = index_df[index_df["label"] == 1].to_dict("records")
    negatives = index_df[index_df["label"] == 0].to_dict("records")
    requested_positive = positive_n
    requested_negative = negative_n
    positive_n = min(positive_n, len(positives))
    negative_n = min(negative_n, positive_n * 4, len(negatives))
    if positive_n == 0 or negative_n == 0:
        raise RuntimeError("Sampling needs both positive and negative samples.")

    sampled_pos = sample_diverse_by_repo(positives, positive_n, rng)
    sampled_neg = sample_diverse_by_repo(negatives, negative_n, rng)
    sampled = sampled_pos + sampled_neg
    rng.shuffle(sampled)
    sampled_df = pd.DataFrame(sampled)
    sampled_df.to_csv(paths.sampled_dataset, index=False)
    write_sample_audit(
        paths,
        requested_positive,
        requested_negative,
        positive_n,
        negative_n,
        len(positives),
        len(negatives),
        sampled_df,
        len(index_df),
    )
    logger.info(
        "Sampled data "
        f"requested_pos={requested_positive} requested_neg={requested_negative} "
        f"available_pos={len(positives)} available_neg={len(negatives)} "
        f"selected_pos={positive_n} selected_neg={negative_n} repos={sampled_df['repo'].nunique()}"
    )

    train_df, val_df, test_df, split_protocol = split_repo_aware(sampled_df)
    split_df = pd.concat(
        [
            train_df.assign(split="train"),
            val_df.assign(split="val"),
            test_df.assign(split="test"),
        ],
        ignore_index=True,
    )
    split_df["split_protocol"] = split_protocol
    split_df.to_csv(paths.splits, index=False)
    write_split_summary(paths, split_df)
    split_counts = split_df["split"].value_counts().to_dict()
    logger.info(
        f"Wrote splits train={split_counts.get('train', 0)} val={split_counts.get('val', 0)} "
        f"test={split_counts.get('test', 0)} protocol={split_protocol}"
    )
    return split_df, True


def split_repo_aware(sampled_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    fallback_protocol = "70/15/15 stratified by label (repo-aware exact split unavailable)"
    if "repo" not in sampled_df.columns or sampled_df["repo"].nunique() < 3:
        return (*split_stratified_by_label(sampled_df), fallback_protocol)

    global_pos_rate = float(sampled_df["label"].mean())
    expected_sizes = expected_split_sizes(len(sampled_df))

    def score_split(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> float:
        expected = {"train": TRAIN_RATIO, "val": VAL_RATIO, "test": TEST_RATIO}
        frames = {"train": train_df, "val": val_df, "test": test_df}
        score = 0.0
        total = len(sampled_df)
        for name, frame in frames.items():
            if frame.empty:
                return float("inf")
            score += abs((len(frame) / total) - expected[name])
            score += abs(float(frame["label"].mean()) - global_pos_rate)
        return score

    best_split = None
    best_score = float("inf")
    try:
        for offset in range(50):
            train_splitter = GroupShuffleSplit(n_splits=1, train_size=TRAIN_RATIO, random_state=SEED + offset)
            train_idx, temp_idx = next(train_splitter.split(sampled_df, sampled_df["label"], groups=sampled_df["repo"]))
            train_df = sampled_df.iloc[train_idx]
            temp_df = sampled_df.iloc[temp_idx]
            relative_val = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
            temp_groups = temp_df["repo"]
            if temp_groups.nunique() < 2:
                continue
            val_splitter = GroupShuffleSplit(n_splits=1, train_size=relative_val, random_state=SEED + offset)
            val_idx, test_idx = next(val_splitter.split(temp_df, temp_df["label"], groups=temp_groups))
            val_df = temp_df.iloc[val_idx]
            test_df = temp_df.iloc[test_idx]
            actual_sizes = {"train": len(train_df), "val": len(val_df), "test": len(test_df)}
            if actual_sizes != expected_sizes:
                continue
            candidate_score = score_split(train_df, val_df, test_df)
            if candidate_score < best_score:
                best_score = candidate_score
                best_split = (train_df, val_df, test_df)
        if best_split is None:
            raise ValueError("No repo-disjoint split has exact 70/15/15 sizes.")
        return (*best_split, "70/15/15 repo-aware with label-balance optimization")
    except ValueError:
        return (*split_stratified_by_label(sampled_df), fallback_protocol)


def split_stratified_by_label(sampled_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sizes = expected_split_sizes(len(sampled_df))
    train_df, temp_df = train_test_split(
        sampled_df,
        train_size=sizes["train"],
        random_state=SEED,
        stratify=sampled_df["label"],
    )
    val_df, test_df = train_test_split(
        temp_df,
        train_size=sizes["val"],
        random_state=SEED,
        stratify=temp_df["label"],
    )
    return train_df, val_df, test_df


def extract_targets_once(archive_path: str, targets: list[str], output_dir: Path, logger: RunLogger) -> None:
    seven_zip = find_7z_cli()
    if seven_zip:
        list_file = output_dir / "targets.txt"
        list_file.write_text("\n".join(targets), encoding="utf-8")
        completed = subprocess.run(
            [seven_zip, "x", "-y", "-scsUTF-8", f"-o{output_dir}", archive_path, f"@{list_file}"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
        )
        if completed.returncode == 1:
            logger.warning(f"7z extraction completed with warnings archive={archive_path}")
            if completed.stderr:
                logger.warning(completed.stderr[-4000:])
        elif completed.returncode != 0:
            logger.error(f"7z extraction failed archive={archive_path} returncode={completed.returncode}")
            logger.error(completed.stderr[-4000:])
            raise RuntimeError(f"7z extraction failed for {archive_path}")
        return

    logger.info("7z CLI not found; falling back to py7zr one-shot extraction for this archive.")
    py7zr = require_py7zr()
    with py7zr.SevenZipFile(archive_path, mode="r") as archive:
        archive.extract(path=output_dir, targets=targets)


def cached_source_ids(source_path: Path, ids_path: Path) -> set[str]:
    if ids_path.exists() and source_path.exists() and ids_path.stat().st_mtime >= source_path.stat().st_mtime:
        return {line.strip() for line in ids_path.read_text(encoding="utf-8").splitlines() if line.strip()}

    ids = set()
    if not source_path.exists():
        return ids
    with source_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                ids.add(str(json.loads(line)["sample_id"]))
    ids_path.write_text("\n".join(sorted(ids)) + "\n", encoding="utf-8")
    return ids


def write_cache_ids(ids_path: Path, ids: set[str]) -> None:
    ids_path.write_text("\n".join(sorted(ids)) + "\n", encoding="utf-8")


def cache_metadata_matches(meta_path: Path, expected: dict[str, object]) -> bool:
    if not meta_path.exists():
        return False
    try:
        return json.loads(meta_path.read_text(encoding="utf-8")) == expected
    except (OSError, json.JSONDecodeError):
        return False


def write_cache_metadata(meta_path: Path, metadata: dict[str, object]) -> None:
    meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")


def tensor_cache_ids_match(
    cache_path: Path,
    ids_path: Path,
    expected_ids: set[str],
    logger: RunLogger,
) -> bool:
    if ids_path.exists() and ids_path.stat().st_mtime >= cache_path.stat().st_mtime:
        cached_ids = {line.strip() for line in ids_path.read_text(encoding="utf-8").splitlines() if line.strip()}
        return cached_ids == expected_ids

    logger.info(f"Create missing cache ID sidecar once: {ids_path}")
    cached = load_trusted_artifact(cache_path, map_location="cpu")
    cached_ids = set(map(str, cached.keys()))
    write_cache_ids(ids_path, cached_ids)
    del cached
    return cached_ids == expected_ids


def cache_sampled_sources(split_df: pd.DataFrame, paths: Paths, force: bool, logger: RunLogger) -> set[str]:
    expected_ids = set(split_df["sample_id"].astype(str))
    if paths.sampled_sources.exists() and not force:
        existing_ids = cached_source_ids(paths.sampled_sources, paths.source_cache_ids)
        if expected_ids == existing_ids:
            logger.info(f"Reuse sampled source cache: {paths.sampled_sources}")
            return expected_ids
        logger.warning("Source cache is incomplete for the current split; rebuilding it.")

    audit_rows = []
    valid_ids = set()
    with paths.sampled_sources.open("w", encoding="utf-8") as f:
        grouped = split_df.groupby("archive_path", sort=False)
        for archive_path, group_df in tqdm(grouped, total=len(grouped), desc="cache sources by archive"):
            records = group_df.to_dict("records")
            targets = [str(row["member_name"]) for row in records]
            logger.info(f"Extract sampled sources archive={archive_path} targets={len(targets)}")
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                extract_targets_once(str(archive_path), targets, tmp_path, logger)
                for row in records:
                    extracted = tmp_path / str(row["member_name"])
                    if not extracted.exists():
                        logger.warning(f"Skip missing extracted member: {row['member_name']}")
                        audit_rows.append(
                            {
                                "sample_id": row["sample_id"],
                                "archive_path": archive_path,
                                "member_name": row["member_name"],
                                "status": "missing_after_extract",
                            }
                        )
                        continue
                    try:
                        code = extracted.read_bytes().decode("utf-8", errors="ignore")
                    except OSError as exc:
                        logger.warning(f"Skip unreadable extracted member: {row['member_name']} error={exc}")
                        audit_rows.append(
                            {
                                "sample_id": row["sample_id"],
                                "archive_path": archive_path,
                                "member_name": row["member_name"],
                                "status": f"read_error:{type(exc).__name__}",
                            }
                        )
                        continue
                    sample_id = str(row["sample_id"])
                    valid_ids.add(sample_id)
                    f.write(json.dumps({"sample_id": sample_id, "code": code}, ensure_ascii=False) + "\n")
                    audit_rows.append(
                        {
                            "sample_id": sample_id,
                            "archive_path": archive_path,
                            "member_name": row["member_name"],
                            "status": "ok",
                        }
                    )
    pd.DataFrame(audit_rows).to_csv(paths.source_extract_audit, index=False)
    write_cache_ids(paths.source_cache_ids, valid_ids)
    logger.info(
        f"Wrote sampled source cache valid={len(valid_ids)} missing={len(expected_ids - valid_ids)} "
        f"audit={paths.source_extract_audit}"
    )
    return valid_ids


def filter_and_resplit_missing_sources(
    split_df: pd.DataFrame,
    valid_ids: set[str],
    paths: Paths,
    logger: RunLogger,
) -> pd.DataFrame:
    sample_ids = split_df["sample_id"].astype(str)
    missing_count = int((~sample_ids.isin(valid_ids)).sum())
    if missing_count == 0:
        return split_df

    logger.warning(f"Remove {missing_count} samples missing from source cache and rebuild exact 70/15/15 split.")
    sampled_df = split_df[sample_ids.isin(valid_ids)].copy()
    sampled_df = sampled_df.drop(columns=["split", "split_protocol"], errors="ignore")
    if sampled_df["label"].nunique() < 2:
        raise RuntimeError("Cannot continue after missing-source filtering because one label has no samples.")

    train_df, val_df, test_df, protocol = split_repo_aware(sampled_df)
    filtered_split = pd.concat(
        [
            train_df.assign(split="train"),
            val_df.assign(split="val"),
            test_df.assign(split="test"),
        ],
        ignore_index=True,
    )
    filtered_split["split_protocol"] = protocol
    sampled_df.to_csv(paths.sampled_dataset, index=False)
    filtered_split.to_csv(paths.splits, index=False)
    write_split_summary(paths, filtered_split)

    if paths.sample_audit.exists():
        audit = pd.read_csv(paths.sample_audit)
        counts = sampled_df["label"].value_counts().to_dict()
        audit.loc[:, "selected_positive"] = counts.get(1, 0)
        audit.loc[:, "selected_negative"] = counts.get(0, 0)
        audit.loc[:, "selected_total"] = len(sampled_df)
        audit.loc[:, "repo_count"] = sampled_df["repo"].nunique()
        audit.loc[:, "missing_source_count"] = missing_count
        audit.to_csv(paths.sample_audit, index=False)
    return filtered_split


def load_source_cache(paths: Paths) -> dict[str, str]:
    source_by_id = {}
    with paths.sampled_sources.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            source_by_id[row["sample_id"]] = row["code"]
    return source_by_id


def token_cache_metadata(smell: str) -> dict[str, object]:
    return {
        "schema_version": TOKEN_SCHEMA_VERSION,
        "tokenizer": UNIXCODER_NAME,
        "max_token_length": MAX_TOKEN_LENGTH,
        "feature_envy_class_context": smell == "FeatureEnvy",
    }


def cache_tokenization(
    split_df: pd.DataFrame,
    paths: Paths,
    tokenizer,
    smell: str,
    force: bool,
    logger: RunLogger,
) -> None:
    metadata = token_cache_metadata(smell)
    if paths.token_cache.exists() and paths.token_stats.exists() and not force:
        expected_ids = set(split_df["sample_id"].astype(str))
        if cache_metadata_matches(paths.token_cache_meta, metadata) and tensor_cache_ids_match(
            paths.token_cache,
            paths.token_cache_ids,
            expected_ids,
            logger,
        ):
            logger.info(f"Reuse token cache/stats: {paths.token_cache}, {paths.token_stats}")
            return
        logger.info("Token cache schema or IDs changed; rebuilding token cache/stats.")
    logger.info("Start tokenization cache and token length audit.")
    source_by_id = load_source_cache(paths)
    token_by_id = {}
    stats_rows = []
    for row in tqdm(split_df.to_dict("records"), desc="tokenize cache"):
        sample_id = row["sample_id"]
        code = source_by_id[sample_id]
        class_name = str(row.get("class_name", "unknown_class"))
        add_class_context = smell == "FeatureEnvy" and class_name != "unknown_class"
        model_input = f"// containing_class: {class_name}\n{code}" if add_class_context else code
        full_tokens = tokenizer(model_input, truncation=False, padding=False, return_tensors=None)
        token_length = len(full_tokens["input_ids"])
        cached_tokens = tokenizer(
            model_input,
            truncation=True,
            max_length=MAX_TOKEN_LENGTH,
            padding=False,
            return_tensors=None,
        )
        token_by_id[sample_id] = {
            "input_ids": cached_tokens["input_ids"],
            "attention_mask": cached_tokens["attention_mask"],
        }
        stats_rows.append(
            {
                "sample_id": sample_id,
                "split": row["split"],
                "label": row["label"],
                "member_name": row["member_name"],
                "class_name": class_name,
                "class_context_added": add_class_context,
                "token_length": token_length,
                "cached_length": len(cached_tokens["input_ids"]),
                "is_truncated": token_length > MAX_TOKEN_LENGTH,
            }
        )
    torch.save(token_by_id, paths.token_cache)
    write_cache_ids(paths.token_cache_ids, set(map(str, token_by_id.keys())))
    write_cache_metadata(paths.token_cache_meta, metadata)
    pd.DataFrame(stats_rows).sort_values("token_length", ascending=False).to_csv(paths.token_stats, index=False)
    stats_df = pd.DataFrame(stats_rows)
    logger.info(
        "Wrote token cache/stats "
        f"rows={len(stats_df)} max_token_length={int(stats_df['token_length'].max())} "
        f"truncated={int(stats_df['is_truncated'].sum())} file={paths.token_stats}"
    )


def java_parser():
    import tree_sitter_java as tsjava
    from tree_sitter import Language, Parser

    language = Language(tsjava.language())
    try:
        parser = Parser(language)
    except TypeError:
        parser = Parser()
        if hasattr(parser, "set_language"):
            parser.set_language(language)
        else:
            parser.language = language
    return parser


def node_type_bucket(node_type: str) -> int:
    digest = hashlib.md5(node_type.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % NODE_TYPE_HASH_BUCKETS


def node_feature(node, depth: int, sibling_index: int) -> list[float]:
    vec = [0.0] * NODE_FEATURE_DIM
    vec[node_type_bucket(node.type)] = 1.0
    offset = NODE_TYPE_HASH_BUCKETS
    child_count = len(node.children)
    vec[offset + 0] = min(depth, 32) / 32.0
    vec[offset + 1] = min(child_count, 32) / 32.0
    vec[offset + 2] = min(sibling_index, 32) / 32.0
    vec[offset + 3] = 1.0 if getattr(node, "is_named", False) else 0.0
    vec[offset + 4] = 1.0 if child_count == 0 else 0.0
    return vec


def edge_feature(edge_type: str) -> list[float]:
    vec = [0.0] * EDGE_FEATURE_DIM
    vec[EDGE_TYPES[edge_type]] = 1.0
    return vec


def code_to_graph(code: str, parser, language: str) -> tuple[Data, str]:
    if language != "java":
        raise NotImplementedError("C# needs tree-sitter-c-sharp support added here.")
    tree = parser.parse(code.encode("utf-8", errors="ignore"))
    features = []
    edges = []
    edge_attrs = []
    node_types = []
    node_texts = []
    control_nodes = []
    node_index_by_key = {}
    pending_receiver_edges = []

    def node_key(node) -> tuple[int, int, str]:
        return (int(node.start_byte), int(node.end_byte), str(node.type))

    def add_edge(src: int, dst: int, edge_type: str) -> None:
        edges.append((src, dst))
        edge_attrs.append(edge_feature(edge_type))

    last_child_by_parent: dict[int, int] = {}
    stack = [(tree.root_node, None, 0, 0)]
    while stack and len(features) < MAX_AST_NODES:
        node, parent_idx, depth, sibling_index = stack.pop()
        current_idx = len(features)
        node_index_by_key[node_key(node)] = current_idx
        features.append(node_feature(node, depth, sibling_index))
        node_types.append(node.type)
        node_texts.append(node.text.decode("utf-8", errors="ignore") if node.type == "identifier" else "")
        if parent_idx is not None:
            add_edge(parent_idx, current_idx, "parent_to_child")
            add_edge(current_idx, parent_idx, "child_to_parent")
            previous_sibling = last_child_by_parent.get(parent_idx)
            if previous_sibling is not None:
                add_edge(previous_sibling, current_idx, "next_sibling")
                add_edge(current_idx, previous_sibling, "prev_sibling")
            last_child_by_parent[parent_idx] = current_idx
        if node.type in {"if_statement", "for_statement", "enhanced_for_statement", "while_statement", "do_statement", "switch_statement"}:
            control_nodes.append(current_idx)
        if node.type in {"method_invocation", "field_access"}:
            receiver = node.child_by_field_name("object")
            if receiver is not None:
                pending_receiver_edges.append((node_key(receiver), current_idx))

        children = list(node.children)
        for child_offset in range(len(children) - 1, -1, -1):
            stack.append((children[child_offset], current_idx, depth + 1, child_offset))

    if not features:
        fake = type("FakeNode", (), {"type": "program", "children": [], "is_named": True})()
        features.append(node_feature(fake, 0, 0))

    for receiver_key, access_idx in pending_receiver_edges:
        receiver_idx = node_index_by_key.get(receiver_key)
        if receiver_idx is not None:
            add_edge(receiver_idx, access_idx, "member_receiver")
            add_edge(access_idx, receiver_idx, "member_receiver")

    last_identifier_by_name = {}
    identifier_occurrences = {}
    for idx, (node_type, node_text) in enumerate(zip(node_types, node_texts)):
        if node_type != "identifier" or len(node_text) < MIN_SAME_IDENTIFIER_LENGTH:
            continue
        occurrence_count = identifier_occurrences.get(node_text, 0)
        if occurrence_count >= MAX_SAME_IDENTIFIER_OCCURRENCES:
            continue
        if node_text in last_identifier_by_name:
            add_edge(last_identifier_by_name[node_text], idx, "same_identifier")
            add_edge(idx, last_identifier_by_name[node_text], "same_identifier")
        last_identifier_by_name[node_text] = idx
        identifier_occurrences[node_text] = occurrence_count + 1

    statement_types = {"block", "expression_statement", "local_variable_declaration", "return_statement", "method_invocation"}
    for control_idx in control_nodes:
        for candidate_idx in range(control_idx + 1, min(control_idx + 32, len(node_types))):
            if node_types[candidate_idx] in statement_types:
                add_edge(control_idx, candidate_idx, "control_flow_hint")
                break

    x = torch.tensor(features, dtype=torch.float)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty((2, 0), dtype=torch.long)
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float) if edge_attrs else torch.empty((0, EDGE_FEATURE_DIM), dtype=torch.float)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr), "ok"


def ast_cache_metadata(language: str) -> dict[str, object]:
    return {
        "schema_version": AST_SCHEMA_VERSION,
        "language": language,
        "max_ast_nodes": MAX_AST_NODES,
        "node_type_hash_buckets": NODE_TYPE_HASH_BUCKETS,
        "edge_types": EDGE_TYPES,
        "min_same_identifier_length": MIN_SAME_IDENTIFIER_LENGTH,
        "max_same_identifier_occurrences": MAX_SAME_IDENTIFIER_OCCURRENCES,
    }


def cache_ast_graphs(split_df: pd.DataFrame, paths: Paths, language: str, force: bool, logger: RunLogger) -> None:
    metadata = ast_cache_metadata(language)
    if paths.ast_graphs.exists() and paths.ast_parse_result.exists() and not force:
        expected_ids = set(split_df["sample_id"].astype(str))
        if cache_metadata_matches(paths.ast_cache_meta, metadata) and tensor_cache_ids_match(
            paths.ast_graphs,
            paths.ast_cache_ids,
            expected_ids,
            logger,
        ):
            logger.info(f"Reuse AST graph cache/parse audit: {paths.ast_graphs}, {paths.ast_parse_result}")
            return
        logger.info("AST cache schema or IDs changed; rebuilding AST graph cache.")

    logger.info("Start AST parse and graph cache.")
    source_by_id = load_source_cache(paths)
    parser = java_parser()
    graph_by_id = {}
    parse_rows = []
    for row in tqdm(split_df.to_dict("records"), desc="parse ast"):
        sample_id = row["sample_id"]
        try:
            graph, status = code_to_graph(source_by_id[sample_id], parser, language)
        except Exception as exc:
            fake = type("FakeNode", (), {"type": "program", "children": [], "is_named": True})()
            graph = Data(
                x=torch.tensor([node_feature(fake, 0, 0)], dtype=torch.float),
                edge_index=torch.empty((2, 0), dtype=torch.long),
                edge_attr=torch.empty((0, EDGE_FEATURE_DIM), dtype=torch.float),
            )
            status = f"fallback:{type(exc).__name__}"
        graph_by_id[sample_id] = graph
        parse_rows.append(
            {
                "sample_id": sample_id,
                "status": status,
                "num_nodes": int(graph.num_nodes),
                "num_edges": int(graph.edge_index.size(1)),
            }
        )
    torch.save(graph_by_id, paths.ast_graphs)
    write_cache_ids(paths.ast_cache_ids, set(map(str, graph_by_id.keys())))
    write_cache_metadata(paths.ast_cache_meta, metadata)
    pd.DataFrame(parse_rows).to_csv(paths.ast_parse_result, index=False)
    status_counts = pd.Series([row["status"] for row in parse_rows]).value_counts().to_dict()
    logger.info(f"Wrote AST graphs rows={len(parse_rows)} status_counts={status_counts} file={paths.ast_parse_result}")


def prepare_artifacts(args: argparse.Namespace, paths: Paths, logger: RunLogger) -> pd.DataFrame:
    positive_n, negative_n = smell_sample_size(args.smell, args.positive, args.negative, args.dev)
    logger.info(f"Prepare artifacts requested_positive={positive_n} requested_negative={negative_n}")
    force_index = FORCE_REBUILD_INDEX or args.force_rebuild_index
    force_sample = FORCE_RESAMPLE or args.force_resample or force_index
    index_df = build_dataset_index(args.language, args.smell, paths, force_index, logger)
    split_df, sample_rebuilt = sample_and_split(index_df, paths, positive_n, negative_n, force_sample, logger)
    downstream_changed = force_sample or sample_rebuilt
    valid_source_ids = cache_sampled_sources(split_df, paths, downstream_changed, logger)
    split_df = filter_and_resplit_missing_sources(split_df, valid_source_ids, paths, logger)
    tokenizer = AutoTokenizer.from_pretrained(UNIXCODER_NAME)
    cache_tokenization(
        split_df,
        paths,
        tokenizer,
        args.smell,
        FORCE_REBUILD_TOKENS or args.force_rebuild_tokens or downstream_changed,
        logger,
    )
    cache_ast_graphs(
        split_df,
        paths,
        args.language,
        FORCE_REBUILD_AST or args.force_rebuild_ast or downstream_changed,
        logger,
    )
    return split_df


class CachedSmellDataset(Dataset):
    def __init__(self, split_df: pd.DataFrame, split: str, token_by_id: dict[str, dict[str, list[int]]], graph_by_id: dict[str, Data]):
        self.df = split_df[split_df["split"] == split].reset_index(drop=True)
        self.token_by_id = token_by_id
        self.graph_by_id = graph_by_id

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, object]:
        row = self.df.iloc[idx]
        sample_id = row["sample_id"]
        tokens = self.token_by_id[sample_id]
        return {
            "input_ids": torch.tensor(tokens["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(tokens["attention_mask"], dtype=torch.long),
            "graph": self.graph_by_id[sample_id],
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
        }


def collate_fn(batch: list[dict[str, object]], tokenizer) -> dict[str, object]:
    encoded = tokenizer.pad(
        [{"input_ids": item["input_ids"], "attention_mask": item["attention_mask"]} for item in batch],
        return_tensors="pt",
    )
    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "graph_batch": Batch.from_data_list([item["graph"] for item in batch]),
        "labels": torch.stack([item["label"] for item in batch]),
    }


def make_sampler(dataset: CachedSmellDataset) -> WeightedRandomSampler:
    labels = dataset.df["label"].astype(int).tolist()
    counts = pd.Series(labels).value_counts().to_dict()
    weights = [1.0 / counts[label] for label in labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def build_model(model_name: str) -> nn.Module:
    if model_name == "unixcoder":
        return UnixCoderClassifier(UNIXCODER_NAME, dropout=DROPOUT)
    if model_name == "ast_gat":
        return ASTGATClassifier(NODE_FEATURE_DIM, GAT_HIDDEN_DIM, GAT_HEADS, GAT_LAYERS, EDGE_FEATURE_DIM, dropout=DROPOUT)
    if model_name == "fusion":
        return FusionClassifier(UNIXCODER_NAME, NODE_FEATURE_DIM, GAT_HIDDEN_DIM, GAT_HEADS, GAT_LAYERS, EDGE_FEATURE_DIM, dropout=DROPOUT)
    raise ValueError(f"Unknown model: {model_name}")


def move_batch(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    return {
        "input_ids": batch["input_ids"].to(device),
        "attention_mask": batch["attention_mask"].to(device),
        "graph_batch": batch["graph_batch"].to(device),
        "labels": batch["labels"].to(device),
    }


def compute_metrics(labels: list[int], probs: list[float], threshold: float = 0.5) -> dict[str, float]:
    y_true = np.asarray(labels)
    y_prob = np.asarray(probs)
    y_pred = (y_prob >= threshold).astype(int)
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "pr_auc": average_precision_score(y_true, y_prob),
    }
    try:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
    except ValueError:
        metrics["roc_auc"] = float("nan")
    return {key: float(value) for key, value in metrics.items()}


def tune_threshold(labels: list[int], probs: list[float]) -> tuple[float, dict[str, float]]:
    best_threshold = 0.5
    best_metrics = None
    for threshold in np.linspace(0.05, 0.95, 91):
        current_threshold = float(threshold)
        metrics = compute_metrics(labels, probs, threshold=current_threshold)
        if is_better_threshold(metrics, best_metrics, current_threshold, best_threshold):
            best_threshold = current_threshold
            best_metrics = metrics
    return best_threshold, best_metrics if best_metrics is not None else compute_metrics(labels, probs)


def run_epoch(model, loader, criterion, device, is_train: bool, optimizer=None, scheduler=None, threshold: float = 0.5):
    model.train(is_train)
    total_loss = 0.0
    labels = []
    probs = []
    for batch in tqdm(loader, leave=False, desc="train" if is_train else "eval"):
        batch = move_batch(batch, device)
        with torch.set_grad_enabled(is_train):
            logits = model(batch["input_ids"], batch["attention_mask"], batch["graph_batch"])
            loss = criterion(logits, batch["labels"])
            if is_train:
                if optimizer is None:
                    raise ValueError("optimizer is required for training")
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
        total_loss += float(loss.item()) * batch["labels"].size(0)
        prob = torch.softmax(logits.detach(), dim=-1)[:, 1]
        labels.extend(batch["labels"].detach().cpu().tolist())
        probs.extend(prob.cpu().tolist())
    return total_loss / max(1, len(loader.dataset)), compute_metrics(labels, probs, threshold), labels, probs


def class_weights(split_df: pd.DataFrame, device: torch.device) -> torch.Tensor:
    train_df = split_df[split_df["split"] == "train"]
    counts = train_df["label"].value_counts().to_dict()
    total = len(train_df)
    return torch.tensor([total / (2 * counts.get(0, 1)), total / (2 * counts.get(1, 1))], dtype=torch.float, device=device)


def is_better(candidate: dict[str, float], best: dict[str, float] | None, epoch: int, best_epoch: int) -> bool:
    if best is None:
        return True
    eps = 1e-4
    if candidate["mcc"] > best["mcc"] + eps:
        return True
    if abs(candidate["mcc"] - best["mcc"]) <= eps and candidate["f1"] > best["f1"] + eps:
        return True
    if abs(candidate["mcc"] - best["mcc"]) <= eps and abs(candidate["f1"] - best["f1"]) <= eps:
        return epoch < best_epoch
    return False


def is_better_threshold(
    candidate: dict[str, float],
    best: dict[str, float] | None,
    threshold: float,
    best_threshold: float,
) -> bool:
    if best is None:
        return True
    eps = 1e-4
    if candidate["mcc"] > best["mcc"] + eps:
        return True
    if abs(candidate["mcc"] - best["mcc"]) <= eps and candidate["f1"] > best["f1"] + eps:
        return True
    if abs(candidate["mcc"] - best["mcc"]) <= eps and abs(candidate["f1"] - best["f1"]) <= eps:
        return abs(threshold - 0.5) < abs(best_threshold - 0.5)
    return False


def result_input_type(model_name: str, target_smell: str) -> str:
    source_input = "method source + containing class context" if target_smell == "FeatureEnvy" else "raw method source"
    if model_name == "unixcoder":
        return source_input
    if model_name == "ast_gat":
        return "typed AST graph with member-receiver edges"
    return f"{source_input} + typed AST graph"


def write_result(args, paths: Paths, split_df: pd.DataFrame, metrics: dict[str, float], runtime: float) -> None:
    split_sizes = split_df["split"].value_counts().to_dict()
    split_protocol = (
        str(split_df["split_protocol"].iloc[0])
        if "split_protocol" in split_df and not split_df.empty
        else "70/15/15 protocol unavailable"
    )
    detected_hardware = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    hardware = args.hardware or os.environ.get("HARDWARE") or detected_hardware
    target_smell = getattr(args, "target_smell", args.smell)
    row = {
        "member": args.member,
        "task_name": TASK_NAME,
        "dataset_name": "DeepLearningSmells",
        "language": args.language.capitalize(),
        "smells": target_smell,
        "input_type": result_input_type(args.model, target_smell),
        "split_protocol": split_protocol,
        "seed": SEED,
        "train_size": split_sizes.get("train", 0),
        "validation_size": split_sizes.get("val", 0),
        "test_size": split_sizes.get("test", 0),
        "model": MODEL_DISPLAY_NAMES.get(args.model, args.model),
        "checkpoint_selection": "best validation MCC then F1",
        "hardware": hardware,
        "runtime": f"{runtime:.1f}s",
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "mcc": metrics["mcc"],
        "roc_auc": metrics["roc_auc"],
        "notes": (
            f"target_smell={target_smell}; PR-AUC={metrics['pr_auc']:.6f}; "
            f"checkpoint={paths.best_checkpoint if SAVE_BEST_MODEL and not args.no_save_model else 'not_saved'}"
        ),
    }
    exists = paths.result_csv.exists()
    with paths.result_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def train_eval_test(args: argparse.Namespace, paths: Paths, split_df: pd.DataFrame, logger: RunLogger) -> None:
    logger.info("Start train/eval/test.")
    tokenizer = AutoTokenizer.from_pretrained(UNIXCODER_NAME)
    token_by_id = load_trusted_artifact(paths.token_cache, map_location="cpu")
    graph_by_id = load_trusted_artifact(paths.ast_graphs, map_location="cpu")

    train_ds = CachedSmellDataset(split_df, "train", token_by_id, graph_by_id)
    val_ds = CachedSmellDataset(split_df, "val", token_by_id, graph_by_id)
    test_ds = CachedSmellDataset(split_df, "test", token_by_id, graph_by_id)
    logger.info(f"Dataset sizes train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    collate = partial(collate_fn, tokenizer=tokenizer)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=make_sampler(train_ds), collate_fn=collate, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate, num_workers=NUM_WORKERS)

    device = get_device()
    model = build_model(args.model).to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = FocalLoss(alpha=class_weights(split_df, device), gamma=FOCAL_GAMMA)
    scheduler = None
    if not args.eval_only and args.model in {"unixcoder", "fusion"}:
        total_steps = max(1, args.epochs * len(train_loader))
        warmup_steps = round(total_steps * WARMUP_RATIO)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        logger.info(f"Linear LR schedule total_steps={total_steps} warmup_steps={warmup_steps}")

    started = time.time()
    best_metrics = None
    best_epoch = 10**9
    best_threshold = 0.5
    stale_epochs = 0

    if args.eval_only:
        logger.info(f"Eval-only mode loading checkpoint: {paths.best_checkpoint}")
        if not paths.best_checkpoint.exists():
            raise FileNotFoundError(f"Missing checkpoint for eval-only: {paths.best_checkpoint}")
        checkpoint = load_trusted_artifact(paths.best_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        best_threshold = float(checkpoint.get("threshold", 0.5))
    else:
        for epoch in range(1, args.epochs + 1):
            train_loss, train_metrics, _, _ = run_epoch(
                model,
                train_loader,
                criterion,
                device,
                is_train=True,
                optimizer=optimizer,
                scheduler=scheduler,
            )
            val_loss, _, val_labels, val_probs = run_epoch(model, val_loader, criterion, device, is_train=False)
            threshold, val_metrics = tune_threshold(val_labels, val_probs)
            print(
                f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"val_mcc={val_metrics['mcc']:.4f} val_f1={val_metrics['f1']:.4f} threshold={threshold:.2f}"
            )
            logger.info(
                f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"val_mcc={val_metrics['mcc']:.4f} val_f1={val_metrics['f1']:.4f} threshold={threshold:.2f}"
            )
            if is_better(val_metrics, best_metrics, epoch, best_epoch):
                best_metrics = val_metrics
                best_epoch = epoch
                best_threshold = threshold
                stale_epochs = 0
                if SAVE_BEST_MODEL and not args.no_save_model:
                    torch.save(
                        {"model_state": model.state_dict(), "epoch": epoch, "val_metrics": val_metrics, "threshold": threshold},
                        paths.best_checkpoint,
                    )
                    logger.info(f"Saved best checkpoint epoch={epoch} path={paths.best_checkpoint}")
            else:
                stale_epochs += 1
                if stale_epochs >= PATIENCE:
                    logger.info(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
                    break

        if SAVE_BEST_MODEL and not args.no_save_model and paths.best_checkpoint.exists():
            checkpoint = load_trusted_artifact(paths.best_checkpoint, map_location=device)
            model.load_state_dict(checkpoint["model_state"])
            best_threshold = float(checkpoint.get("threshold", best_threshold))

    test_loss, test_metrics, _, _ = run_epoch(model, test_loader, criterion, device, is_train=False, threshold=best_threshold)
    runtime = time.time() - started
    logger.info(f"test_loss={test_loss:.4f} threshold={best_threshold:.2f} test_metrics={test_metrics}")
    write_result(args, paths, split_df, test_metrics, runtime)
    logger.info(f"Result appended to: {paths.result_csv}")


def main() -> None:
    args = parse_args()
    set_seed(SEED)
    smells = resolve_smells(args.smell)
    models = resolve_models(args.model)

    for current_smell in smells:
        prepare_args = argparse.Namespace(**vars(args))
        prepare_args.smell = current_smell
        prepare_args.target_smell = current_smell
        prepare_paths = get_paths(prepare_args.language, current_smell, "prepare")
        prepare_logger = RunLogger(prepare_paths.log_file)

        print(f"\n===== {prepare_args.language}/{current_smell}/prepare =====")
        prepare_logger.info(
            f"Prepare start language={prepare_args.language} smell={current_smell} "
            f"models={models} dev={prepare_args.dev}"
        )
        try:
            split_df = prepare_artifacts(prepare_args, prepare_paths, prepare_logger)
            prepare_logger.info(f"Artifacts ready in: {prepare_paths.artifact_dir}")
        except Exception:
            prepare_logger.exception("Prepare failed.")
            raise

        if args.prepare_only:
            prepare_logger.info("Prepare-only run finished successfully.")
            continue

        for current_model in models:
            run_args = argparse.Namespace(**vars(prepare_args))
            run_args.model = current_model
            paths = get_paths(run_args.language, current_smell, current_model)
            logger = RunLogger(paths.log_file)
            print(f"\n===== {run_args.language}/{current_smell}/{current_model} =====")
            logger.info(
                f"Model run start language={run_args.language} smell={current_smell} "
                f"model={current_model} dev={run_args.dev}"
            )
            try:
                set_seed(SEED)
                train_eval_test(run_args, paths, split_df, logger)
                logger.info("Model run finished successfully.")
            except Exception:
                logger.exception("Model run failed.")
                raise


if __name__ == "__main__":
    main()
