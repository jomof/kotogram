"""Dataset and processing logic for style classification (V2 Binary / Memory-Mapped)."""

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, cast

import torch
from torch.utils.data import Dataset

from kotogram.japanese_parser import JapaneseParser
from kotogram.model import (
    NUM_FORMALITY_PRAGMATIC_CLASSES,
    NUM_GENDER_PRAGMATIC_CLASSES,
    NUM_GRAMMATICALITY_CLASSES,
    NUM_REGISTER_CLASSES,
)
from kotogram.tokenizer import FEATURE_FIELDS, Tokenizer
from train.binary_io import (
    EXT_FEAT_PREFIX,
    EXT_KC_PREFIX,
    EXT_LABELS,
    EXT_OFFSETS,
)
from train.kc import KcFamilyId, compute_kc_targets, initialize_disallow_filter
from train.types import Sample, TrainingBatch

# V2: No cache version check needed for raw binary files (handled by label.py generation)


@dataclass
class DatasetConfig:
    """Configuration for StyleDataset loading and processing."""

    parser: Optional[JapaneseParser] = None
    verbose: bool = True
    grammaticality_labels: Optional[List[int]] = (
        None  # Ignored in V2 (filtering done via subsetting if needed)
    )

    cache_name: Optional[str] = "vocab.json"
    sample_ratio: float = 1.0


class StyleDataset(Dataset[Sample]):
    """
    PyTorch Dataset for style classification using memory-mapped binary files.
    replaces the legacy dictionary-based cache.
    """

    def __init__(
        self,
        data_dir: str,
        tokenizer: Tokenizer,
        indices: Optional[torch.Tensor] = None,
        sample_ratio: float = 1.0,
    ):
        # pylint: disable=too-many-positional-arguments
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.verbose = True

        # Initialize disallow filter for KC target computation
        compound_1_vocab = tokenizer.field_vocabs.get("compound_1", {})
        initialize_disallow_filter(compound_1_vocab)

        # Load Main Offsets (Sentences)
        offsets_path = os.path.join(data_dir, EXT_OFFSETS)
        if not self._check_exists(offsets_path):
            raise FileNotFoundError(
                f"Dataset offsets not found at {offsets_path}. Please run 'bin/train_style --label'."
            )

        # MMap Offsets (Int32)
        # Note: torch.from_file requires size (number of elements)
        size_bytes = self._get_size(offsets_path)
        self.offsets = self._load_tensor(
            offsets_path, shared=True, size=size_bytes // 4, dtype=torch.int32
        )

        # Determine total samples
        total_samples = len(self.offsets) - 1

        # Handle Indices (Subsetting / Splitting)
        if indices is not None:
            self.indices = indices
        else:
            # Default: use all
            self.indices = torch.arange(total_samples, dtype=torch.long)

        self.features = self._init_features(data_dir)
        self.labels = self._init_labels(data_dir)

        # Per-grammar-point priors (float32 vector; index == gp numeric id).
        # NaN means "unset, use defaults". Required when GP targets are present.
        self.gp_priors: torch.Tensor

        # Register Ragged Labels
        reg_ids_path = os.path.join(data_dir, f"{EXT_LABELS}_reg_ids.bin")
        if self._check_exists(reg_ids_path):
            sz = self._get_size(reg_ids_path) // 4
            self.labels["reg_ids"] = self._load_tensor(
                reg_ids_path, shared=True, size=sz, dtype=torch.int32
            )

        reg_off_path = os.path.join(data_dir, f"{EXT_LABELS}_reg_ids_{EXT_OFFSETS}")
        if self._check_exists(reg_off_path):
            sz = self._get_size(reg_off_path) // 4
            self.labels["reg_offsets"] = self._load_tensor(
                reg_off_path, shared=True, size=sz, dtype=torch.int32
            )

        self._apply_balanced_sampling(sample_ratio)

        self._len = len(self.indices)

        self.kc_maps: Dict[str, Dict[str, torch.Tensor]] = self._init_kc_targets(
            self.data_dir
        )

    def _apply_balanced_sampling(self, sample_ratio: float) -> None:
        if sample_ratio == 1.0 or "gram" not in self.labels:
            return
        if self.verbose:
            print(
                f"Sampling {sample_ratio:.1%} of grammatic and matching ungrammatic..."
            )
        current_labels = self.labels["gram"][self.indices]
        gram_indices = self.indices[current_labels == 1]
        ungram_indices = self.indices[current_labels == 0]

        gram_total = int(gram_indices.numel())
        ungram_total = int(ungram_indices.numel())

        if sample_ratio <= 1.0:
            target_gram = int(gram_total * sample_ratio)
            target_ungram = target_gram
        else:
            target_gram = gram_total
            target_ungram = int(gram_total * sample_ratio)

        target_gram = min(target_gram, gram_total)
        target_ungram = min(target_ungram, ungram_total)

        gram_sel = (
            gram_indices[torch.randperm(gram_total)[:target_gram]]
            if gram_total > 0 and target_gram > 0
            else torch.tensor([], dtype=torch.long)
        )
        ungram_sel = (
            ungram_indices[torch.randperm(ungram_total)[:target_ungram]]
            if ungram_total > 0 and target_ungram > 0
            else torch.tensor([], dtype=torch.long)
        )

        if gram_sel.numel() or ungram_sel.numel():
            self.indices = torch.cat([gram_sel, ungram_sel])
            self.indices, _ = torch.sort(self.indices)

    def _check_exists(self, path: str) -> bool:
        return os.path.exists(path)

    def _get_size(self, path: str) -> int:
        return os.path.getsize(path)

    def _load_tensor(
        self, path: str, shared: bool, size: int, dtype: torch.dtype
    ) -> torch.Tensor:
        return torch.from_file(path, shared=shared, size=size, dtype=dtype)

    def _init_features(self, data_dir: str) -> Dict[str, torch.Tensor]:
        features = {}
        for field in FEATURE_FIELDS:
            path = os.path.join(data_dir, f"{EXT_FEAT_PREFIX}{field}.bin")
            if self._check_exists(path):
                sz = self._get_size(path) // 4
                features[field] = self._load_tensor(
                    path, shared=True, size=sz, dtype=torch.int32
                )
        return features

    def _init_labels(self, data_dir: str) -> Dict[str, torch.Tensor]:
        labels = {}
        label_specs = [
            ("f_val", torch.float32, 4),
            ("f_prag", torch.uint8, 1),
            ("g_val", torch.float32, 4),
            ("g_prag", torch.uint8, 1),
            ("gram", torch.uint8, 1),
        ]

        for name, dtype, itemsize in label_specs:
            fname = f"{EXT_LABELS}_{name}"
            path = os.path.join(data_dir, fname)
            if self._check_exists(path):
                sz = self._get_size(path) // itemsize
                # Note: valid typecodes for array.array 'B' is unsigned char -> uint8
                if name == "gram" or "prag" in name:
                    # torch.from_file supports uint8 (ByteTensor)
                    pass
                labels[name] = self._load_tensor(
                    path, shared=True, size=sz, dtype=dtype
                )
        return labels

    def _init_kc_targets(self, data_dir: str) -> Dict[str, Dict[str, torch.Tensor]]:
        """Load pre-computed KC targets from binary files.

        The label phase writes KC targets as jagged arrays:
        - kc_{family}_ids.bin: flat array of KC IDs
        - kc_{family}_offsets.bin: offset array for sample boundaries

        Returns:
            Dict mapping family name to {"ids": tensor, "offsets": tensor}.
        """
        kc_maps: Dict[str, Dict[str, torch.Tensor]] = {}

        # Scan for all KC families that were pre-computed
        for family in KcFamilyId:
            family_name = family.value  # e.g., "bag_reading_gram"
            ids_path = os.path.join(data_dir, f"{EXT_KC_PREFIX}{family_name}_ids.bin")
            off_path = os.path.join(
                data_dir, f"{EXT_KC_PREFIX}{family_name}_{EXT_OFFSETS}"
            )

            if self._check_exists(ids_path) and self._check_exists(off_path):
                ids_sz = self._get_size(ids_path) // 4
                off_sz = self._get_size(off_path) // 4
                kc_maps[family_name] = {
                    "ids": self._load_tensor(
                        ids_path, shared=True, size=ids_sz, dtype=torch.int32
                    ),
                    "offsets": self._load_tensor(
                        off_path, shared=True, size=off_sz, dtype=torch.int32
                    ),
                }

        # Load grammar point targets (pos/neg jagged arrays from DB source)
        gp_pos_ids_path = os.path.join(data_dir, "gp_pos_ids.bin")
        gp_pos_off_path = os.path.join(data_dir, "gp_pos_offsets.bin")
        if self._check_exists(gp_pos_ids_path) and self._check_exists(gp_pos_off_path):
            kc_maps["grammar_point_pos"] = {
                "ids": self._load_tensor(
                    gp_pos_ids_path,
                    shared=True,
                    size=self._get_size(gp_pos_ids_path) // 4,
                    dtype=torch.int32,
                ),
                "offsets": self._load_tensor(
                    gp_pos_off_path,
                    shared=True,
                    size=self._get_size(gp_pos_off_path) // 4,
                    dtype=torch.int32,
                ),
            }

        gp_neg_ids_path = os.path.join(data_dir, "gp_neg_ids.bin")
        gp_neg_off_path = os.path.join(data_dir, "gp_neg_offsets.bin")
        if self._check_exists(gp_neg_ids_path) and self._check_exists(gp_neg_off_path):
            kc_maps["grammar_point_neg"] = {
                "ids": self._load_tensor(
                    gp_neg_ids_path,
                    shared=True,
                    size=self._get_size(gp_neg_ids_path) // 4,
                    dtype=torch.int32,
                ),
                "offsets": self._load_tensor(
                    gp_neg_off_path,
                    shared=True,
                    size=self._get_size(gp_neg_off_path) // 4,
                    dtype=torch.int32,
                ),
            }

        # Optional per-GP priors vector
        gp_priors_path = os.path.join(data_dir, "gp_priors.bin")
        if (
            "grammar_point_pos" in kc_maps or "grammar_point_neg" in kc_maps
        ) and not self._check_exists(gp_priors_path):
            raise FileNotFoundError(
                f"Missing gp_priors.bin at {gp_priors_path} while grammar point targets are present. "
                "Run scripts/label.py against a corpus.db with grammar.prior column so the cache "
                "includes per-GP priors (NaN for unset priors)."
            )
        if self._check_exists(gp_priors_path):
            sz = self._get_size(gp_priors_path) // 4
            self.gp_priors = self._load_tensor(
                gp_priors_path, shared=True, size=sz, dtype=torch.float32
            )
        else:
            # No GP targets; keep an empty tensor.
            self.gp_priors = torch.empty((0,), dtype=torch.float32)

        return kc_maps

    def __len__(self) -> int:
        return self._len

    def get_sentence_by_idx(self, real_idx: int) -> str:
        """Get original sentence text for a sample by its real dataset index.

        Uses linecache for efficient random access to sentences.txt which is
        written during the --label phase.
        """
        import linecache

        sentences_path = os.path.join(self.data_dir, "sentences.txt")
        if not os.path.exists(sentences_path):
            return ""
        # linecache is 1-indexed
        line = linecache.getline(sentences_path, real_idx + 1)
        return line.strip()

    def __getitem__(self, idx: int) -> Sample:
        # Resolve real index
        real_idx = int(self.indices[idx].item())

        # Get start/end (tokens)
        start = int(self.offsets[real_idx].item())
        end = int(self.offsets[real_idx + 1].item())

        # Features
        feature_ids: Dict[str, Any] = {}
        for field, tensor in self.features.items():
            # Slice (View) - defer copy to collate
            feature_ids[field] = tensor[int(start) : int(end)]

        # Labels
        # f_val, etc are 1:1 with samples
        # f_val = self.labels["f_val"][real_idx].item()
        # f_prag = self.labels["f_prag"][real_idx].item()
        # g_val = self.labels["g_val"][real_idx].item()
        # g_prag = self.labels["g_prag"][real_idx].item()
        # gram = self.labels["gram"][real_idx].item()

        # Register (Ragged)
        r_start = self.labels["reg_offsets"][int(real_idx)].item()
        r_end = self.labels["reg_offsets"][int(real_idx) + 1].item()
        reg_list = self.labels["reg_ids"][int(r_start) : int(r_end)].tolist()
        if not reg_list:
            reg_list = [0]

        return Sample(
            feature_ids=feature_ids,
            formality_value=float(self.labels["f_val"][int(real_idx)].item()),
            formality_pragmatic=int(self.labels["f_prag"][int(real_idx)].item()),
            gender_value=float(self.labels["g_val"][int(real_idx)].item()),
            gender_pragmatic=int(self.labels["g_prag"][int(real_idx)].item()),
            grammaticality_label=int(self.labels["gram"][int(real_idx)].item()),
            register_labels=reg_list,
            original_sentence="",  # Binary format drops raw text to save space
            kotogram="",
            kc_targets=self._get_kc_targets(
                int(real_idx), int(start), int(end), feature_ids
            ),
            idx=int(real_idx),
        )

    def _get_kc_targets(
        self, real_idx: int, _start: int, _end: int, feature_ids: Dict[str, List[int]]
    ) -> Dict[KcFamilyId, Any]:
        """Get KC targets for a sample.

        Uses pre-computed targets from binary files if available,
        otherwise falls back to computing on-the-fly.

        For GRAMMAR_POINT family, returns a dict with "pos" and "neg" lists.
        """
        # If pre-computed KC targets are available, use them
        if self.kc_maps:
            result: Dict[KcFamilyId, Any] = {}
            for family in KcFamilyId:
                family_name = family.value

                # Special handling for GRAMMAR_POINT (has pos/neg arrays)
                if family == KcFamilyId.GRAMMAR_POINT:
                    gp_pos = []
                    gp_neg = []
                    if "grammar_point_pos" in self.kc_maps:
                        data = self.kc_maps["grammar_point_pos"]
                        gp_start = int(data["offsets"][real_idx].item())
                        gp_end = int(data["offsets"][real_idx + 1].item())
                        gp_pos = data["ids"][gp_start:gp_end].tolist()
                    if "grammar_point_neg" in self.kc_maps:
                        data = self.kc_maps["grammar_point_neg"]
                        gp_start = int(data["offsets"][real_idx].item())
                        gp_end = int(data["offsets"][real_idx + 1].item())
                        gp_neg = data["ids"][gp_start:gp_end].tolist()
                    result[family] = {"pos": gp_pos, "neg": gp_neg}
                    continue

                if family_name in self.kc_maps:
                    data = self.kc_maps[family_name]
                    # Get the jagged slice for this sample
                    kc_start = int(data["offsets"][real_idx].item())
                    kc_end = int(data["offsets"][real_idx + 1].item())
                    result[family] = data["ids"][kc_start:kc_end].tolist()
                else:
                    result[family] = []
            return result

        # Fallback: Compute on demand (slower, for backwards compatibility)
        return compute_kc_targets(cast(Any, feature_ids))

    @classmethod
    def from_multiple_tsv(
        cls,
        _tsv_paths: List[str],
        tokenizer: Tokenizer,
        config: Optional[DatasetConfig] = None,
    ) -> "StyleDataset":
        """
        Factory method to load StyleDataset.
        Ignores tsv_paths in V2, looking instead for binary cache.
        """
        if config is None:
            config = DatasetConfig()

        from train import paths as train_paths

        cache_dir = train_paths.get_style_dataset_cache_dir()

        # Check for binary cache
        if not os.path.exists(os.path.join(cache_dir, EXT_OFFSETS)):
            raise FileNotFoundError(
                f"Binary dataset not found in {cache_dir}. "
                "Please run 'bin/train_style --label' to formulate the dataset."
            )

        if config.verbose:
            print(f"Loading binary dataset from {cache_dir}...")

        return cls(
            data_dir=cache_dir,
            tokenizer=tokenizer,
            sample_ratio=config.sample_ratio,
        )

    def split(
        self, train_ratio: float = 0.8, seed: int = 42
    ) -> Tuple["StyleDataset", "StyleDataset"]:
        """Split dataset into train and validation."""
        # pylint: disable=attribute-defined-outside-init
        # Subset indices
        torch.manual_seed(seed)
        total = len(self)
        perm = torch.randperm(total)
        n_train = int(total * train_ratio)
        train_idx = self.indices[perm[:n_train]]
        val_idx = self.indices[perm[n_train:]]

        # Return new objects sharing same mmaps (handled by init)
        train_ds = StyleDataset(self.data_dir, self.tokenizer, indices=train_idx)
        val_ds = StyleDataset(self.data_dir, self.tokenizer, indices=val_idx)

        # Share the big mmaps to save file descriptors?
        # Python instances share underlying storage?
        # torch.from_file maps are file descriptors.
        # If I create new StyleDataset, it opens new FDs.
        # Better: Pass existing mmaps?
        # Refactor __init__ to accept shared resources.

        train_ds.features = self.features
        train_ds.labels = self.labels
        train_ds.offsets = self.offsets
        train_ds.kc_maps = self.kc_maps

        return train_ds, val_ds

    def filter_by_grammaticality(self, label: int = 1) -> "StyleDataset":
        """Return a new dataset subset with only samples confirming to the grammaticality label."""
        # pylint: disable=attribute-defined-outside-init
        if "gram" not in self.labels:
            if self.verbose:
                print("Warning: No grammaticality labels found, returning self.")
            return self

        # Get labels for current indices
        current_labels = self.labels["gram"][self.indices]

        # Filter
        mask = current_labels == label
        new_indices = self.indices[mask]

        # Create new dataset sharing mmaps
        filtered_ds = StyleDataset(self.data_dir, self.tokenizer, indices=new_indices)

        # Optimization: Share existing mmaps manually to skip re-opening files
        filtered_ds.features = self.features
        filtered_ds.labels = self.labels
        filtered_ds.offsets = self.offsets
        filtered_ds.kc_maps = self.kc_maps

        return filtered_ds

    def create_style_oversampler(
        self,
        formality_boost: float = 5.0,
        gender_boost: float = 15.0,
    ) -> "torch.utils.data.WeightedRandomSampler":
        """Create a sampler that oversamples non-neutral formality and gender examples.

        Args:
            formality_boost: Multiplier for non-neutral formality (|f| > 0.25)
            gender_boost: Multiplier for non-neutral gender (|g| > 0.25)

        Returns:
            WeightedRandomSampler for use in DataLoader
        """
        from torch.utils.data import WeightedRandomSampler

        weights = torch.ones(len(self), dtype=torch.float32)

        # Get formality and gender values for all samples in the dataset
        for i in range(len(self)):
            real_idx = self.indices[i]
            f_val = float(self.labels["f_val"][int(real_idx)].item())
            g_val = float(self.labels["g_val"][int(real_idx)].item())

            # Boost non-neutral formality (|f| > 0.25)
            if abs(f_val) > 0.25:
                weights[i] *= formality_boost

            # Boost non-neutral gender (|g| > 0.25)
            if abs(g_val) > 0.25:
                weights[i] *= gender_boost

        # Use replacement=True to allow oversampling
        return WeightedRandomSampler(
            weights.tolist(), num_samples=len(self), replacement=True
        )

    def get_formality_class_weights(self) -> torch.Tensor:
        # compute from self.labels["f_prag"] (subset by self.indices)
        if "f_prag" not in self.labels:
            return torch.ones(NUM_FORMALITY_PRAGMATIC_CLASSES)
        lbls = self.labels["f_prag"][self.indices]
        # bincount
        counts = torch.bincount(lbls, minlength=NUM_FORMALITY_PRAGMATIC_CLASSES).float()
        total = counts.sum()
        w = total / (counts + 1e-5)
        return w / w.sum() * NUM_FORMALITY_PRAGMATIC_CLASSES

    def get_gender_class_weights(self) -> torch.Tensor:
        if "g_prag" not in self.labels:
            return torch.ones(NUM_GENDER_PRAGMATIC_CLASSES)
        lbls = self.labels["g_prag"][self.indices]
        counts = torch.bincount(lbls, minlength=NUM_GENDER_PRAGMATIC_CLASSES).float()
        total = counts.sum()
        w = total / (counts + 1e-5)
        return w / w.sum() * NUM_GENDER_PRAGMATIC_CLASSES

    def get_grammaticality_class_weights(self) -> torch.Tensor:
        if "gram" not in self.labels:
            return torch.ones(NUM_GRAMMATICALITY_CLASSES)
        lbls = self.labels["gram"][self.indices]
        counts = torch.bincount(lbls, minlength=NUM_GRAMMATICALITY_CLASSES).float()
        total = counts.sum()
        w = total / (counts + 1e-5)
        return w / w.sum() * NUM_GRAMMATICALITY_CLASSES


def _collate_features(
    batch: List[Sample], batch_size: int, max_seq_len: int
) -> Dict[str, torch.Tensor]:
    feature_tensors: Dict[str, torch.Tensor] = {}
    if not batch:
        return feature_tensors

    # Hardcoded pad_id=0
    pad_id = 0

    for f in FEATURE_FIELDS:
        feature_tensors[f"input_ids_{f}"] = torch.full(
            (batch_size, max_seq_len), pad_id, dtype=torch.long
        )

    for i, s in enumerate(batch):
        for f in FEATURE_FIELDS:
            seq = s.feature_ids.get(f)
            if seq is None:
                continue

            if len(seq) > max_seq_len:
                seq = seq[:max_seq_len]

            if isinstance(seq, list):
                seq_t = torch.tensor(seq, dtype=torch.long)
            else:
                seq_t = seq

            feature_tensors[f"input_ids_{f}"][i, : len(seq_t)] = seq_t

    return feature_tensors


def _collate_register_labels(batch: List[Sample], batch_size: int) -> torch.Tensor:
    """Collate multi-hot register labels."""
    reg_targets = torch.zeros((batch_size, NUM_REGISTER_CLASSES), dtype=torch.float)
    for i, s in enumerate(batch):
        for rid in s.register_labels:
            if 0 <= rid < NUM_REGISTER_CLASSES:
                reg_targets[i, rid] = 1.0
    return reg_targets


def collate_fn(
    batch: List[Sample],
    max_seq_len: Optional[int] = None,
) -> TrainingBatch:
    """Collate samples into padded batches."""
    # pad_id is always 0 for current tokenizers

    # Helper to count feature length robustly
    def _get_len(s: Sample) -> int:
        for val in s.feature_ids.values():
            return len(val)
        return 0

    batch_size = len(batch)
    if max_seq_len is None and batch:
        max_seq_len = max(_get_len(s) for s in batch)
    elif max_seq_len is None:
        max_seq_len = 0

    attention_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.float)

    # Calculate attention mask
    if batch:
        for i, s in enumerate(batch):
            slen = _get_len(s)
            use_len = min(slen, max_seq_len)
            attention_mask[i, :use_len] = 1.0

    # Collect features
    feature_tensors = _collate_features(batch, batch_size, max_seq_len)
    reg_labels = _collate_register_labels(batch, batch_size)

    return TrainingBatch(
        feature_inputs=feature_tensors,
        attention_mask=attention_mask,
        formality_value=torch.tensor(
            [s.formality_value for s in batch], dtype=torch.float
        ),
        formality_pragmatic=torch.tensor(
            [s.formality_pragmatic for s in batch], dtype=torch.long
        ),
        gender_value=torch.tensor([s.gender_value for s in batch], dtype=torch.float),
        gender_pragmatic=torch.tensor(
            [s.gender_pragmatic for s in batch], dtype=torch.long
        ),
        grammaticality_labels=torch.tensor(
            [s.grammaticality_label for s in batch], dtype=torch.long
        ),
        register_labels=reg_labels,
        original_sentence=[s.original_sentence for s in batch],
        kotogram=[s.kotogram for s in batch],
        indices=torch.tensor([s.idx for s in batch], dtype=torch.long),
        kc_targets=[s.kc_targets for s in batch],
    )


# pylint: disable=too-many-locals
def _get_kc_pos_indices(
    kc_targets: List[Dict[Any, List[int]]],
    field: Any,
    vocab_size: int,
    device: torch.device,
    special_ids: Set[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Helper to compute sparse position indices and mask."""
    max_pos = 64

    # OPTIMIZED: Pre-collect all valid IDs as Python lists, then create tensors in bulk
    # This avoids creating torch.tensor() per sample in the loop
    all_pos_data: List[List[int]] = []
    all_mask_data: List[List[bool]] = []

    for target_dict in kc_targets:
        val_list = target_dict.get(field, [])

        # Filter special IDs and out-of-vocab
        valid_ids = [v for v in val_list if v < vocab_size and v not in special_ids]

        # Truncate to max_pos
        if len(valid_ids) > max_pos:
            valid_ids = valid_ids[:max_pos]

        # Pad to max_pos
        n_valid = len(valid_ids)
        padded_ids = valid_ids + [0] * (max_pos - n_valid)
        mask = [True] * n_valid + [False] * (max_pos - n_valid)

        all_pos_data.append(padded_ids)
        all_mask_data.append(mask)

    # Create tensors in bulk (single torch.tensor call per batch)
    pos_inds = torch.tensor(all_pos_data, dtype=torch.long, device=device)
    pos_mask = torch.tensor(all_mask_data, dtype=torch.bool, device=device)

    return pos_inds, pos_mask


def create_kc_batch(
    batch: TrainingBatch, tokenizer: Tokenizer, target_specs: Dict[KcFamilyId, int]
) -> Dict[str, torch.Tensor]:
    # pylint: disable=too-many-locals
    """
    Create KC targets (multi-hot) from a batch of input IDs.

    Args:
        batch: TrainingBatch (output of collate_fn)
        tokenizer: Tokenizer instance (to identify special tokens)
        target_specs: Dict mapping target name (field) to vocab size

    Returns:
        Dict mapping 'kc_targets_{field}' to (batch_size, vocab_size) float tensor
    """
    result: Dict[str, torch.Tensor] = {}

    # Get special tokens to ignore
    special_ids = {0, tokenizer.unk_id, tokenizer.cls_id}

    # Helper for device
    device = batch.attention_mask.device

    # Note: We rely on batch.kc_targets being populated by collate_fn from Sample objects
    if not batch.kc_targets:
        # Fallback if empty (shouldn't happen with valid collation)
        return result

    # Global effective mask initialization
    batch_size = len(batch.kc_targets)
    global_has_pos = torch.zeros(batch_size, dtype=torch.bool, device=device)

    # Strict iteration over target_specs which MUST be Dict[KcFamilyId, int]
    for target_family, vocab_size in target_specs.items():
        # Strict data alignment: The batch MUST contain data for the requested target.
        if target_family not in batch.kc_targets[0]:
            raise KeyError(
                f"Data for confirmed KcFamilyId '{target_family}' missing from batch targets."
            )

        kc_key = target_family

        # Import here to avoid circular import
        from train.kc import (
            KcDbMultilabelFamily,
            KcPnuFamily,
            get_family,
            is_family_db_sourced,
            is_family_sparse,
        )

        # Special handling for DB-sourced families (e.g., GRAMMAR_POINT, GENDER, FORMALITY)
        # These use different data structures: KcPnuFamily uses pos/neg arrays, KcMseFamily uses scalars
        if is_family_db_sourced(target_family):
            family_def = get_family(target_family)
            name = target_family.name.lower()

            if isinstance(family_def, KcPnuFamily):
                # PNU families (e.g., GRAMMAR_POINT) have pos/neg arrays
                max_pos = 64

                # Extract pos and neg IDs separately
                all_pos_data: List[List[int]] = []
                all_pos_mask_data: List[List[bool]] = []
                all_neg_data: List[List[int]] = []
                all_neg_mask_data: List[List[bool]] = []

                for target_dict in batch.kc_targets:
                    gp_data = target_dict.get(kc_key, {"pos": [], "neg": []})
                    pos_ids = gp_data.get("pos", [])
                    neg_ids = gp_data.get("neg", [])

                    # Filter and pad positive IDs
                    valid_pos = [v for v in pos_ids if v < vocab_size][:max_pos]
                    n_pos = len(valid_pos)
                    all_pos_data.append(valid_pos + [0] * (max_pos - n_pos))
                    all_pos_mask_data.append(
                        [True] * n_pos + [False] * (max_pos - n_pos)
                    )

                    # Filter and pad negative IDs
                    valid_neg = [v for v in neg_ids if v < vocab_size][:max_pos]
                    n_neg = len(valid_neg)
                    all_neg_data.append(valid_neg + [0] * (max_pos - n_neg))
                    all_neg_mask_data.append(
                        [True] * n_neg + [False] * (max_pos - n_neg)
                    )

                result[f"kc_gp_pos_inds_{name}"] = torch.tensor(
                    all_pos_data, dtype=torch.long, device=device
                )
                result[f"kc_gp_pos_mask_{name}"] = torch.tensor(
                    all_pos_mask_data, dtype=torch.bool, device=device
                )
                result[f"kc_gp_neg_inds_{name}"] = torch.tensor(
                    all_neg_data, dtype=torch.long, device=device
                )
                result[f"kc_gp_neg_mask_{name}"] = torch.tensor(
                    all_neg_mask_data, dtype=torch.bool, device=device
                )

                # Update global positive presence
                global_has_pos |= torch.tensor(
                    [
                        len(t.get(kc_key, {}).get("pos", [])) > 0
                        for t in batch.kc_targets
                    ],
                    dtype=torch.bool,
                    device=device,
                )
            else:
                # Continuous families (gender/formality) get scalar values from batch
                if name == "gender":
                    result[f"kc_continuous_{name}"] = batch.gender_value.to(device)
                elif name == "formality":
                    result[f"kc_continuous_{name}"] = batch.formality_value.to(device)
                # Classification versions: map continuous to class indices
                elif name == "gender_class":
                    # Map: -1.0 → 0, 0.0 → 1, 1.0 → 2
                    gender_class = ((batch.gender_value + 1.0) / 2.0 * 2).long()
                    gender_class = gender_class.clamp(0, 2)
                    result[f"kc_class_{name}"] = gender_class.to(device)
                elif name == "formality_class":
                    # DB values: -1.0=v_casual, -0.5=casual, 0.0=neutral, 0.5=formal, 1.0=v_formal
                    # Class indices: 0=v_casual, 1=casual, 2=neutral, 3=formal, 4=v_formal
                    formality_class = ((batch.formality_value + 1.0) / 0.5).long()
                    formality_class = formality_class.clamp(0, 4)
                    result[f"kc_class_{name}"] = formality_class.to(device)
                elif isinstance(family_def, KcDbMultilabelFamily):
                    # Multi-label families (register) use multi-hot targets from batch
                    # batch.register_labels is already [B, num_classes] multi-hot tensor
                    result[f"kc_multilabel_{name}"] = batch.register_labels.to(device)

            continue

        # Sparse Implementation (Indices + Mask)
        # We use sparse representation for all vocab sizes for consistency.
        pos_inds, pos_mask = _get_kc_pos_indices(
            batch.kc_targets, kc_key, vocab_size, device, special_ids
        )
        # Using f"kc_pos_inds_{target_family.name}" is safer/readable than ID.
        result[f"kc_pos_inds_{target_family.name.lower()}"] = pos_inds
        result[f"kc_pos_mask_{target_family.name.lower()}"] = pos_mask

        # Dense targets for non-sparse families
        if not is_family_sparse(target_family):
            # OPTIMIZED: Use scatter_ with pos_inds/pos_mask instead of nested Python loops
            dense_targets = torch.zeros(
                (batch_size, vocab_size), dtype=torch.float, device=device
            )
            # Use torch.where instead of clone() to avoid allocation
            # Where pos_mask is True, use pos_inds; where False, use 0
            valid_inds = torch.where(pos_mask, pos_inds, torch.zeros_like(pos_inds))

            # Scatter 1.0 to all valid positions
            dense_targets.scatter_(1, valid_inds, 1.0)

            # Clear any accidental assignments to index 0 (special token)
            dense_targets[:, 0] = 0.0

            result[f"kc_targets_{target_family.name.lower()}"] = dense_targets

        # Accumulate global positive presence
        global_has_pos |= pos_mask.any(dim=1)

    result["kc_has_pos_effective"] = global_has_pos

    return result
