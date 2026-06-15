from __future__ import annotations

import json
import hashlib
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Iterable, Tuple


@dataclass(frozen=True)
class EncodedStay:
    """Immutable representation of an encoded ICU stay.

    Attributes
    ----------
    patientunitstayid: str
        Local stay identifier (used only for provenance checks, never exposed to models).
    tokens: List[int]
        Token IDs for the stay, must include a leading CLS token and respect the max
        sequence length of 256.
    label: int
        Binary hospital‑mortality label (0 or 1).
    split_name: str
        One of "train", "validation", "test".
    """

    patientunitstayid: str
    tokens: List[int]
    label: int
    split_name: str

    def __post_init__(self) -> None:
        if not self.patientunitstayid or not str(self.patientunitstayid).strip():
            raise ValueError("patientunitstayid must be non‑empty")
        if self.split_name not in {"train", "validation", "test"}:
            raise ValueError("split_name must be 'train', 'validation', or 'test'")
        if self.label not in (0, 1):
            raise ValueError("label must be 0 or 1")
        if len(self.tokens) > 256:
            raise ValueError("tokens exceed max_seq_len of 256")
        if not self.tokens:
            raise ValueError("tokens list may not be empty")
        # CLS token ID is defined by the tokenizer contract as 3
        if self.tokens[0] != 3:
            raise ValueError("first token must be [CLS] (id 3)")


@dataclass
class ShardIndexEntry:
    shard_id: int
    checksum: str  # SHA‑256 hex digest of the shard file content


class EncodedDataset:
    """Restartable, bounded‑memory dataset loader.

    The on‑disk layout is

    ```
    <root>/
        shard_0.json
        shard_1.json
        ...
        index.json
    ```

    *Each shard* contains a JSON list of encoded stay records. The index records the
    SHA‑256 checksum for each shard, enabling the loader to detect incomplete or
    corrupted shards. The loader raises on duplicate stay identifiers or any
    contract violation.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise ValueError(f"encoded dataset root must be a directory: {self.root}")
        self._index: List[ShardIndexEntry] = self._read_index()
        self._stays: Dict[str, EncodedStay] = {}
        self._load_shards()

    # ---------------------------------------------------------------------
    # Index handling
    # ---------------------------------------------------------------------
    def _read_index(self) -> List[ShardIndexEntry]:
        index_path = self.root / "index.json"
        if not index_path.is_file():
            raise ValueError(f"index.json not found in {self.root}")
        raw = json.loads(index_path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError("index.json must contain a list of entries")
        entries: List[ShardIndexEntry] = []
        for entry in raw:
            if not isinstance(entry, dict) or "shard_id" not in entry or "checksum" not in entry:
                raise ValueError("malformed index entry")
            entries.append(ShardIndexEntry(shard_id=int(entry["shard_id"]), checksum=str(entry["checksum"])) )
        # Ensure deterministic order by shard_id
        entries.sort(key=lambda e: e.shard_id)
        return entries

    # ---------------------------------------------------------------------
    # Shard loading and validation
    # ---------------------------------------------------------------------
    def _load_shards(self) -> None:
        for entry in self._index:
            shard_path = self.root / f"shard_{entry.shard_id}.json"
            if not shard_path.is_file():
                raise ValueError(f"shard file missing: {shard_path}")
            content = shard_path.read_bytes()
            checksum = hashlib.sha256(content).hexdigest()
            if checksum != entry.checksum:
                raise ValueError(f"checksum mismatch for shard {entry.shard_id}")
            stays_raw = json.loads(content.decode("utf-8"))
            if not isinstance(stays_raw, list):
                raise ValueError("shard must contain a list of stay records")
            for rec in stays_raw:
                stay = self._parse_stay_record(rec)
                if stay.patientunitstayid in self._stay_ids:
                    raise ValueError(f"duplicate stay identifier {stay.patientunitstayid}")
                self._stay_ids.add(stay.patientunitstayid)
                self._len += 1

    def _parse_stay_record(self, rec: object) -> EncodedStay:
        if not isinstance(rec, dict):
            raise ValueError("stay record must be a dict")
        required = {"patientunitstayid", "tokens", "label", "split_name"}
        missing = required - rec.keys()
        if missing:
            raise ValueError(f"stay record missing fields: {missing}")
        return EncodedStay(
            patientunitstayid=str(rec["patientunitstayid"]),
            tokens=list(rec["tokens"]),
            label=int(rec["label"]),
            split_name=str(rec["split_name"]),
        )

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def stays(self) -> List[EncodedStay]:
        """Return a list of all encoded stays in deterministic order."""
        all_stays = []
        for entry in self._index:
            shard_path = self.root / f"shard_{entry.shard_id}.json"
            content = shard_path.read_bytes()
            stays_raw = json.loads(content.decode("utf-8"))
            for rec in stays_raw:
                all_stays.append(self._parse_stay_record(rec))
        all_stays.sort(key=lambda s: s.patientunitstayid)
        return all_stays

    def __len__(self) -> int:
        return self._len

    def __iter__(self):
        for entry in self._index:
            shard_path = self.root / f"shard_{entry.shard_id}.json"
            content = shard_path.read_bytes()
            stays_raw = json.loads(content.decode("utf-8"))
            for rec in stays_raw:
                yield self._parse_stay_record(rec)

    # ---------------------------------------------------------------------
    # Helper for test writers – creates a shard and updates the index.
    # ---------------------------------------------------------------------
    @staticmethod
    def write_shard(stays: Iterable[Dict[str, object]], shard_id: int, root: str | Path) -> None:
        """Write a single shard file and update the index.

        Parameters
        ----------
        stays: iterable of mapping objects with the same schema required by
            :class:`EncodedStay`.
        shard_id: integer identifier for the shard (0‑based).
        root: directory containing the dataset.
        """
        root_path = Path(root)
        root_path.mkdir(parents=True, exist_ok=True)
        shard_path = root_path / f"shard_{shard_id}.json"
        # Serialize stays as a JSON list
        data = list(stays)
        shard_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        # Compute checksum
        checksum = hashlib.sha256(shard_path.read_bytes()).hexdigest()
        # Update index.json atomically
        index_path = root_path / "index.json"
        if index_path.is_file():
            index_raw = json.loads(index_path.read_text(encoding="utf-8"))
        else:
            index_raw = []
        # Remove any existing entry for this shard_id
        index_raw = [e for e in index_raw if e.get("shard_id") != shard_id]
        index_raw.append({"shard_id": shard_id, "checksum": checksum})
        tmp = root_path / "index.json.tmp"
        tmp.write_text(json.dumps(index_raw, indent=2), encoding="utf-8")
        tmp.replace(index_path)
