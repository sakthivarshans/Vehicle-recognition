"""
database/db_manager.py
----------------------
VehicleDatabase: FAISS-backed reference database for vehicle recognition.

Stores L2-normalised embedding vectors indexed by vehicle identity
(make + model + year).  Similarity search uses inner product, which equals
cosine similarity on L2-normalised vectors.

New vehicles are added by calling enrol() — no model retraining needed.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class VehicleDatabase:
    """
    FAISS reference database for vehicle embeddings.

    Parameters
    ----------
    index_path    : path to save/load the FAISS index (.index file)
    metadata_path : path to save/load vehicle metadata (.json file)
    embedding_dim : dimensionality of the embedding vectors
    """

    def __init__(
        self,
        index_path: str = "database/vehicle_db.index",
        metadata_path: str = "database/metadata.json",
        embedding_dim: int = 256,
    ) -> None:
        try:
            import faiss
            self._faiss = faiss
        except ImportError:
            raise ImportError(
                "faiss-cpu is required.\n"
                "Install with: pip install faiss-cpu"
            )

        self.index_path    = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.embedding_dim = embedding_dim
        self.metadata: List[Dict] = []

        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)

        if self.index_path.exists() and self.metadata_path.exists():
            self.load()
        else:
            # Inner product index (= cosine similarity for L2-normalised vectors)
            self._index = self._faiss.IndexFlatIP(embedding_dim)
            logger.info(
                "New database created (embedding_dim=%d)", embedding_dim
            )

    # ── Enrolment ─────────────────────────────────────────────────────────────
    def enrol(
        self,
        make: str,
        model: str,
        year: str,
        embeddings: np.ndarray,
    ) -> None:
        """
        Add a new vehicle to the database.

        Parameters
        ----------
        make       : vehicle manufacturer (e.g. 'Toyota')
        model      : vehicle model name (e.g. 'Supra')
        year       : year or range (e.g. '2020' or '2018-2020')
        embeddings : shape (N, embedding_dim) — one row per reference image.
                     Will be mean-pooled to a single representative vector.
        """
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Embedding dim mismatch: expected {self.embedding_dim}, "
                f"got {embeddings.shape[1]}"
            )

        # Mean-pool multiple reference embeddings, then re-normalise
        mean_emb = embeddings.mean(axis=0)
        norm = np.linalg.norm(mean_emb)
        if norm < 1e-8:
            raise ValueError("Embedding has zero norm — check your input images.")
        mean_emb = (mean_emb / norm).astype(np.float32).reshape(1, -1)

        self._index.add(mean_emb)
        self.metadata.append({
            "make":  make,
            "model": model,
            "year":  year,
            "id":    len(self.metadata),
        })
        self.save()
        logger.info(
            "Enrolled: %s %s (%s) | DB size: %d", make, model, year, len(self)
        )

    # ── Query ─────────────────────────────────────────────────────────────────
    def query(
        self,
        embedding: np.ndarray,
        top_k: int = 5,
        threshold: float = 0.65,
    ) -> List[Dict]:
        """
        Find the most similar vehicles in the database.

        Parameters
        ----------
        embedding : shape (1, embedding_dim) or (embedding_dim,), L2-normalised
        top_k     : number of nearest neighbours to return
        threshold : minimum similarity score to accept a match

        Returns
        -------
        List of dicts with keys: make, model, year, score, rank.
        Returns [{"make": "unknown", ...}] when all scores are below threshold.
        """
        if len(self) == 0:
            logger.warning("Database is empty — returning unknown.")
            return [{"make": "unknown", "model": "unknown", "year": "", "score": 0.0, "rank": 1}]

        emb = np.array(embedding, dtype=np.float32)
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)

        # Re-normalise query just in case
        norm = np.linalg.norm(emb)
        if norm > 1e-8:
            emb = emb / norm

        k = min(top_k, len(self))
        scores, indices = self._index.search(emb, k)

        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
            if idx < 0 or score < threshold:
                continue
            entry = self.metadata[idx].copy()
            entry["score"] = float(score)
            entry["rank"]  = rank
            results.append(entry)

        if not results:
            return [{"make": "unknown", "model": "unknown", "year": "", "score": 0.0, "rank": 1}]

        return results

    # ── Persistence ───────────────────────────────────────────────────────────
    def save(self) -> None:
        """Persist the index and metadata to disk."""
        self._faiss.write_index(self._index, str(self.index_path))
        with open(self.metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=2)
        logger.debug("Database saved (%d entries).", len(self))

    def load(self) -> None:
        """Load the index and metadata from disk."""
        self._index = self._faiss.read_index(str(self.index_path))
        with open(self.metadata_path) as f:
            self.metadata = json.load(f)
        logger.info(
            "Database loaded from disk: %d entries, embedding_dim=%d",
            len(self), self.embedding_dim,
        )

    # ── Management ────────────────────────────────────────────────────────────
    def remove(self, make: str, model: str) -> bool:
        """
        Remove a vehicle entry and rebuild the index.
        Returns True if removed, False if not found.
        """
        target = [(i, m) for i, m in enumerate(self.metadata)
                  if m["make"].lower() == make.lower()
                  and m["model"].lower() == model.lower()]

        if not target:
            logger.warning("Vehicle not found in database: %s %s", make, model)
            return False

        # Rebuild index without the removed entries
        remove_ids = {i for i, _ in target}
        new_meta   = [m for i, m in enumerate(self.metadata) if i not in remove_ids]

        # Reconstruct vectors for kept entries
        if new_meta:
            # Reconstruct all stored vectors by re-searching with identity
            all_vecs = self._reconstruct_all()
            kept_vecs = np.stack(
                [all_vecs[i] for i in range(len(self.metadata)) if i not in remove_ids]
            )
            self._index = self._faiss.IndexFlatIP(self.embedding_dim)
            self._index.add(kept_vecs.astype(np.float32))
        else:
            self._index = self._faiss.IndexFlatIP(self.embedding_dim)

        # Re-assign IDs
        for new_id, entry in enumerate(new_meta):
            entry["id"] = new_id
        self.metadata = new_meta
        self.save()

        logger.info("Removed: %s %s | DB size now: %d", make, model, len(self))
        return True

    def _reconstruct_all(self) -> np.ndarray:
        """Reconstruct all stored vectors from the flat index."""
        n = self._index.ntotal
        vecs = np.zeros((n, self.embedding_dim), dtype=np.float32)
        self._faiss.extract_index_ivf  # noqa — just checking faiss version
        # IndexFlatIP stores vectors directly; reconstruct via search trick
        for i in range(n):
            self._index.reconstruct(i, vecs[i])
        return vecs

    def list_vehicles(self) -> None:
        """Print a formatted table of all enrolled vehicles."""
        if not self.metadata:
            print("Database is empty.")
            return
        print(f"\n{'ID':>4}  {'Make':<15}  {'Model':<25}  {'Year':<10}")
        print("─" * 58)
        for entry in self.metadata:
            print(
                f"{entry['id']:>4}  {entry['make']:<15}  "
                f"{entry['model']:<25}  {entry['year']:<10}"
            )
        print(f"\nTotal: {len(self)} vehicles\n")

    def __len__(self) -> int:
        return len(self.metadata)

    def __repr__(self) -> str:
        return (
            f"VehicleDatabase(entries={len(self)}, "
            f"embedding_dim={self.embedding_dim}, "
            f"index={self.index_path})"
        )


# ── Standalone smoke test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    import tempfile
    logging.basicConfig(level=logging.INFO)
    print("=== VehicleDatabase smoke test ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        db = VehicleDatabase(
            index_path=f"{tmpdir}/test.index",
            metadata_path=f"{tmpdir}/test.json",
            embedding_dim=256,
        )
        print(f"Empty DB: {db}")

        # Enrol two vehicles
        emb_a = np.random.randn(3, 256).astype(np.float32)
        db.enrol("Toyota", "Supra", "2020", emb_a)

        emb_b = np.random.randn(2, 256).astype(np.float32)
        db.enrol("BMW", "M3", "2019", emb_b)

        print(f"After enrolment: {db}")
        db.list_vehicles()

        # Query with Toyota embedding (should match Toyota Supra)
        query_emb = emb_a[0].reshape(1, -1)
        query_emb = query_emb / np.linalg.norm(query_emb)
        results = db.query(query_emb, top_k=2, threshold=0.0)
        print("Query results:")
        for r in results:
            print(f"  {r['make']} {r['model']} — score={r['score']:.4f}")

        # Remove one
        db.remove("BMW", "M3")
        print(f"After remove: {len(db)} vehicles")

    print("\nVehicleDatabase smoke test PASSED")
