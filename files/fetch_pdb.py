"""
ConformerFlow — Phase 1: PDB Data Fetcher
Fetches NMR ensemble entries and paired X-ray structures from RCSB PDB.
"""

import os
import time
import json
import logging
import requests
import argparse
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RCSB_SEARCH_URL  = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_DATA_URL    = "https://data.rcsb.org/rest/v1/core/entry"
RCSB_DOWNLOAD    = "https://files.rcsb.org/download"


# ──────────────────────────────────────────────
# RCSB Search Queries
# ──────────────────────────────────────────────

def query_nmr_entries(min_conformers: int = 5, max_results: int = 20000) -> list:
    """
    Query RCSB for NMR structures with at least min_conformers models.
    Filters: SOLUTION NMR, protein, minimum conformer count.
    """
    query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "exptl.method",
                        "operator": "exact_match",
                        "value": "SOLUTION NMR"
                    }
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "entity_poly.rcsb_entity_polymer_type",
                        "operator": "exact_match",
                        "value": "Protein"
                    }
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "pdbx_nmr_ensemble.conformers_submitted_total_number",
                        "operator": "greater_or_equal",
                        "value": min_conformers
                    }
                }
            ]
        },
        "return_type": "entry",
        "request_options": {
            "paginate": {"start": 0, "rows": max_results},
            "results_content_type": ["experimental"]
        }
    }
    return _execute_search(query)


def query_xray_entries(max_results: int = 5000) -> list:
    """
    Query RCSB for high-quality X-ray crystal structures of proteins.
    Filters: X-RAY DIFFRACTION, protein, resolution <= 2.5 Angstrom.
    """
    query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "exptl.method",
                        "operator": "exact_match",
                        "value": "X-RAY DIFFRACTION"
                    }
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "entity_poly.rcsb_entity_polymer_type",
                        "operator": "exact_match",
                        "value": "Protein"
                    }
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "refine.ls_d_res_high",
                        "operator": "less_or_equal",
                        "value": 2.5
                    }
                }
            ]
        },
        "return_type": "entry",
        "request_options": {
            "paginate": {"start": 0, "rows": max_results},
            "results_content_type": ["experimental"]
        }
    }
    return _execute_search(query)


def _execute_search(query: dict) -> list:
    """Execute a RCSB search query and return list of PDB IDs."""
    try:
        resp = requests.post(RCSB_SEARCH_URL, json=query, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        ids = [hit["identifier"] for hit in data.get("result_set", [])]
        logger.info(f"Search returned {len(ids)} entries.")
        return ids
    except Exception as e:
        logger.error(f"Search query failed: {e}")
        return []


# ──────────────────────────────────────────────
# Metadata Fetching
# ──────────────────────────────────────────────

def fetch_entry_metadata(pdb_id: str) -> dict:
    """Fetch metadata for a single PDB entry."""
    url = f"{RCSB_DATA_URL}/{pdb_id.upper()}"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


# ──────────────────────────────────────────────
# PDB File Downloading
# ──────────────────────────────────────────────

def download_pdb(pdb_id: str, out_dir: Path, file_format: str = "pdb"):
    """
    Download a single PDB file.
    file_format: 'pdb' for legacy format, 'cif' for mmCIF.
    Returns path to downloaded file, or None on failure.
    """
    pdb_id = pdb_id.upper()
    ext = "pdb" if file_format == "pdb" else "cif"
    out_path = out_dir / f"{pdb_id}.{ext}"

    if out_path.exists():
        return out_path  # already downloaded

    url = f"{RCSB_DOWNLOAD}/{pdb_id}.{ext}"
    try:
        resp = requests.get(url, timeout=60, stream=True)
        resp.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        return out_path
    except Exception as e:
        logger.warning(f"Failed to download {pdb_id}: {e}")
        return None


def batch_download(pdb_ids: list,
                   out_dir: Path,
                   file_format: str = "pdb",
                   max_workers: int = 8,
                   delay: float = 0.1) -> dict:
    """
    Download multiple PDB files in parallel with rate limiting.
    Returns dict mapping pdb_id -> local path.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    def _download_one(pdb_id):
        time.sleep(delay)
        return pdb_id, download_pdb(pdb_id, out_dir, file_format)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_download_one, pid): pid for pid in pdb_ids}
        for future in tqdm(as_completed(futures), total=len(pdb_ids), desc="Downloading"):
            pdb_id, path = future.result()
            if path is not None:
                results[pdb_id] = str(path)

    logger.info(f"Downloaded {len(results)}/{len(pdb_ids)} files to {out_dir}")
    return results


# ──────────────────────────────────────────────
# Paired X-ray / NMR Detection
# ──────────────────────────────────────────────

def find_paired_entries(nmr_ids: list,
                        xray_ids: list,
                        sample_size: int = 500) -> list:
    """
    Find proteins that have BOTH an NMR and X-ray structure deposited.
    Uses UniProt mapping via RCSB REST API to identify pairs.
    These will be held out entirely from training.

    Returns list of dicts: {nmr_id, xray_id, uniprot_id}
    """
    logger.info("Finding paired NMR/X-ray entries via UniProt mapping...")

    def get_uniprot(pdb_id):
        url = f"https://data.rcsb.org/rest/v1/core/uniprot/{pdb_id.upper()}/1"
        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                if data:
                    accessions = data[0].get("rcsb_uniprot_accession", [])
                    return accessions[0] if accessions else None
        except Exception:
            return None

    # Sample for tractability during initial pipeline run
    nmr_sample  = nmr_ids[:sample_size]
    xray_sample = xray_ids[:sample_size]

    logger.info(f"Mapping {len(nmr_sample)} NMR entries to UniProt...")
    nmr_uniprot = {}
    for pid in tqdm(nmr_sample, desc="NMR→UniProt"):
        up = get_uniprot(pid)
        if up:
            nmr_uniprot[pid] = up
        time.sleep(0.05)

    logger.info(f"Mapping {len(xray_sample)} X-ray entries to UniProt...")
    xray_uniprot = {}
    for pid in tqdm(xray_sample, desc="Xray→UniProt"):
        up = get_uniprot(pid)
        if up:
            xray_uniprot[pid] = up
        time.sleep(0.05)

    # Invert: uniprot -> list of xray ids
    uniprot_to_xray = {}
    for pid, up in xray_uniprot.items():
        uniprot_to_xray.setdefault(up, []).append(pid)

    # Find overlaps
    pairs = []
    for nmr_id, up in nmr_uniprot.items():
        if up in uniprot_to_xray:
            for xray_id in uniprot_to_xray[up]:
                pairs.append({
                    "nmr_id":     nmr_id,
                    "xray_id":    xray_id,
                    "uniprot_id": up
                })

    logger.info(f"Found {len(pairs)} paired NMR/X-ray entries (held-out validation set).")
    return pairs


# ──────────────────────────────────────────────
# Main Pipeline
# ──────────────────────────────────────────────

def run_fetch_pipeline(output_dir: str,
                       min_conformers: int = 5,
                       max_nmr: int = 15000,
                       max_xray: int = 5000,
                       download_files: bool = True,
                       max_workers: int = 8):
    """
    Full data fetching pipeline:
      1. Query NMR entries from RCSB
      2. Query X-ray entries from RCSB
      3. Find paired entries (held-out validation set)
      4. Download PDB files
      5. Save all manifests as JSON
    """
    base = Path(output_dir)
    base.mkdir(parents=True, exist_ok=True)
    (base / "nmr").mkdir(exist_ok=True)
    (base / "xray").mkdir(exist_ok=True)
    (base / "paired").mkdir(exist_ok=True)

    # Step 1 & 2: Query
    logger.info("=== Step 1: Querying NMR entries ===")
    nmr_ids = query_nmr_entries(min_conformers=min_conformers, max_results=max_nmr)

    logger.info("=== Step 2: Querying X-ray entries ===")
    xray_ids = query_xray_entries(max_results=max_xray)

    # Step 3: Find paired entries
    logger.info("=== Step 3: Finding paired NMR/X-ray entries ===")
    pairs = find_paired_entries(nmr_ids, xray_ids)
    paired_nmr_ids  = {p["nmr_id"]  for p in pairs}
    paired_xray_ids = {p["xray_id"] for p in pairs}

    # Remove paired entries from training pool
    nmr_train_ids  = [i for i in nmr_ids  if i not in paired_nmr_ids]
    xray_train_ids = [i for i in xray_ids if i not in paired_xray_ids]

    logger.info(f"NMR   — total: {len(nmr_ids):,} | train: {len(nmr_train_ids):,} | held-out: {len(paired_nmr_ids):,}")
    logger.info(f"X-ray — total: {len(xray_ids):,} | held-out: {len(paired_xray_ids):,}")

    # Step 4: Save manifests
    manifests = {
        "nmr_all":         nmr_ids,
        "nmr_train":       nmr_train_ids,
        "xray_all":        xray_ids,
        "xray_train":      xray_train_ids,
        "paired":          pairs,
        "paired_nmr_ids":  list(paired_nmr_ids),
        "paired_xray_ids": list(paired_xray_ids),
    }
    for name, data in manifests.items():
        path = base / f"{name}.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved {name}.json ({len(data):,} entries)")

    # Step 5: Download PDB files
    if download_files:
        logger.info("=== Step 5a: Downloading NMR training PDB files ===")
        batch_download(nmr_train_ids, base / "nmr",
                       file_format="pdb", max_workers=max_workers)

        logger.info("=== Step 5b: Downloading paired held-out PDB files ===")
        all_paired = list(paired_nmr_ids) + list(paired_xray_ids)
        batch_download(all_paired, base / "paired",
                       file_format="pdb", max_workers=max_workers)

    logger.info("=== Fetch pipeline complete ===")
    return manifests


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ConformerFlow — PDB Data Fetcher")
    parser.add_argument("--output_dir",     type=str, default="./pdb_data",
                        help="Directory to save downloaded PDB files")
    parser.add_argument("--min_conformers", type=int, default=5,
                        help="Minimum NMR conformers required per entry")
    parser.add_argument("--max_nmr",        type=int, default=15000,
                        help="Maximum NMR entries to fetch")
    parser.add_argument("--max_xray",       type=int, default=5000,
                        help="Maximum X-ray entries to fetch")
    parser.add_argument("--no_download",    action="store_true",
                        help="Only fetch IDs, skip PDB file downloads")
    parser.add_argument("--max_workers",    type=int, default=8,
                        help="Parallel download threads")
    args = parser.parse_args()

    run_fetch_pipeline(
        output_dir     = args.output_dir,
        min_conformers = args.min_conformers,
        max_nmr        = args.max_nmr,
        max_xray       = args.max_xray,
        download_files = not args.no_download,
        max_workers    = args.max_workers,
    )
