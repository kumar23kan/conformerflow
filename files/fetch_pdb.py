"""
ConformerFlow — Phase 1: PDB Data Fetcher
Robust version for Google Colab + RCSB compatibility
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
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ---------------------------------------------------
# Logging
# ---------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------
# URLs
# ---------------------------------------------------

RCSB_SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_DATA_URL = "https://data.rcsb.org/rest/v1/core/entry"
RCSB_DOWNLOAD = "https://files.rcsb.org/download"

# ---------------------------------------------------
# Robust Session
# ---------------------------------------------------

def create_session():

    retry_strategy = Retry(
        total=5,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"]
    )

    adapter = HTTPAdapter(
        max_retries=retry_strategy,
        pool_connections=10,
        pool_maxsize=10
    )

    session = requests.Session()

    session.mount("https://", adapter)
    session.mount("http://", adapter)

    session.headers.update({
        "User-Agent": "Mozilla/5.0"
    })

    return session


session = create_session()

# ---------------------------------------------------
# RCSB Queries
# ---------------------------------------------------

def query_nmr_entries(min_conformers=5, max_results=20000):

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
                        "attribute":
                        "pdbx_nmr_ensemble.conformers_submitted_total_number",
                        "operator": "greater_or_equal",
                        "value": min_conformers
                    }
                }
            ]
        },
        "return_type": "entry",
        "request_options": {
            "paginate": {
                "start": 0,
                "rows": max_results
            }
        }
    }

    return execute_search(query)


def query_xray_entries(max_results=5000):

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
            "paginate": {
                "start": 0,
                "rows": max_results
            }
        }
    }

    return execute_search(query)


def execute_search(query):

    try:

        resp = session.post(
            RCSB_SEARCH_URL,
            json=query,
            timeout=120
        )

        resp.raise_for_status()

        data = resp.json()

        ids = [x["identifier"] for x in data.get("result_set", [])]

        logger.info(f"Search returned {len(ids)} entries")

        return ids

    except Exception as e:

        logger.error(f"Search failed: {e}")

        return []


# ---------------------------------------------------
# Metadata
# ---------------------------------------------------

def fetch_entry_metadata(pdb_id):

    url = f"{RCSB_DATA_URL}/{pdb_id.upper()}"

    try:

        resp = session.get(url, timeout=30)

        if resp.status_code == 200:
            return resp.json()

    except Exception as e:

        logger.warning(f"Metadata failed for {pdb_id}: {e}")

    return None


# ---------------------------------------------------
# UniProt Mapping
# ---------------------------------------------------

def get_uniprot_accessions(pdb_id):

    metadata = fetch_entry_metadata(pdb_id)

    if metadata is None:
        return []

    accessions = set()

    try:

        polymer_entities = metadata.get(
            "rcsb_entry_container_identifiers",
            {}
        ).get("polymer_entity_ids", [])

        for entity_id in polymer_entities:

            url = (
                f"https://data.rcsb.org/rest/v1/core/"
                f"polymer_entity/{pdb_id.upper()}/{entity_id}"
            )

            resp = session.get(url, timeout=20)

            if resp.status_code != 200:
                continue

            data = resp.json()

            refs = data.get(
                "rcsb_polymer_entity_container_identifiers",
                {}
            )

            uniprot_ids = refs.get("uniprot_ids", [])

            for up in uniprot_ids:
                accessions.add(up)

    except Exception as e:

        logger.warning(f"UniProt mapping failed for {pdb_id}: {e}")

    return list(accessions)


# ---------------------------------------------------
# Downloading
# ---------------------------------------------------

def download_structure(pdb_id, out_dir, preferred_format="pdb"):

    pdb_id = pdb_id.upper()

    formats = ["pdb", "cif"]

    if preferred_format == "cif":
        formats = ["cif", "pdb"]

    for fmt in formats:

        ext = "pdb" if fmt == "pdb" else "cif"

        out_path = out_dir / f"{pdb_id}.{ext}"

        if out_path.exists():
            return out_path

        url = f"{RCSB_DOWNLOAD}/{pdb_id}.{ext}"

        try:

            resp = session.get(
                url,
                timeout=120,
                stream=True
            )

            if resp.status_code == 200:

                with open(out_path, "wb") as f:

                    for chunk in resp.iter_content(chunk_size=8192):

                        if chunk:
                            f.write(chunk)

                logger.info(f"Downloaded {pdb_id}.{ext}")

                return out_path

            else:

                logger.warning(
                    f"{pdb_id}.{ext} unavailable "
                    f"(status {resp.status_code})"
                )

        except Exception as e:

            logger.warning(f"Failed {pdb_id}.{ext}: {e}")

    return None


def batch_download(
    pdb_ids,
    out_dir,
    preferred_format="pdb",
    max_workers=2,
    delay=0.2
):

    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    def worker(pid):

        time.sleep(delay)

        return pid, download_structure(
            pid,
            out_dir,
            preferred_format
        )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:

        futures = {
            executor.submit(worker, pid): pid
            for pid in pdb_ids
        }

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Downloading"
        ):

            try:

                pid, path = future.result()

                if path:
                    results[pid] = str(path)

            except Exception as e:

                logger.warning(f"Worker failed: {e}")

    logger.info(
        f"Downloaded {len(results)}/{len(pdb_ids)} structures"
    )

    return results


# ---------------------------------------------------
# Pair Detection
# ---------------------------------------------------

def find_paired_entries(
    nmr_ids,
    xray_ids,
    sample_size=500
):

    logger.info("Finding paired entries")

    nmr_sample = nmr_ids[:sample_size]
    xray_sample = xray_ids[:sample_size]

    nmr_map = {}

    logger.info("Mapping NMR structures")

    for pid in tqdm(nmr_sample):

        ups = get_uniprot_accessions(pid)

        if ups:
            nmr_map[pid] = ups

        time.sleep(0.05)

    xray_map = {}

    logger.info("Mapping X-ray structures")

    for pid in tqdm(xray_sample):

        ups = get_uniprot_accessions(pid)

        if ups:
            xray_map[pid] = ups

        time.sleep(0.05)

    pairs = []

    for nmr_id, nmr_ups in nmr_map.items():

        for xray_id, xray_ups in xray_map.items():

            overlap = set(nmr_ups).intersection(xray_ups)

            if overlap:

                for up in overlap:

                    pairs.append({
                        "nmr_id": nmr_id,
                        "xray_id": xray_id,
                        "uniprot_id": up
                    })

    logger.info(f"Found {len(pairs)} paired entries")

    return pairs


# ---------------------------------------------------
# Main Pipeline
# ---------------------------------------------------

def run_fetch_pipeline(
    output_dir,
    min_conformers=5,
    max_nmr=15000,
    max_xray=5000,
    download_files=True,
    max_workers=2
):

    base = Path(output_dir)

    base.mkdir(parents=True, exist_ok=True)

    (base / "nmr").mkdir(exist_ok=True)
    (base / "xray").mkdir(exist_ok=True)
    (base / "paired").mkdir(exist_ok=True)

    logger.info("Querying NMR entries")
    nmr_ids = query_nmr_entries(
        min_conformers=min_conformers,
        max_results=max_nmr
    )

    logger.info("Querying X-ray entries")
    xray_ids = query_xray_entries(
        max_results=max_xray
    )

    logger.info("Finding paired entries")
    pairs = find_paired_entries(
        nmr_ids,
        xray_ids
    )

    paired_nmr = {x["nmr_id"] for x in pairs}
    paired_xray = {x["xray_id"] for x in pairs}

    manifests = {
        "nmr_all": nmr_ids,
        "xray_all": xray_ids,
        "paired": pairs
    }

    for name, data in manifests.items():

        with open(base / f"{name}.json", "w") as f:
            json.dump(data, f, indent=2)

    if download_files:

        logger.info("Downloading NMR structures")

        batch_download(
            nmr_ids,
            base / "nmr",
            preferred_format="pdb",
            max_workers=max_workers
        )

        logger.info("Downloading paired structures")

        paired_ids = list(paired_nmr | paired_xray)

        batch_download(
            paired_ids,
            base / "paired",
            preferred_format="pdb",
            max_workers=max_workers
        )

    logger.info("Pipeline complete")


# ---------------------------------------------------
# CLI
# ---------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./pdb_data"
    )

    parser.add_argument(
        "--min_conformers",
        type=int,
        default=5
    )

    parser.add_argument(
        "--max_nmr",
        type=int,
        default=15000
    )

    parser.add_argument(
        "--max_xray",
        type=int,
        default=5000
    )

    parser.add_argument(
        "--no_download",
        action="store_true"
    )

    parser.add_argument(
        "--max_workers",
        type=int,
        default=2
    )

    args = parser.parse_args()

    run_fetch_pipeline(
        output_dir=args.output_dir,
        min_conformers=args.min_conformers,
        max_nmr=args.max_nmr,
        max_xray=args.max_xray,
        download_files=not args.no_download,
        max_workers=args.max_workers
    )
