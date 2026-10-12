"""Upload Indonesian legal navigation trajectory dataset to Hugging Face Hub."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from huggingface_hub import HfApi

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DEFAULT_DATASET_FILE = DATA_DIR / "d3_navigation_train.jsonl"
DEFAULT_README_FILE = DATA_DIR / "HF_DATASET_README.md"
DEFAULT_REPO_ID = "ersaasyhar/d3-indo-legal-navigation"


def upload_dataset(
    repo_id: str = DEFAULT_REPO_ID,
    dataset_file: Path = DEFAULT_DATASET_FILE,
    readme_file: Path = DEFAULT_README_FILE,
    private: bool = False,
) -> None:
    api = HfApi()
    user = api.whoami().get("name")
    logger.info("Authenticated as Hugging Face user: %s", user)

    logger.info("Creating dataset repo (if not existing): %s", repo_id)
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)

    sample_count = sum(1 for _ in open(dataset_file, "r", encoding="utf-8"))
    logger.info("Uploading %s -> train.jsonl (%d samples)", dataset_file, sample_count)
    api.upload_file(
        path_or_fileobj=str(dataset_file),
        path_in_repo="train.jsonl",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"Upload training trajectories ({sample_count} samples)",
    )

    if readme_file.exists():
        logger.info("Uploading %s -> README.md", readme_file)
        api.upload_file(
            path_or_fileobj=str(readme_file),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Upload dataset card README.md",
        )

    logger.info("Dataset successfully uploaded to: https://huggingface.co/datasets/%s", repo_id)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload navigation trajectory dataset to Hugging Face Hub."
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="HF dataset repo ID")
    parser.add_argument(
        "--dataset-file", type=Path, default=DEFAULT_DATASET_FILE, help="Path to jsonl file"
    )
    parser.add_argument(
        "--readme-file", type=Path, default=DEFAULT_README_FILE, help="Path to README card"
    )
    parser.add_argument("--private", action="store_true", help="Make dataset repo private")
    args = parser.parse_args()

    upload_dataset(
        repo_id=args.repo_id,
        dataset_file=args.dataset_file,
        readme_file=args.readme_file,
        private=args.private,
    )


if __name__ == "__main__":
    main()
