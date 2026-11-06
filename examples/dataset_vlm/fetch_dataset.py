"""Fetch a small VQAv2 subset for the dataset-VLM example into a local cache.

What this is for:
    The real-model config needs a dataset: images, questions and ground-truth
    answers. This script downloads a small subset of VQAv2 at run time and writes
    it under ``datasets/``, which ``.gitignore`` already excludes, so nothing ever
    reaches the index. Nothing here is redistributed with the repository.

What it fetches and its license:
    - Questions and annotations from VQA v2.0 (the VQA Consortium), licensed
      under Creative Commons Attribution 4.0 International
      (https://visualqa.org/terms.html). The two validation zips are small and
      fully downloaded, then read in memory.
    - The images those questions reference, from the COCO val2014 split, fetched
      ONE image at a time from COCO's own bucket
      (http://images.cocodataset.org/val2014/<name>). COCO's terms
      (https://cocodataset.org/#termsofuse) license the annotations CC BY 4.0 and
      state the consortium does not own the images, whose use must follow the
      Flickr Terms of Use. This example uses images only, caches them locally and
      redistributes none of them - the same approach as ``examples/distributed_sensing``.

What it contains:
    - :func:`fetch`: write ``limit`` samples (images + a manifest) into a folder.
    - a command line: ``--task`` (only ``vqa``), ``--limit``, ``--dest``.

How it fits into CADET:
    Run it before ``examples/dataset_vlm/real.yaml``, whose ``source.path`` points
    at the default destination. The stub configs need nothing from here.
"""

import argparse
import io
import json
import os
import sys
import urllib.request
import zipfile

QUESTIONS_URL = "https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip"
ANNOTATIONS_URL = "https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip"
IMAGE_URL = "http://images.cocodataset.org/val2014/{name}"
IMAGE_NAME = "COCO_val2014_{image_id:012d}.jpg"
DEFAULT_DEST = os.path.join("datasets", "dataset_vlm", "vqa")
DEFAULT_LIMIT = 50


def _download_json_from_zip(url, timeout_s=180.0):
    """Download a zip into memory and return the single JSON it contains."""
    with urllib.request.urlopen(url, timeout=timeout_s) as response:
        blob = response.read()
    archive = zipfile.ZipFile(io.BytesIO(blob))
    names = [n for n in archive.namelist() if n.lower().endswith(".json")]
    if not names:
        raise RuntimeError("no JSON entry in {}".format(url))
    with archive.open(names[0]) as handle:
        return json.load(handle)


def _fetch_image(image_id, dest, timeout_s=60.0):
    """Fetch one COCO val2014 image into ``dest``; return its file name or None."""
    name = IMAGE_NAME.format(image_id=image_id)
    out_path = os.path.join(dest, name)
    if os.path.exists(out_path):
        return name
    try:
        with urllib.request.urlopen(IMAGE_URL.format(name=name), timeout=timeout_s) as response:
            data = response.read()
    except Exception as exc:
        print("skip image {}: {}".format(image_id, exc), file=sys.stderr)
        return None
    with open(out_path, "wb") as handle:
        handle.write(data)
    return name


def fetch(dest, limit):
    """Write ``limit`` VQAv2 samples (images + manifest) into ``dest``."""
    if not os.path.isdir(dest):
        os.makedirs(dest)
    print("downloading VQAv2 val questions and annotations ...")
    questions = _download_json_from_zip(QUESTIONS_URL)["questions"]
    annotations = _download_json_from_zip(ANNOTATIONS_URL)["annotations"]
    answers_by_qid = {
        a["question_id"]: {
            "multiple_choice_answer": a.get("multiple_choice_answer"),
            "answers": [item["answer"] for item in a.get("answers", [])],
        }
        for a in annotations
    }
    samples = []
    for question in questions:
        if len(samples) >= limit:
            break
        qid = question["question_id"]
        gt = answers_by_qid.get(qid)
        if gt is None:
            continue
        name = _fetch_image(question["image_id"], dest)
        if name is None:
            continue
        samples.append(
            {
                "sample_id": qid,
                "question_id": qid,
                "image": name,
                "question": question["question"],
                "answers": gt["answers"],
                "multiple_choice_answer": gt["multiple_choice_answer"],
            }
        )
        print("  [{}/{}] image {} q {}".format(len(samples), limit, name, qid))
    manifest_path = os.path.join(dest, "vqa_manifest.json")
    with open(manifest_path, "w") as handle:
        json.dump({"task": "vqa", "samples": samples}, handle, indent=2)
    return len(samples), manifest_path


def main(argv=None):
    """Parse the arguments and fetch the subset."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", default="vqa", choices=["vqa"], help="dataset task to fetch")
    parser.add_argument("--dest", default=DEFAULT_DEST, help="folder to write into")
    parser.add_argument(
        "--limit", type=int, default=DEFAULT_LIMIT, help="how many samples to write"
    )
    args = parser.parse_args(argv)
    written, manifest = fetch(args.dest, args.limit)
    if not written:
        print("no samples fetched", file=sys.stderr)
        return 1
    print("wrote {} sample(s) and {}".format(written, manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
