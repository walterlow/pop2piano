"""
Download script with friendly filenames (titles instead of YouTube IDs)

Usage:
python download/download_friendly.py train_dataset.csv /output/dir --num_audio 5
"""

import os
import multiprocessing
import tempfile
import shutil
import glob
import re

import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from omegaconf import OmegaConf


def sanitize_filename(name: str, max_length: int = 100) -> str:
    """Convert a string to a safe filename."""
    # Remove or replace invalid characters
    name = re.sub(r'[<>:"/\\|?*]', '', name)
    # Replace multiple spaces/underscores with single underscore
    name = re.sub(r'[\s_]+', '_', name)
    # Remove leading/trailing whitespace and dots
    name = name.strip().strip('.')
    # Truncate to max length
    if len(name) > max_length:
        name = name[:max_length]
    return name


def download_piano(
    url: str,
    output_dir: str,
    postprocess=True,
    dry_run=False,
) -> int:
    with tempfile.TemporaryDirectory() as tmpdir:
        output = f"{tmpdir}/%(uploader)s___%(title)s___%(id)s___%(duration)d.%(ext)s"

        if postprocess:
            postprocess_call = '--postprocessor-args "ffmpeg:-ac 1 -ar 16000"'
        else:
            postprocess_call = ""
        result = os.system(
            f"""youtube-dl -o "{output}" \\
                --extract-audio \\
                --audio-quality 0 \\
                --audio-format wav \\
                --retries 50 \\
                {"--get-filename" if dry_run else ""}\\
                {postprocess_call} \\
                --force-ipv4 \\
                --yes-playlist \\
                --ignore-errors \\
                {url}"""
        )

        if not dry_run:
            files = os.listdir(tmpdir)

            for filename in files:
                filename_wo_ext, ext = os.path.splitext(filename)
                uploader, title, ytid, duration = filename_wo_ext.split("___")

                # Create friendly filename from title
                friendly_name = sanitize_filename(f"{uploader}_{title}")

                meta = OmegaConf.create()
                meta.piano = OmegaConf.create()
                meta.piano.uploader = uploader
                meta.piano.title = title
                meta.piano.ytid = ytid
                meta.piano.duration = int(duration)
                meta.piano.friendly_name = friendly_name

                # Save with friendly name
                OmegaConf.save(meta, os.path.join(output_dir, friendly_name + ".yaml"))
                shutil.move(
                    os.path.join(tmpdir, filename),
                    os.path.join(output_dir, f"{friendly_name}{ext}"),
                )

    return result


def download_piano_main(piano_list, output_dir, dry_run=False):
    """
    piano_list : list of youtube id
    """
    os.makedirs(output_dir, exist_ok=True)
    Parallel(n_jobs=multiprocessing.cpu_count())(
        delayed(download_piano)(
            url=f"https://www.youtube.com/watch?v={ytid}",
            output_dir=output_dir,
            postprocess=True,
            dry_run=dry_run,
        )
        for ytid in tqdm(piano_list)
    )


def download_pop(piano_id, pop_id, output_dir, dry_run):
    """Download pop song and save with friendly name."""
    # First, find the piano yaml to get the friendly name
    yaml_files = glob.glob(os.path.join(output_dir, "*.yaml"))
    piano_friendly_name = None
    piano_yaml_path = None

    for yaml_file in yaml_files:
        meta = OmegaConf.load(yaml_file)
        if hasattr(meta, 'piano') and meta.piano.ytid == piano_id:
            # Handle yaml files from original script (no friendly_name)
            if 'friendly_name' in meta.piano:
                piano_friendly_name = meta.piano.friendly_name
            else:
                piano_friendly_name = piano_id
            piano_yaml_path = yaml_file
            break

    if piano_friendly_name is None:
        print(f"Warning: Could not find piano metadata for {piano_id}")
        return

    output_file_template = "%(id)s___%(title)s___%(duration)d.%(ext)s"
    pop_output_dir = os.path.join(output_dir, piano_friendly_name)
    os.makedirs(pop_output_dir, exist_ok=True)
    output_template = os.path.join(pop_output_dir, output_file_template)
    url = f"https://www.youtube.com/watch?v={pop_id}"

    result = os.system(
        f"""youtube-dl -o "{output_template}" \\
            --extract-audio \\
            --audio-quality 0 \\
            --audio-format wav \\
            --retries 25 \\
            {"--get-filename" if dry_run else ""}\\
            --match-filter 'duration < 300 & duration > 150'\\
            --postprocessor-args "ffmpeg:-ac 2 -ar 44100" \\
            {url}"""
    )

    if not dry_run:
        files = glob.glob(os.path.join(pop_output_dir, "*.wav"))
        for filename in files:
            filename_wo_ext, ext = os.path.splitext(os.path.basename(filename))
            parts = filename_wo_ext.split("___")
            if len(parts) != 3:
                continue
            ytid, title, duration = parts

            # Create friendly name for pop song
            pop_friendly_name = sanitize_filename(title)

            # Update the piano yaml with pop info
            meta = OmegaConf.load(piano_yaml_path)
            meta.song = OmegaConf.create()
            meta.song.ytid = ytid
            meta.song.title = title
            meta.song.duration = int(duration)
            meta.song.friendly_name = pop_friendly_name
            OmegaConf.save(meta, piano_yaml_path)

            # Rename the file to friendly name
            shutil.move(
                filename,
                os.path.join(pop_output_dir, f"{pop_friendly_name}{ext}"),
            )


def download_pop_main(piano_list, pop_list, output_dir, dry_run=False):
    """
    piano_list : list of youtube id
    pop_list : corresponding youtube id of pop songs
    """
    Parallel(n_jobs=multiprocessing.cpu_count())(
        delayed(download_pop)(
            piano_id=piano_id,
            pop_id=pop_id,
            output_dir=output_dir,
            dry_run=dry_run,
        )
        for piano_id, pop_id in tqdm(list(zip(piano_list, pop_list)))
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Piano cover downloader with friendly filenames")

    parser.add_argument("dataset", type=str, default=None, help="provided csv")
    parser.add_argument("output_dir", type=str, default=None, help="output dir")
    parser.add_argument(
        "--num_audio",
        type=int,
        default=None,
        help="if specified, only {num_audio} pairs will be downloaded",
    )
    parser.add_argument(
        "--dry_run", default=False, action="store_true", help="whether dry_run"
    )
    args = parser.parse_args()

    df = pd.read_csv(args.dataset)
    df = df[: args.num_audio]
    piano_list = df["piano_ids"].tolist()
    download_piano_main(piano_list, args.output_dir, args.dry_run)

    available_piano_list = glob.glob(args.output_dir + "/**/*.yaml", recursive=True)

    # Build a mapping of ytid -> friendly_name from yaml files
    ytid_to_friendly = {}
    for yaml_path in available_piano_list:
        meta = OmegaConf.load(yaml_path)
        if hasattr(meta, 'piano'):
            ytid = meta.piano.ytid
            # Handle yaml files from original script (no friendly_name)
            if 'friendly_name' in meta.piano:
                ytid_to_friendly[ytid] = meta.piano.friendly_name
            else:
                # Use ytid as fallback for old yaml files
                ytid_to_friendly[ytid] = ytid

    df.index = df["piano_ids"]

    failed_piano = []
    for piano_id_to_be_downloaded in tqdm(df["piano_ids"]):
        if piano_id_to_be_downloaded in ytid_to_friendly:
            continue
        else:
            failed_piano.append(piano_id_to_be_downloaded)

    if len(failed_piano) > 0:
        print(f"{len(failed_piano)} of files are failed to be downloaded")
        df = df.drop(index=failed_piano)

    piano_list = df["piano_ids"].tolist()
    pop_list = df["pop_ids"].tolist()

    download_pop_main(
        piano_list, pop_list, output_dir=args.output_dir, dry_run=args.dry_run
    )
