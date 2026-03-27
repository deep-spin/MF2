import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import re
from typing import List, Tuple


@dataclass
class SrtBlock:
    index: int
    start_ms: int
    end_ms: int
    lines: List[str]


SRT_RANGE_PATTERN = re.compile(
    r"^\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}$"
)


def parse_hhmmss_to_ms(ts: str) -> int:
    parts = ts.strip().split(":")
    if len(parts) != 3:
        raise ValueError(f"Invalid timestamp format: {ts}")
    h, m, s = parts
    return (int(h) * 3600 + int(m) * 60 + int(s)) * 1000


def parse_srt_time_to_ms(ts: str) -> int:
    # Example: 00:12:34,567
    main, millis = ts.strip().split(",")
    return parse_hhmmss_to_ms(main) + int(millis)


def ms_to_srt(ms: int) -> str:
    if ms < 0:
        ms = 0
    total_seconds, millis = divmod(ms, 1000)
    hours, rem = divmod(total_seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def parse_srt_blocks(content: str) -> List[SrtBlock]:
    lines = content.splitlines()
    blocks: List[SrtBlock] = []
    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue

        # Index line
        idx_line = lines[i].strip()
        i += 1
        if i >= len(lines):
            break

        # Time line
        time_line = lines[i].strip()
        i += 1
        if " --> " not in time_line:
            continue
        start_str, end_str = time_line.split(" --> ")
        start_ms = parse_srt_time_to_ms(start_str)
        end_ms = parse_srt_time_to_ms(end_str)

        text_lines: List[str] = []
        while i < len(lines) and lines[i].strip():
            text_lines.append(lines[i])
            i += 1

        try:
            index = int(idx_line)
        except ValueError:
            index = len(blocks) + 1

        blocks.append(
            SrtBlock(
                index=index,
                start_ms=start_ms,
                end_ms=end_ms,
                lines=text_lines,
            )
        )

    return blocks


def validate_srt_timestamp_lines(content: str) -> Tuple[int, List[str]]:
    """
    Validate SRT timestamp range line format.
    Returns:
        - total number of detected range lines (containing '-->')
        - list of malformed range lines
    """
    total_range_lines = 0
    malformed: List[str] = []
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if "-->" not in line:
            continue
        total_range_lines += 1
        if not SRT_RANGE_PATTERN.match(line):
            malformed.append(line)
    return total_range_lines, malformed


def crop_blocks(blocks: List[SrtBlock], window_start_ms: int, window_end_ms: int) -> List[SrtBlock]:
    cropped: List[SrtBlock] = []
    for block in blocks:
        # Keep blocks that overlap with target window.
        if block.end_ms < window_start_ms or block.start_ms > window_end_ms:
            continue
        start_ms = max(block.start_ms, window_start_ms)
        end_ms = min(block.end_ms, window_end_ms)
        if end_ms < start_ms:
            continue
        cropped.append(
            SrtBlock(
                index=len(cropped) + 1,
                start_ms=start_ms,
                end_ms=end_ms,
                lines=block.lines,
            )
        )
    return cropped


def render_srt(blocks: List[SrtBlock]) -> str:
    out_lines: List[str] = []
    for block in blocks:
        out_lines.append(str(block.index))
        out_lines.append(f"{ms_to_srt(block.start_ms)} --> {ms_to_srt(block.end_ms)}")
        out_lines.extend(block.lines)
        out_lines.append("")
    return "\n".join(out_lines).rstrip() + "\n"


def parse_window(window_text: str) -> Tuple[int, int]:
    parts = window_text.strip().split("-")
    if len(parts) != 2:
        raise ValueError(f"Invalid Timestamps value: {window_text}")
    start_ms = parse_hhmmss_to_ms(parts[0])
    end_ms = parse_hhmmss_to_ms(parts[1])
    if end_ms < start_ms:
        raise ValueError(f"Invalid range (end < start): {window_text}")
    return start_ms, end_ms


def main() -> None:
    parser = argparse.ArgumentParser(description="Crop claim transcripts by timestamp windows from CSV.")
    parser.add_argument("--csv_path", required=True, help="Path to successfully_processed_claims.csv")
    parser.add_argument("--transcripts_full_dir", required=True, help="Path to transcripts_full directory")
    parser.add_argument("--cropped_transcripts_dir", required=True, help="Path to output cropped_transcripts directory")
    parser.add_argument("--report_path", required=True, help="Path to output markdown report")
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    full_dir = Path(args.transcripts_full_dir)
    out_dir = Path(args.cropped_transcripts_dir)
    report_path = Path(args.report_path)

    out_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    written = 0
    missing_inputs: List[str] = []
    empty_crops: List[str] = []
    malformed_timestamp_files: List[str] = []
    malformed_timestamp_examples: List[str] = []
    files_checked = 0
    total_range_lines_checked = 0
    unique_source_files = set()

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            movie_id = row["movie_id"].strip()
            claim_id = row["claim_id"].strip()
            src_file = full_dir / movie_id / f"{movie_id}_claim_{claim_id}.srt"
            unique_source_files.add(src_file)

    # First pass: timestamp format validation for all referenced files.
    for src_file in sorted(unique_source_files):
        if not src_file.exists():
            continue
        files_checked += 1
        content = src_file.read_text(encoding="utf-8")
        total_ranges, malformed = validate_srt_timestamp_lines(content)
        total_range_lines_checked += total_ranges
        if malformed:
            rel = src_file.relative_to(full_dir)
            malformed_timestamp_files.append(str(rel))
            for bad_line in malformed[:3]:
                malformed_timestamp_examples.append(f"{rel}: {bad_line}")

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_rows += 1
            movie_id = row["movie_id"].strip()
            claim_id = row["claim_id"].strip()
            timestamps = row["Timestamps"].strip()

            src_file = full_dir / movie_id / f"{movie_id}_claim_{claim_id}.srt"
            dst_file = out_dir / movie_id / f"{movie_id}_claim_{claim_id}.srt"

            if not src_file.exists():
                missing_inputs.append(f"{movie_id}_claim_{claim_id}")
                continue

            try:
                window_start_ms, window_end_ms = parse_window(timestamps)
                content = src_file.read_text(encoding="utf-8")
                blocks = parse_srt_blocks(content)
                cropped_blocks = crop_blocks(blocks, window_start_ms, window_end_ms)

                dst_file.parent.mkdir(parents=True, exist_ok=True)
                dst_file.write_text(render_srt(cropped_blocks), encoding="utf-8")

                if not cropped_blocks:
                    empty_crops.append(f"{movie_id}_claim_{claim_id}")
                written += 1
            except Exception:
                missing_inputs.append(f"{movie_id}_claim_{claim_id}")

    report_lines = [
        "# Cropped Transcript Generation Report",
        "",
        "## Timestamp Format Check (Before Cropping)",
        f"- Referenced transcript files checked: {files_checked}",
        f"- Total subtitle timestamp range lines checked: {total_range_lines_checked}",
        f"- Files with malformed timestamp range lines: {len(malformed_timestamp_files)}",
        "",
    ]

    if malformed_timestamp_files:
        report_lines.extend(
            [
                "### Files With Malformed Timestamp Lines",
                "",
                *[f"- `{item}`" for item in malformed_timestamp_files],
                "",
                "### Malformed Timestamp Examples",
                "",
                *[f"- `{item}`" for item in malformed_timestamp_examples[:30]],
                "",
            ]
        )
    else:
        report_lines.extend(
            [
                "- All detected SRT timestamp range lines match `HH:MM:SS,mmm --> HH:MM:SS,mmm`.",
                "",
            ]
        )

    report_lines.extend([
        f"- CSV input: `{csv_path}`",
        f"- Full transcripts directory: `{full_dir}`",
        f"- Cropped transcripts directory: `{out_dir}`",
        "",
        "## Cropping Method",
        "- For each CSV row, use `movie_id` and `claim_id` to locate source file: `transcripts_full/{movie_id}/{movie_id}_claim_{claim_id}.srt`.",
        "- Parse the row `Timestamps` field (`start-end`) into a target time window.",
        "- Parse subtitle blocks from the source SRT.",
        "- Keep only subtitle blocks that overlap the window.",
        "- Clip the first/last overlapping subtitle timestamps to fit exactly in the requested window.",
        "- Re-index remaining blocks from 1 and write to mirrored output path in `cropped_transcripts`.",
        "",
        "## Summary",
        f"- Total CSV rows processed: {total_rows}",
        f"- Output files written: {written}",
        f"- Missing/failed inputs: {len(missing_inputs)}",
        f"- Empty crops (no subtitle overlap with window): {len(empty_crops)}",
        "",
        "## Output Structure",
        "- Files are written as: `cropped_transcripts/{movie_id}/{movie_id}_claim_{claim_id}.srt`",
        "- This mirrors the structure of `transcripts_full`.",
        "",
    ])

    if missing_inputs:
        report_lines.extend(
            [
                "## Missing/Failed Inputs",
                "",
                *[f"- `{item}`" for item in missing_inputs],
                "",
            ]
        )

    if empty_crops:
        report_lines.extend(
            [
                "## Empty Crops",
                "",
                *[f"- `{item}`" for item in empty_crops],
                "",
            ]
        )

    report_path.write_text("\n".join(report_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
