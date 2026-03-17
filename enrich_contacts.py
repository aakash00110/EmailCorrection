import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass

import pandas as pd


FIELDS_TO_ENRICH = ("email", "first_name", "last_name")


def _is_empty(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


_SCHEME_RE = re.compile(r"^\s*(?:https?://)?", re.IGNORECASE)
_WWW_RE = re.compile(r"^\s*www\.", re.IGNORECASE)


def normalize_domain(raw: str) -> str:
    if raw is None:
        return ""
    value = str(raw).strip().lower()
    if value == "":
        return ""

    value = _SCHEME_RE.sub("", value)
    value = _WWW_RE.sub("", value)

    value = value.split("/", 1)[0]
    value = value.split("?", 1)[0]
    value = value.split("#", 1)[0]

    value = value.strip().strip(".")
    return value


def _find_domain_column(columns) -> str:
    for name in columns:
        if isinstance(name, str) and name.strip().lower() == "domain":
            return name
    return ""


@dataclass(frozen=True)
class RunStats:
    rows_scanned: int
    rows_missing_info: int
    rows_enriched_successfully: int
    rows_with_no_domain_match: int
    process_completed: bool


def build_master_index(master_df: pd.DataFrame) -> dict[str, dict[str, str]]:
    domain_col = _find_domain_column(master_df.columns)
    if not domain_col:
        raise ValueError(
            f"Master CSV must contain a 'domain' column (case-insensitive). "
            f"Found columns: {list(master_df.columns)}"
        )

    master_df = master_df.copy()
    master_df["__domain_norm"] = master_df[domain_col].map(normalize_domain)

    index: dict[str, dict[str, str]] = {}
    for _, row in master_df.iterrows():
        domain_norm = row.get("__domain_norm", "")
        if _is_empty(domain_norm):
            continue
        if domain_norm in index:
            continue

        record: dict[str, str] = {}
        for field in FIELDS_TO_ENRICH:
            if field in master_df.columns:
                value = row.get(field, "")
                if _is_empty(value):
                    record[field] = ""
                else:
                    record[field] = str(value).strip()
            else:
                record[field] = ""

        index[domain_norm] = record

    return index


def enrich_target_df(target_df: pd.DataFrame, master_index: dict[str, dict[str, str]]) -> tuple[pd.DataFrame, RunStats]:
    domain_col = _find_domain_column(target_df.columns)
    if not domain_col:
        raise ValueError(
            f"Target CSV must contain a 'domain' column (case-insensitive). "
            f"Found columns: {list(target_df.columns)}"
        )

    target_df = target_df.copy()
    target_df["__domain_norm"] = target_df[domain_col].map(normalize_domain)

    rows_scanned = int(len(target_df))
    rows_missing_info = 0
    rows_enriched_successfully = 0
    rows_with_no_domain_match = 0

    for idx, row in target_df.iterrows():
        missing_any = False
        for field in FIELDS_TO_ENRICH:
            if field not in target_df.columns:
                target_df[field] = ""
            if _is_empty(row.get(field, "")):
                missing_any = True
        if not missing_any:
            continue

        rows_missing_info += 1

        domain_norm = row.get("__domain_norm", "")
        if _is_empty(domain_norm) or domain_norm not in master_index:
            rows_with_no_domain_match += 1
            continue

        master_record = master_index[domain_norm]
        changed = False
        for field in FIELDS_TO_ENRICH:
            current_value = row.get(field, "")
            if not _is_empty(current_value):
                continue
            candidate = master_record.get(field, "")
            if _is_empty(candidate):
                continue
            target_df.at[idx, field] = candidate
            changed = True

        if changed:
            rows_enriched_successfully += 1

    target_df = target_df.drop(columns=["__domain_norm"])

    return (
        target_df,
        RunStats(
            rows_scanned=rows_scanned,
            rows_missing_info=rows_missing_info,
            rows_enriched_successfully=rows_enriched_successfully,
            rows_with_no_domain_match=rows_with_no_domain_match,
            process_completed=True,
        ),
    )


def read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[])


def write_csv(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enrich missing contact fields in a target CSV using a master CSV by domain.")
    parser.add_argument("--target", required=True, help="Path to the target CSV (incomplete leads).")
    parser.add_argument("--master", required=True, help="Path to the master CSV (reference dataset).")
    parser.add_argument("--in-place", action="store_true", help="Overwrite the target CSV file with enriched output.")
    parser.add_argument("--output", help="Write enriched CSV to this path instead of overwriting the target.")
    parser.add_argument("--dry-run", action="store_true", help="Do not write any files; only print stats.")
    parser.add_argument(
        "--reset-email",
        action="store_true",
        help="Clear existing email values in the target before enrichment.",
    )
    parser.add_argument("--json", action="store_true", help="Print stats as JSON (single line).")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    if args.in_place and args.output:
        raise SystemExit("Use either --in-place or --output, not both.")
    if not args.in_place and not args.output and not args.dry_run:
        raise SystemExit("Choose an output mode: --in-place, --output, or --dry-run.")

    target_df = read_csv(args.target)
    master_df = read_csv(args.master)

    if args.reset_email and "email" in target_df.columns:
        target_df = target_df.copy()
        target_df["email"] = ""

    master_index = build_master_index(master_df)

    enriched_df, stats = enrich_target_df(target_df, master_index)

    if args.json:
        print(json.dumps(stats.__dict__, separators=(",", ":"), sort_keys=True))
    else:
        print(f"Rows scanned: {stats.rows_scanned}")
        print(f"Rows missing info: {stats.rows_missing_info}")
        print(f"Rows enriched successfully: {stats.rows_enriched_successfully}")
        print(f"Rows with no domain match: {stats.rows_with_no_domain_match}")
        print("Process completed.")

    if args.dry_run:
        return 0

    output_path = args.target if args.in_place else args.output
    if output_path is None:
        raise SystemExit("Missing output path.")

    write_csv(enriched_df, output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
