# -*- coding: utf-8 -*-

import csv
import os
import re
from pathlib import Path


def env_int(name, default):
    value = os.environ.get(name)
    return default if value is None or value == "" else int(value)


def env_float(name, default):
    value = os.environ.get(name)
    return default if value is None or value == "" else float(value)


def env_bool(name, default):
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value not in {"0", "false", "False", "no", "No"}


def env_csv(name, default=None):
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return [] if default is None else default
    return [item.strip() for item in value.split(",") if item.strip()]


def env_output_dir(name, default, base_dir=None):
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    path = Path(value)
    if path.is_absolute() or base_dir is None:
        return path
    return base_dir / path


def format_float(value, precision=16):
    if value == "":
        return ""
    return f"{float(value):.{precision}f}"


def safe_filename_stem(value):
    return re.sub(r"[^0-9A-Za-z._-]+", "_", str(value).lower()).strip("._-")


def save_rows_to_csv(filename, rows, fieldnames=None):
    if not rows:
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with open(filename, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_progress(current, total, prefix="", width=32):
    ratio = current / total
    completed = int(width * ratio)
    bar = "#" * completed + "-" * (width - completed)
    print(f"\r{prefix} [{bar}] {current}/{total} {ratio * 100:6.2f}%", end="", flush=True)


def select_enabled_items(
    suite_names,
    suite_mapping,
    enabled_ids=None,
    id_key="id",
    suite_label="suite",
    empty_message="No item selected.",
):
    items = []
    for suite_name in suite_names:
        if suite_name not in suite_mapping:
            valid_names = ", ".join(suite_mapping)
            raise ValueError(f"Unknown {suite_label}: {suite_name}. Valid {suite_label}s: {valid_names}")
        items.extend(suite_mapping[suite_name])

    if enabled_ids:
        enabled_ids = set(enabled_ids)
        items = [item for item in items if item[id_key] in enabled_ids]

    if not items:
        raise ValueError(empty_message)
    return items


def select_named_items(items, enabled_names, name_key="name", item_label="item"):
    if not enabled_names:
        return items

    enabled_names = set(enabled_names)
    selected = [item for item in items if item[name_key] in enabled_names]
    missing_names = enabled_names - {item[name_key] for item in items}
    if missing_names:
        valid_names = ", ".join(item[name_key] for item in items)
        raise ValueError(f"Unknown {item_label}: {', '.join(sorted(missing_names))}. Valid {item_label}s: {valid_names}")
    if not selected:
        raise ValueError(f"No {item_label} selected.")
    return selected
