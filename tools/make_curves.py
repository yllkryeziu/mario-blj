"""Inlines the reduced ladder curves into the learning curve page."""

from __future__ import annotations

import json
import os

from absl import app, flags, logging

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_CURVES = flags.DEFINE_string("curves", os.path.join(_ROOT, "results", "curves_page.json"),
                              "Payload written by tools/prep_curves.py.")
_TEMPLATE = flags.DEFINE_string("template", os.path.join(_ROOT, "tools", "curves_template.html"),
                                "Template carrying the __REPLAY_DATA__ placeholder.")
_OUT = flags.DEFINE_string("out", os.path.join(_ROOT, "results", "curves.html"),
                           "Page to write.")

PLACEHOLDER = "__REPLAY_DATA__"


def main(argv: list[str]) -> None:
    """Writes the page and reports its size.

    Args:
        argv: Unused positional arguments.

    Raises:
        ValueError: If the template has no placeholder to fill.
    """
    del argv
    with open(_CURVES.value, encoding="utf-8") as handle:
        payload = json.load(handle)
    with open(_TEMPLATE.value, encoding="utf-8") as handle:
        template = handle.read()
    if PLACEHOLDER not in template:
        raise ValueError(f"{_TEMPLATE.value} has no {PLACEHOLDER}")

    html = template.replace(PLACEHOLDER, json.dumps(payload, separators=(",", ":")))
    os.makedirs(os.path.dirname(_OUT.value), exist_ok=True)
    with open(_OUT.value, "w", encoding="utf-8") as handle:
        handle.write(html)

    solved = sum(rung["solved"] for rung in payload["rungs"])
    total = sum(rung["total"] for rung in payload["rungs"])
    logging.info("wrote %s", _OUT.value)
    print(f"wrote {_OUT.value}")
    print(f"  {len(payload['rungs'])} rungs, {total} runs, {solved} solved")
    print(f"  page {os.path.getsize(_OUT.value):,} B "
          f"({os.path.getsize(_OUT.value) / 16e6 * 100:.1f}% of the 16 MB budget)")


if __name__ == "__main__":
    app.run(main)
