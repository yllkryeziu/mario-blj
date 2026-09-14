"""Renders the 2D replay viewer by inlining the packed replays into the HTML template.

The viewer has to work from a file:// URL and out of a results directory that is not served by
anything, so it cannot fetch its data. The whole payload is substituted into the template as a
JSON literal instead, which makes the output a single self contained page.
"""

import argparse
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    """Substitutes the packed replays into the template and writes the page.

    Raises:
        OSError: If the replays, the template or the output path cannot be opened.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--replays", default=os.path.join(ROOT, "results", "replays.json"))
    parser.add_argument("--template", default=os.path.join(ROOT, "tools", "viewer_template.html"))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    with open(args.replays, encoding="utf-8") as handle:
        payload = json.load(handle)
    with open(args.template, encoding="utf-8") as handle:
        template = handle.read()

    html = template.replace("__REPLAY_DATA__", json.dumps(payload, separators=(",", ":")))
    with open(args.out, "w", encoding="utf-8") as handle:
        handle.write(html)
    print(f"wrote {args.out} ({os.path.getsize(args.out) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
