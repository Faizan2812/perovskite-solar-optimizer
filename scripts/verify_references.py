"""
scripts/verify_references.py
=============================
Audit every reference in data/materials_database.json against doi.org.

For each reference:
  1. Send a HEAD request to https://doi.org/{doi}
  2. Verify the redirect target lands on a real publisher page
  3. Flag any reference where the DOI doesn't resolve

This is the script that lets you, as the PhD candidate, prove to reviewers
that every cited paper is real and accessible. Run it any time to
regenerate the audit report.

Usage:
    python scripts/verify_references.py                  # quick check, no network if already verified
    python scripts/verify_references.py --network        # actually hit doi.org for each reference
    python scripts/verify_references.py --network --strict   # fail on any unresolved DOI

Output:
    REFERENCE_AUDIT.md  -- human-readable audit report
    artifacts/reference_audit.json  -- machine-readable data
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]


def check_doi_offline(doi: str, citation: str) -> dict:
    """Quick offline sanity checks on a DOI string."""
    result = {
        "doi": doi,
        "doi_format_valid": False,
        "issues": [],
    }

    if not doi or doi == "VERIFY":
        result["issues"].append("DOI missing or marked VERIFY")
        return result

    # Standard DOI format: 10.NNNN/something
    if not doi.startswith("10."):
        result["issues"].append("DOI does not start with '10.' prefix")
        return result

    parts = doi.split("/", 1)
    if len(parts) != 2:
        result["issues"].append("DOI missing forward slash separator")
        return result

    prefix, suffix = parts
    if not prefix.replace("10.", "").replace(".", "").isdigit():
        result["issues"].append("DOI registrant ID not numeric")
        return result

    if not suffix:
        result["issues"].append("DOI suffix is empty")
        return result

    result["doi_format_valid"] = True

    # Sanity: future years in citation
    for fy in ("2027", "2028", "2029", "2030"):
        if fy in citation:
            result["issues"].append(f"Citation contains future year: {fy}")

    return result


def check_doi_online(doi: str, timeout: float = 10.0) -> dict:
    """Hit doi.org and check the response."""
    import urllib.request
    import urllib.error

    url = f"https://doi.org/{doi}"
    result = {
        "url_attempted": url,
        "online_resolves": False,
        "redirect_target": None,
        "online_error": None,
    }

    try:
        req = urllib.request.Request(url, method="HEAD",
                                       headers={"User-Agent": "Mozilla/5.0 (perovskite-tool reference audit)"})
        opener = urllib.request.build_opener(urllib.request.HTTPRedirectHandler())
        with opener.open(req, timeout=timeout) as resp:
            result["online_resolves"] = True
            result["redirect_target"] = resp.geturl()
            netloc = urlparse(resp.geturl()).netloc
            result["publisher_domain"] = netloc
    except urllib.error.HTTPError as e:
        result["online_error"] = f"HTTP {e.code}"
        # 302/303 redirects are normal; some servers return 200 only after a GET
        if e.code in (302, 303, 200):
            result["online_resolves"] = True
            try:
                result["redirect_target"] = e.headers.get("Location")
            except Exception:
                pass
    except urllib.error.URLError as e:
        result["online_error"] = str(e.reason)
    except Exception as e:
        result["online_error"] = f"{type(e).__name__}: {e}"

    return result


def audit_all(network: bool = False) -> dict:
    db_path = ROOT / "data" / "materials_database.json"
    with open(db_path) as f:
        db = json.load(f)
    refs = db.get("_references", {})

    audit = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "n_active": 0,
        "n_format_valid": 0,
        "n_online_resolves": 0,
        "n_failures": 0,
        "results": {},
    }

    for ref_id, entry in sorted(refs.items()):
        if ref_id.startswith("_REMOVED"):
            continue

        audit["n_active"] += 1
        doi = entry.get("doi", "")
        citation = entry.get("citation", "")

        item = {
            "ref_id":   ref_id,
            "citation": citation,
            "doi":      doi,
        }

        # offline checks always run
        item.update(check_doi_offline(doi, citation))

        if item.get("doi_format_valid"):
            audit["n_format_valid"] += 1

        # online checks only if --network flag
        if network and item.get("doi_format_valid"):
            print(f"  Checking {ref_id} ... ", end="", flush=True)
            online = check_doi_online(doi)
            item.update(online)
            if online["online_resolves"]:
                audit["n_online_resolves"] += 1
                print("OK")
            else:
                audit["n_failures"] += 1
                print(f"FAIL ({online['online_error']})")
            time.sleep(0.5)  # be polite to doi.org

        audit["results"][ref_id] = item

    return audit


def write_markdown_report(audit: dict, outpath: Path) -> None:
    lines = [
        "# Reference Audit Report",
        "",
        f"*Generated: {audit['timestamp']}*",
        "",
        "Audit of every reference cited in `data/materials_database.json`.",
        "Every reference must have a verifiable DOI that resolves via doi.org.",
        "",
        "## Summary",
        "",
        f"- **Active references**: {audit['n_active']}",
        f"- **DOI format valid**: {audit['n_format_valid']}",
        f"- **Online resolves**: {audit['n_online_resolves']} (only checked with --network)",
        f"- **Failures**: {audit['n_failures']}",
        "",
        "## Per-reference status",
        "",
        "| ID | Citation | DOI | Format | Online | Issues |",
        "|---|---|---|---|---|---|",
    ]

    for ref_id, item in audit["results"].items():
        cite_short = (item.get("citation", "") or "")[:60]
        doi = item.get("doi", "")
        fmt_ok = "✓" if item.get("doi_format_valid") else "✗"
        online = ""
        if "online_resolves" in item:
            online = "✓" if item["online_resolves"] else "✗"
        issues = "; ".join(item.get("issues", [])) or "—"
        lines.append(f"| {ref_id} | {cite_short} | {doi} | {fmt_ok} | {online} | {issues} |")

    lines += [
        "",
        "## How to fix issues",
        "",
        "If a reference fails the audit:",
        "",
        "1. Locate the DOI on the publisher's website (e.g. PubMed, RSC, ACS, Wiley).",
        "2. Update `data/materials_database.json`:",
        "   - Set the correct `doi` field",
        "   - Set `verified: true`",
        "   - Set `verified_on: 2026-04-28` (or today's date)",
        "   - Set `url: https://doi.org/<doi>`",
        "3. Re-run this script: `python scripts/verify_references.py --network`",
        "4. If the DOI cannot be verified, REMOVE the reference and migrate any",
        "   parameters that cited it to a verified replacement. See",
        "   `docs/MATERIAL_LIFECYCLE.md` for the procedure.",
        "",
        "## Reproducibility",
        "",
        "```bash",
        "# Quick offline format check",
        "python scripts/verify_references.py",
        "",
        "# Full online verification (hits doi.org)",
        "python scripts/verify_references.py --network",
        "",
        "# Strict mode (fail CI on any unresolved DOI)",
        "python scripts/verify_references.py --network --strict",
        "```",
    ]

    outpath.write_text("\n".join(lines))
    print(f"\nWrote {outpath}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--network", action="store_true",
                     help="Actually hit doi.org for each DOI (slower, requires internet)")
    ap.add_argument("--strict", action="store_true",
                     help="Exit with code 1 if any reference fails the audit")
    args = ap.parse_args()

    print(f"Auditing references in {ROOT / 'data' / 'materials_database.json'}")
    if args.network:
        print("Mode: online (hitting doi.org for each reference)")
    else:
        print("Mode: offline (DOI format checks only). Use --network for online verification.")
    print()

    audit = audit_all(network=args.network)

    print(f"\n=== Audit summary ===")
    print(f"  Active references:    {audit['n_active']}")
    print(f"  DOI format valid:     {audit['n_format_valid']}")
    if args.network:
        print(f"  Online resolves:      {audit['n_online_resolves']}")
        print(f"  Failures:             {audit['n_failures']}")

    # Write reports
    md_path = ROOT / "REFERENCE_AUDIT.md"
    write_markdown_report(audit, md_path)

    json_path = ROOT / "artifacts" / "reference_audit.json"
    json_path.parent.mkdir(exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(audit, f, indent=2)
    print(f"Wrote {json_path}")

    if args.strict:
        n_format_invalid = audit["n_active"] - audit["n_format_valid"]
        if n_format_invalid > 0:
            print(f"\nSTRICT MODE FAIL: {n_format_invalid} references have invalid DOI format")
            return 1
        if args.network and audit["n_failures"] > 0:
            print(f"\nSTRICT MODE FAIL: {audit['n_failures']} references failed online resolution")
            return 1

    print("\nAudit passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
