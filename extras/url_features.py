"""
Feature extraction from raw URLs.

The URL-Phish dataset describes each URL with 22 numeric lexical and structural
features. To let the dashboard accept a URL typed by a user rather than only
pre-computed rows, those features have to be reproduced from the URL string.
"""

from __future__ import annotations

import ipaddress
import math
import re
from collections import Counter
from urllib.parse import urlparse

import pandas as pd
import tldextract

# A no-cache extractor avoids writing a public-suffix snapshot into the project
# directory on first use.
_EXTRACT = tldextract.TLDExtract(cache_dir=None)

# The 22 numeric features, in the order they appear in the published dataset.
FEATURE_NAMES = [
    "url_len",
    "dom_len",
    "is_ip",
    "tld_len",
    "subdom_cnt",
    "letter_cnt",
    "digit_cnt",
    "special_cnt",
    "eq_cnt",
    "qm_cnt",
    "amp_cnt",
    "dot_cnt",
    "dash_cnt",
    "under_cnt",
    "letter_ratio",
    "digit_ratio",
    "spec_ratio",
    "is_https",
    "slash_cnt",
    "entropy",
    "path_len",
    "query_len",
]

# Short, readable descriptions for the dashboard's explanation panel.
FEATURE_DESCRIPTIONS = {
    "url_len": "Total length of the URL",
    "dom_len": "Length of the registrable domain",
    "is_ip": "Host is a raw IP address rather than a domain name",
    "tld_len": "Length of the public suffix",
    "subdom_cnt": "Number of subdomain labels",
    "letter_cnt": "Count of alphabetic characters",
    "digit_cnt": "Count of digits",
    "special_cnt": "Count of non-alphanumeric characters",
    "eq_cnt": "Count of equals signs",
    "qm_cnt": "Count of question marks",
    "amp_cnt": "Count of ampersands",
    "dot_cnt": "Count of dots",
    "dash_cnt": "Count of hyphens",
    "under_cnt": "Count of underscores",
    "letter_ratio": "Proportion of characters that are letters",
    "digit_ratio": "Proportion of characters that are digits",
    "spec_ratio": "Proportion of characters that are special",
    "is_https": "URL uses HTTPS",
    "slash_cnt": "Count of forward slashes",
    "entropy": "Shannon entropy of the URL in bits",
    "path_len": "Length of the URL path",
    "query_len": "Length of the query string",
}


def shannon_entropy(text: str) -> float:
    """Shannon entropy of a string in bits."""
    if not text:
        return 0.0
    counts = Counter(text)
    n = len(text)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def _is_ip_host(host: str) -> bool:
    """Whether the host component is a literal IP address."""
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


def parse_url(url: str) -> dict:
    """Return the structural components the features are derived from."""
    url = (url or "").strip()

    # urlparse needs a scheme to populate netloc rather than path.
    if "://" not in url:
        parsed = urlparse("http://" + url)
        scheme_given = False
    else:
        parsed = urlparse(url)
        scheme_given = True

    host = parsed.hostname or ""
    is_ip = _is_ip_host(host)

    if is_ip:
        registrable, suffix, subdomain = host, "", ""
    else:
        parts = _EXTRACT(host)
        suffix = parts.suffix
        registrable = ".".join(p for p in (parts.domain, parts.suffix) if p)
        subdomain = parts.subdomain

    return {
        "url": url,
        "scheme": parsed.scheme if scheme_given else "",
        "host": host,
        "registrable": registrable,
        "suffix": suffix,
        "subdomain": subdomain,
        "path": parsed.path,
        "query": parsed.query,
        "is_ip": is_ip,
    }


def extract_features(url: str) -> dict:
    """Extract the 22 numeric features from a raw URL.

    Counts are taken over the whole URL string as supplied, which is how the
    published dataset defines them.
    """
    p = parse_url(url)
    text = p["url"]
    n = len(text)

    letters = sum(ch.isalpha() for ch in text)
    digits = sum(ch.isdigit() for ch in text)
    specials = n - letters - digits

    subdom_cnt = len([lbl for lbl in p["subdomain"].split(".") if lbl]) if p["subdomain"] else 0

    features = {
        "url_len": n,
        "dom_len": len(p["registrable"]),
        "is_ip": int(p["is_ip"]),
        "tld_len": len(p["suffix"]),
        "subdom_cnt": subdom_cnt,
        "letter_cnt": letters,
        "digit_cnt": digits,
        "special_cnt": specials,
        "eq_cnt": text.count("="),
        "qm_cnt": text.count("?"),
        "amp_cnt": text.count("&"),
        "dot_cnt": text.count("."),
        "dash_cnt": text.count("-"),
        "under_cnt": text.count("_"),
        "letter_ratio": letters / n if n else 0.0,
        "digit_ratio": digits / n if n else 0.0,
        "spec_ratio": specials / n if n else 0.0,
        "is_https": int(p["scheme"].lower() == "https"),
        "slash_cnt": text.count("/"),
        "entropy": shannon_entropy(text),
        "path_len": len(p["path"]),
        "query_len": len(p["query"]),
    }

    return features


def extract_frame(url: str, feature_order: list[str] | None = None) -> pd.DataFrame:
    """Extract features as a single-row DataFrame ready for a fitted pipeline."""
    features = extract_features(url)
    order = feature_order or FEATURE_NAMES
    return pd.DataFrame([[features[name] for name in order]], columns=order)


# ---------------------------------------------------------------------------
# Verification against the published dataset
# ---------------------------------------------------------------------------
def verify(n_samples: int = 2000, tolerance: float = 1e-6, verbose: bool = True) -> pd.DataFrame:
    """Compare extracted features against the published values.

    Checks the extraction on a random sample of real rows and reports, for each
    feature, the proportion that match. Anything short of a near-perfect match
    indicates the definition recovered here differs from the dataset's own.
    """
    from config import DATA_DIR

    path = DATA_DIR / "urlphish_dataset.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Load the dataset first.")

    df = pd.read_csv(path).sample(
        n=min(n_samples, sum(1 for _ in open(path)) - 1), random_state=42
    )

    rows = []
    for url in df["url"]:
        rows.append(extract_features(url))
    extracted = pd.DataFrame(rows, index=df.index)

    report = []
    for name in FEATURE_NAMES:
        expected = pd.to_numeric(df[name], errors="coerce").fillna(0)
        actual = extracted[name]
        matches = (actual - expected).abs() <= tolerance
        report.append(
            {
                "feature": name,
                "match_rate": round(float(matches.mean()), 4),
                "mismatches": int((~matches).sum()),
            }
        )

    result = pd.DataFrame(report).sort_values("match_rate")

    if verbose:
        overall = result["match_rate"].mean()
        print(f"Verified {len(df)} URLs against the published feature values.\n")
        print(result.to_string(index=False))
        print(f"\nMean match rate across features: {overall:.4f}")
        imperfect = result[result["match_rate"] < 1.0]
        if imperfect.empty:
            print("All features reproduce the published values exactly.")
        else:
            print(
                f"{len(imperfect)} feature(s) do not match exactly; "
                "inspect before relying on URL input."
            )

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Verify URL feature extraction.")
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--url", default=None, help="Extract features for one URL.")
    args = parser.parse_args()

    if args.url:
        p = parse_url(args.url)
        print(f"URL: {p['url']}")
        print(f"  host={p['host']} registrable={p['registrable']} suffix={p['suffix']}")
        print()
        for k, v in extract_features(args.url).items():
            print(f"  {k:15s} {v}")
    else:
        verify(args.samples)