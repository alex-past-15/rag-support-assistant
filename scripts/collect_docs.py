"""Download the selected help articles and extract their content.

Run: python scripts/collect_docs.py
Use --refresh to download again instead of using saved HTML.
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup
from markdownify import markdownify

ROOT = Path(__file__).resolve().parents[1]


def extract(html, url):
    soup = BeautifulSoup(html, "html.parser")
    title = soup.select_one("h1.dc-doc-page-title")
    body = soup.select_one(".dc-doc-page__body")
    if title is None:
        title = soup.find("title")
    if title is None or body is None:
        raise ValueError("Article title/body missing; possible error page or challenge")
    title_text = title.get_text(" ", strip=True)
    for node in body.select("script, style, button, svg"):
        node.decompose()
    base_node = soup.find("base", href=True)
    base = urljoin(url, base_node["href"]) if base_node else url
    images = []
    for node in body.select("a[href], img[src]"):
        attr = "href" if node.name == "a" else "src"
        node[attr] = urljoin(base, node[attr])
        if node.name == "img":
            images.append({"url": node[attr], "alt": node.get("alt", "")})
    text = markdownify(str(body), heading_style="ATX").strip()
    if len(body.get_text(" ", strip=True)) < 150:
        raise ValueError("Article body unexpectedly short")
    return title_text, text, images


def download(url):
    last_error = None
    for attempt in range(3):
        try:
            request = Request(url, headers={"User-Agent": "Mozilla/5.0", "Accept-Language": "ru"})
            with urlopen(request, timeout=35) as response:
                final_url = response.url
                if urlparse(final_url).hostname not in {"yandex.ru", "www.yandex.ru"}:
                    raise ValueError("Unexpected redirect outside documentation host")
                html = response.read().decode("utf-8")
            extract(html, final_url)  # Never cache a challenge as a valid article.
            return html, final_url
        except Exception as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(2 * (attempt + 1))
    raise RuntimeError(str(last_error))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    raw = ROOT / "data/raw"
    processed = ROOT / "data/processed"
    raw.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)
    sources = json.loads((ROOT / "data/sources.json").read_text(encoding="utf-8-sig"))
    report = []
    for source in sources:
        doc_id, url = source["id"], source["url"]
        html_path = raw / f"{doc_id}.html"
        meta_path = raw / f"{doc_id}.json"
        try:
            if html_path.exists() and meta_path.exists() and not args.refresh:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if meta["source_url"] != url:
                    raise ValueError("Cached URL differs; run with --refresh")
                html = html_path.read_text(encoding="utf-8")
            else:
                html, final_url = download(url)
                meta = {"source_url": url, "final_url": final_url,
                        "downloaded_at": datetime.now(timezone.utc).isoformat()}
                html_path.write_text(html, encoding="utf-8")
                meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
                time.sleep(0.5)
            title, body, images = extract(html, meta["final_url"])
            document = {"id": doc_id, "title": title, **meta, "text": body,
                        "images": images, "needs_image_review": bool(images)}
            (processed / f"{doc_id}.json").write_text(
                json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
            (processed / f"{doc_id}.md").write_text(
                f"# {title}\n\nИсточник: {url}\n\nДата загрузки: {meta['downloaded_at']}\n\n{body}\n", encoding="utf-8")
            report.append({"id": doc_id, "status": "ok", "title": title,
                           "characters": len(body), "images": len(images)})
            print(f"{doc_id}: OK ({len(body)} characters, {len(images)} images)", flush=True)
        except Exception as exc:
            report.append({"id": doc_id, "status": "error", "error": str(exc)})
            print(f"{doc_id}: ERROR: {exc}", flush=True)
    (ROOT / "data/collection_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    failures = sum(row["status"] == "error" for row in report)
    print(f"Collected {len(report) - failures}/{len(report)} articles")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
