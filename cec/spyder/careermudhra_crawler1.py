
import argparse
import json
import re
import time
from collections import deque
from datetime import datetime, timezone
from urllib.parse import urljoin, urlparse, urldefrag
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

USER_AGENT = "CareerMudhraCrawler/1.0"
TIMEOUT = 20

# File types that are not HTML pages
SKIP_EXTENSIONS = (
    ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".webp",
    ".zip", ".doc", ".docx", ".xls", ".xlsx", ".mp4"
)


def clean_url(url):
    """Remove fragments and ignore non-web URLs."""
    url = urldefrag(url)[0]
    parsed = urlparse(url)

    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return None

    return parsed._replace(
        netloc=parsed.netloc.lower()
    ).geturl()


def get_robots_parser(start_url, session):
    """Read robots.txt to check which pages may be crawled."""
    parsed = urlparse(start_url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"

    parser = RobotFileParser()
    parser.set_url(robots_url)

    try:
        response = session.get(robots_url, timeout=TIMEOUT)

        if response.status_code == 200:
            parser.parse(response.text.splitlines())
        else:
            # If robots.txt is unavailable, continue with normal checks.
            parser.parse([])

    except requests.RequestException:
        parser.parse([])

    return parser


def get_text(element):
    """Extract clean, whitespace-normalized text."""
    if not element:
        return ""

    return re.sub(
        r"\s+", " ", element.get_text(" ", strip=True)
    ).strip()


def extract_page(url, response):
    """Extract useful content from one HTML page."""
    soup = BeautifulSoup(response.text, "html.parser")

    # Remove code and decorative elements that are not useful as page text.
    for element in soup([
        "script", "style", "noscript", "template", "svg"
    ]):
        element.decompose()

    title = get_text(soup.title)

    description = ""
    description_tag = soup.find(
        "meta", attrs={"name": re.compile("^description$", re.I)}
    )
    if description_tag:
        description = description_tag.get("content", "").strip()

    canonical = ""
    canonical_tag = soup.find(
        "link", rel=lambda value: value and "canonical" in value
    )
    if canonical_tag and canonical_tag.get("href"):
        canonical = urljoin(url, canonical_tag["href"])

    # Prefer the article/main content over the whole webpage.
    main = (
        soup.find("main")
        or soup.find("article")
        or soup.find(attrs={"role": "main"})
        or soup.find(class_=re.compile(
            r"entry-content|post-content|article-content", re.I
        ))
        or soup.body
        or soup
    )

    # Headings
    headings = []
    for heading in main.find_all(
        re.compile(r"^h[1-6]$")
    ):
        text = get_text(heading)
        if text:
            headings.append({
                "level": int(heading.name[1]),
                "text": text
            })

    # Paragraphs, list items and quotations
    content_blocks = []
    for element in main.find_all(["p", "li", "blockquote"]):
        text = get_text(element)
        if text and text not in content_blocks:
            content_blocks.append(text)

    # Tables, preserving rows and cell values
    tables = []
    for table in main.find_all("table"):
        rows = []

        for tr in table.find_all("tr"):
            cells = [
                get_text(cell)
                for cell in tr.find_all(["th", "td"], recursive=False)
            ]

            if cells:
                rows.append(cells)

        if rows:
            tables.append({"rows": rows})

    # Images, including lazy-loaded image URLs
    images = []
    for img in main.find_all("img"):
        src = (
            img.get("src")
            or img.get("data-src")
            or img.get("data-lazy-src")
            or img.get("data-original")
            or ""
        )

        caption = ""
        figure = img.find_parent("figure")
        if figure and figure.find("figcaption"):
            caption = get_text(figure.find("figcaption"))

        images.append({
            "url": urljoin(url, src) if src else "",
            "alt": img.get("alt", "").strip(),
            "title": img.get("title", "").strip(),
            "caption": caption
        })

    # Links
    links = []
    seen_links = set()

    for a in main.find_all("a", href=True):
        link_url = clean_url(urljoin(url, a["href"]))

        if link_url and link_url not in seen_links:
            seen_links.add(link_url)
            links.append({
                "text": get_text(a),
                "url": link_url,
                "title": a.get("title", "").strip()
            })

    return {
        "url": url,
        "final_url": response.url,
        "canonical_url": canonical,
        "http_status": response.status_code,
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "title": title,
        "meta_description": description,
        "headings": headings,
        "paragraphs_and_lists": content_blocks,
        "tables": tables,
        "images": images,
        "links": links,
        "main_text": get_text(main)
    }


def crawl(start_url, max_pages=1, delay=1.0):
    """Scrape one URL or follow same-domain links up to max_pages."""
    start_url = clean_url(start_url)

    if not start_url:
        raise ValueError("Enter a valid URL starting with http:// or https://")

    domain = urlparse(start_url).netloc
    session = requests.Session()
    session.headers.update({
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml"
    })

    robots = get_robots_parser(start_url, session)

    queue = deque([start_url])
    queued = {start_url}
    pages = []
    errors = []

    while queue and len(pages) < max_pages:
        url = queue.popleft()

        if urlparse(url).netloc != domain:
            continue

        if urlparse(url).path.lower().endswith(SKIP_EXTENSIONS):
            continue

        if not robots.can_fetch(USER_AGENT, url):
            errors.append({
                "url": url,
                "error": "Disallowed by robots.txt"
            })
            continue

        try:
            response = session.get(
                url, timeout=TIMEOUT, allow_redirects=True
            )
            response.raise_for_status()

            content_type = response.headers.get(
                "Content-Type", ""
            ).lower()

            if "html" not in content_type:
                errors.append({
                    "url": url,
                    "error": "Not an HTML page"
                })
                continue

            page = extract_page(url, response)
            pages.append(page)

            # Discover internal links for optional spider mode.
            for link in page["links"]:
                next_url = clean_url(link["url"])

                if not next_url or next_url in queued:
                    continue

                if urlparse(next_url).netloc != domain:
                    continue

                if urlparse(next_url).path.lower().endswith(
                    SKIP_EXTENSIONS
                ):
                    continue

                queued.add(next_url)
                queue.append(next_url)

            print(f"Scraped ({len(pages)}/{max_pages}): {url}")

        except requests.RequestException as error:
            errors.append({
                "url": url,
                "error": str(error)
            })
            print(f"Failed: {url} - {error}")

        time.sleep(max(0, delay))

    return {
        "site": f"{urlparse(start_url).scheme}://{domain}",
        "start_url": start_url,
        "scraped_at_utc": datetime.now(timezone.utc).isoformat(),
        "pages_scraped": len(pages),
        "errors": errors,
        "pages": pages
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extract website content into JSON."
    )
    parser.add_argument(
        "url",
        help="URL to scrape, e.g. https://careermudhra.com/services/"
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=1,
        help="Maximum pages to scrape; defaults to 1"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Seconds between requests; defaults to 1"
    )
    parser.add_argument(
        "--output",
        default="scraped_pages.json",
        help="JSON output filename"
    )

    args = parser.parse_args()

    if args.max_pages < 1:
        parser.error("--max-pages must be at least 1")

    result = crawl(
        args.url,
        max_pages=args.max_pages,
        delay=args.delay
    )

    with open(args.output, "w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)

    print(f"\nFinished. Pages scraped: {result['pages_scraped']}")
    print(f"JSON saved to: {args.output}")
    print(f"Errors: {len(result['errors'])}")


if __name__ == "__main__":
    main()
