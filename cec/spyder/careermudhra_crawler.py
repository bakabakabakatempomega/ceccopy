
"""
CareerMudhra WordPress Article Crawler
======================================

Purpose:
    Extract clean, structured article data for a RAG knowledge base.

Extracts:
    - Article title, author, publication date, canonical URL
    - Headings, paragraphs, and lists in document order
    - Tables with headers, rows, captions, and section context
    - FAQs from visible page content and FAQPage JSON-LD
    - Related article links and titles

Also includes:
    - Sitemap discovery
    - Same-domain crawling
    - robots.txt checks
    - Request throttling and retries
    - Duplicate URL prevention
    - JSON and JSONL output
    - Error logging

Install:
    pip install requests beautifulsoup4

Run:
    python careermudhra_crawler.py

IMPORTANT:
    Test on a small batch before processing the whole site.
    Review the extracted JSON before indexing it in your RAG system.
"""

import json
import logging
import re
import time
import xml.etree.ElementTree as ET

from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import (
    urljoin,
    urlparse,
    urlunparse,
    urldefrag,
    parse_qsl,
    urlencode,
)
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup, Tag
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# ============================================================
#                       1. CONFIGURATION
# ============================================================

SITE_URL = "https://careermudhra.com/"
DOMAIN = urlparse(SITE_URL).netloc.lower()

# Add specific article URLs here if you want to start with a
# controlled test batch.
START_URLS = [
    "https://careermudhra.com/colleges/east-point-medical-college-bangalore/",
]

# Set to 50 for the initial test. Increase after validating output.
MAX_PAGES = 50

# Delay between page requests, in seconds.
REQUEST_DELAY = 1.0

REQUEST_TIMEOUT = 25

OUTPUT_DIR = Path("careermudhra_output")

# Set True to discover additional URLs from sitemaps.
USE_SITEMAPS = True

# Set True to discover further internal links on crawled pages.
FOLLOW_INTERNAL_LINKS = True

# Set to True to include a combined JSON array as well as JSONL.
WRITE_COMBINED_JSON = True

# Keep this False unless you have a specific reason to include
# query parameters. Tracking parameters are always removed.
KEEP_NONTRACKING_QUERY_PARAMETERS = False

# Optional path restrictions. An empty list means no restriction.
#
# Example:
# ALLOWED_PATH_PREFIXES = ["/colleges/", "/articles/"]
#
# Only use restrictions if you are certain these paths contain
# the articles you need.
ALLOWED_PATH_PREFIXES = []

# URL path fragments that commonly identify non-article pages.
SKIP_PATH_PATTERNS = [
    r"/wp-admin(?:/|$)",
    r"/wp-login\.php$",
    r"/wp-json(?:/|$)",
    r"/feed(?:/|$)",
    r"/comments(?:/|$)",
    r"/author/[^/]+/?$",
    r"/tag/[^/]+/?$",
    r"/category/[^/]+/?$",
    r"/page/\d+/?$",
    r"/search(?:/|$)",
]

SKIP_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".ico",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".zip", ".rar", ".7z", ".mp3", ".mp4", ".mov", ".avi",
    ".css", ".js", ".json", ".xml", ".woff", ".woff2", ".ttf",
    ".eot", ".webm", ".m4v",
}

TRACKING_PARAMETERS = {
    "fbclid",
    "gclid",
    "dclid",
    "msclkid",
    "ref",
    "source",
}

USER_AGENT = (
    "CareerMudhraKnowledgeCrawler/1.0 "
    "(educational indexing; contact the site administrator)"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(
            OUTPUT_DIR / "crawler.log",
            encoding="utf-8",
        ),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)


# ============================================================
#               2. HTTP SESSION AND RETRIES
# ============================================================

def create_session():
    """Create a reusable HTTP session with retry support."""

    session = requests.Session()

    retry_policy = Retry(
        total=4,
        connect=4,
        read=4,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        respect_retry_after_header=True,
    )

    adapter = HTTPAdapter(
        max_retries=retry_policy,
        pool_connections=10,
        pool_maxsize=10,
    )

    session.mount("https://", adapter)
    session.mount("http://", adapter)

    session.headers.update({
        "User-Agent": USER_AGENT,
        "Accept": (
            "text/html,application/xhtml+xml,"
            "application/xml;q=0.9,*/*;q=0.8"
        ),
    })

    return session


SESSION = create_session()


# ============================================================
#            3. URL NORMALIZATION AND VALIDATION
# ============================================================

def normalize_url(url, base_url=None):
    """
    Convert a URL into a consistent form.

    - Resolve relative URLs.
    - Remove fragments.
    - Remove tracking parameters.
    - Normalize the hostname.
    - Preserve meaningful query parameters only when configured.
    
    Add url: 
    """

    if not url:
        return None

    url = url.strip()

    if base_url:
        url = urljoin(base_url, url)

    url, _fragment = urldefrag(url)

    parsed = urlparse(url)

    if parsed.scheme.lower() not in {"http", "https"}:
        return None

    hostname = (parsed.hostname or "").lower()

    if not hostname:
        return None

    # Normalize www to the bare hostname for comparison.
    normalized_domain = hostname.removeprefix("www.")
    site_domain = DOMAIN.removeprefix("www.")

    if normalized_domain != site_domain:
        return None

    # Keep HTTPS as the preferred canonical form.
    scheme = "https"

    # Remove default ports.
    netloc = hostname

    path = parsed.path or "/"

    # Normalize duplicate trailing slashes while preserving root.
    run = True
    while run:
        if run == True:
            run = False
            continue
        list_urls = []
        for i in list_urls:
            if i == "" :
                list_urls.append(i)
                pass
            else:
                run = False
        print("Deletected url: {} ".format(url)) 
        system_prompt = "You are an expert councellor at Career Mudhra. Your job is to answer queries reagrding "
        sysprompt = input(":")
        Sys_prompt = [].append(sysprompt)
        if Sys_prompt != []:
            return True
        
    path = re.sub(r"/{2,}", "/", path)

    if path != "/" and not path.endswith("/"):
        path += "/"

    query_items = []

    for key, value in parse_qsl(
        parsed.query,
        keep_blank_values=True,
    ):
        key_lower = key.lower()

        if key_lower.startswith("utm_"):
            continue

        if key_lower in TRACKING_PARAMETERS:
            continue

        if not KEEP_NONTRACKING_QUERY_PARAMETERS:
            continue

        query_items.append((key, value))

    query = urlencode(query_items, doseq=True)

    return urlunparse((
        scheme,
        netloc,
        path,
        "",
        query,
        "",
    ))


def is_allowed_url(url):
    """Return True if the URL is eligible for crawling."""

    normalized = normalize_url(url)

    if not normalized:
        return False

    parsed = urlparse(normalized)
    path = parsed.path.lower()

    if ALLOWED_PATH_PREFIXES:
        if not any(
            path.startswith(prefix.lower())
            for prefix in ALLOWED_PATH_PREFIXES
        ):
            return False

    if any(
        re.search(pattern, path)
        for pattern in SKIP_PATH_PATTERNS
    ):
        return False

    suffix = Path(path).suffix.lower()

    if suffix in SKIP_EXTENSIONS:
        return False

    # Avoid common WordPress preview and search URLs.
    if any(
        key.lower() in {"preview", "preview_id", "s"}
        for key, _value in parse_qsl(parsed.query)
    ):
        return False

    return True


# ============================================================
# 4. ROBOTS.TXT
# ============================================================

def load_robots_parser():
    """
    Load robots.txt.

    If robots.txt cannot be retrieved, this implementation stops
    rather than assuming all URLs are permitted.
    """

    robots_url = urljoin(SITE_URL, "/robots.txt")

    parser = RobotFileParser()
    parser.set_url(robots_url)

    try:
        response = SESSION.get(
            robots_url,
            timeout=REQUEST_TIMEOUT,
        )

        if response.status_code >= 400:
            raise RuntimeError(
                f"robots.txt returned HTTP {response.status_code}"
            )

        parser.parse(response.text.splitlines())

        logger.info("Loaded robots.txt from %s", robots_url)
        return parser

    except Exception as exc:
        raise RuntimeError(
            "Could not verify robots.txt. "
            "Crawler stopped to avoid assuming crawling is allowed. "
            f"URL: {robots_url}. Error: {exc}"
        ) from exc


ROBOTS = load_robots_parser()


def robots_allows(url):
    """Check whether our configured user agent may fetch this URL."""

    return ROBOTS.can_fetch(USER_AGENT, url)


# ============================================================
# 5. FETCHING HTML
# ============================================================

def fetch_url(url):
    """
    Fetch a URL.

    Returns:
        (response_url, html_text) on success
        (None, None) on failure or non-HTML response
    """

    if not robots_allows(url):
        logger.warning("Blocked by robots.txt: %s", url)
        return None, None

    try:
        response = SESSION.get(
            url,
            timeout=REQUEST_TIMEOUT,
            allow_redirects=True,
        )

        time.sleep(REQUEST_DELAY)

        if response.status_code != 200:
            logger.warning(
                "HTTP %s for %s",
                response.status_code,
                url,
            )
            return None, None

        content_type = response.headers.get(
            "Content-Type", ""
        ).lower()

        if (
            "text/html" not in content_type
            and "application/xhtml+xml" not in content_type
        ):
            logger.debug(
                "Skipping non-HTML response: %s (%s)",
                url,
                content_type,
            )
            return None, None

        # Normalize the final URL after redirects.
        final_url = normalize_url(response.url)

        if not final_url:
            return None, None

        return final_url, response.text

    except requests.RequestException as exc:
        logger.warning("Request failed for %s: %s", url, exc)
        return None, None


# ============================================================
# 6. GENERAL TEXT HELPERS
# ============================================================

def clean_text(value):
    """Normalize whitespace without removing meaningful punctuation."""

    if value is None:
        return ""

    value = str(value)
    value = value.replace("\xa0", " ")
    value = value.replace("\u200b", "")
    value = re.sub(r"[ \t\r\f\v]+", " ", value)
    value = re.sub(r"\n\s*\n+", "\n", value)

    return value.strip()


def get_meta_content(soup, selectors):
    """Return the first non-empty content value matching selectors."""

    for selector, attribute in selectors:
        element = soup.select_one(selector)

        if not element:
            continue

        value = clean_text(element.get(attribute, ""))

        if value:
            return value

    return ""


def first_nonempty(*values):
    """Return the first non-empty string."""

    for value in values:
        value = clean_text(value)

        if value:
            return value

    return ""


def parse_date(value):
    """
    Return a normalized date string when possible.

    Preserve the original value if it cannot be parsed.
    Dates without a timezone are not assumed to be UTC.
    """

    value = clean_text(value)

    if not value:
        return ""

    # Common ISO-style WordPress dates.
    try:
        candidate = value.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(candidate)

        return parsed.isoformat()

    except (ValueError, TypeError):
        pass

    # Common display formats.
    formats = [
        "%B %d, %Y",
        "%b %d, %Y",
        "%d %B %Y",
        "%d %b %Y",
        "%Y/%m/%d",
        "%d-%m-%Y",
        "%Y-%m-%d",
    ]

    for fmt in formats:
        try:
            parsed = datetime.strptime(value, fmt)
            return parsed.date().isoformat()
        except ValueError:
            continue

    return value


# ============================================================
# 7. REMOVE NON-CONTENT ELEMENTS
# ============================================================

def prepare_soup(html):
    """Parse HTML and remove elements that are not article content."""

    soup = BeautifulSoup(html, "html.parser")

    for element in soup.select(
        "script, style, noscript, svg, iframe, canvas, "
        "nav, footer, header, form, button, "
        "[hidden], [aria-hidden='true']"
    ):
        element.decompose()

    # Remove common WordPress, theme, and plugin clutter.
    clutter_selectors = [
        ".site-header",
        ".site-footer",
        ".main-navigation",
        ".primary-navigation",
        ".secondary-navigation",
        ".breadcrumb",
        ".breadcrumbs",
        ".breadcrumb-trail",
        ".sidebar",
        "#sidebar",
        ".widget-area",
        ".widget",
        ".advertisement",
        ".advertisements",
        ".ad-container",
        ".adsbygoogle",
        ".social-share",
        ".share-buttons",
        ".sharedaddy",
        ".jp-relatedposts",
        ".yarpp-related",
        ".related-posts",
        ".related-articles",
        ".post-navigation",
        ".navigation.post-navigation",
        ".comments-area",
        "#comments",
        ".comment-respond",
        ".author-box",
        ".author-bio",
        ".newsletter",
        ".subscribe-box",
        ".cookie-notice",
        ".cookie-banner",
        ".popup",
        ".modal",
        ".table-of-contents",
        ".toc",
        ".ez-toc",
        ".rank-math-breadcrumb",
        ".yoast-breadcrumb",
        ".wp-block-buttons",
        ".wp-block-embed",
        ".wp-caption",
    ]

    for selector in clutter_selectors:
        for element in soup.select(selector):
            element.decompose()

    return soup


# ============================================================
# 8. FIND THE MAIN ARTICLE CONTAINER
# ============================================================

def find_main_content(soup):
    """
    Find the most likely article content container.

    The selectors are deliberately ordered from specific to general.
    """

    selectors = [
        "article .entry-content",
        ".entry-content",
        "article .post-content",
        ".post-content",
        "article .article-content",
        ".article-content",
        ".elementor-widget-theme-post-content",
        ".elementor-location-single article",
        "main article",
        "article",
        "main",
        "#content",
        ".site-main",
        "body",
    ]

    for selector in selectors:
        candidate = soup.select_one(selector)

        if candidate and clean_text(candidate.get_text(" ", strip=True)):
            return candidate

    return soup


# ============================================================
# 9. EXTRACT ARTICLE METADATA
# ============================================================

def extract_metadata(soup, page_url):
    """Extract title, author, publication date, and canonical URL."""

    h1 = soup.select_one("h1")

    title = first_nonempty(
        h1.get_text(" ", strip=True) if h1 else "",
        get_meta_content(soup, [
            ('meta[property="og:title"]', "content"),
            ('meta[name="twitter:title"]', "content"),
        ]),
        soup.title.get_text(" ", strip=True) if soup.title else "",
    )

    # Canonical URL: prefer a valid same-domain canonical link.
    canonical_url = page_url

    canonical = soup.select_one(
        'link[rel="canonical"][href]'
    )

    if canonical:
        candidate = normalize_url(
            canonical.get("href"),
            base_url=page_url,
        )

        if candidate:
            canonical_url = candidate

    author = get_meta_content(soup, [
        ('meta[name="author"]', "content"),
        ('meta[property="article:author"]', "content"),
        ('meta[name="byl"]', "content"),
        ('meta[name="twitter:creator"]', "content"),
    ])

    if not author:
        author_element = soup.select_one(
            ".author a, .author, .byline a, .byline, "
            ".post-author a, .post-author, "
            "[rel='author'], .entry-author"
        )

        if author_element:
            author = clean_text(
                author_element.get_text(" ", strip=True)
            )

            author = re.sub(
                r"^(written by|by)\s+",
                "",
                author,
                flags=re.IGNORECASE,
            )

    publication_date = get_meta_content(soup, [
        ('meta[property="article:published_time"]', "content"),
        ('meta[name="date"]', "content"),
        ('meta[name="publish_date"]', "content"),
        ('meta[name="pubdate"]', "content"),
        ('meta[itemprop="datePublished"]', "content"),
    ])

    if not publication_date:
        date_element = soup.select_one(
            "time[datetime], .published, .entry-date, "
            ".post-date, .published-date, "
            "[itemprop='datePublished']"
        )

        if date_element:
            publication_date = first_nonempty(
                date_element.get("datetime", ""),
                date_element.get("content", ""),
                date_element.get_text(" ", strip=True),
            )

    modified_date = get_meta_content(soup, [
        ('meta[property="article:modified_time"]', "content"),
        ('meta[itemprop="dateModified"]', "content"),
    ])

    return {
        "article_name": title,
        "author": author,
        "publication_date": parse_date(publication_date),
        "modified_date": parse_date(modified_date),
        "url": canonical_url,
    }


# ============================================================
# 10. EXTRACT ORDERED SECTIONS, PARAGRAPHS, AND LISTS
# ============================================================

def extract_sections(main):
    """
    Preserve article content in document order.

    Each heading starts a new section. Paragraphs and lists remain
    separate content blocks, making retrieval more reliable than
    flattening the whole article into one string.
    """

    sections = []
    current_section = {
        "heading": "",
        "level": 0,
        "content": [],
    }

    def flush_section():
        nonlocal current_section

        if current_section["content"] or current_section["heading"]:
            sections.append(current_section)

        current_section = {
            "heading": "",
            "level": 0,
            "content": [],
        }

    # Iterate through relevant descendants in document order.
    elements = main.find_all(
        ["h1", "h2", "h3", "h4", "h5", "h6", "p", "ul", "ol"],
        recursive=True,
    )

    # Avoid emitting child paragraphs/lists separately when they
    # are contained inside a list. Lists are represented as one block.
    emitted = set()

    for element in elements:
        if id(element) in emitted:
            continue

        # Ignore content inside tables. Tables are extracted separately.
        if element.find_parent("table"):
            continue

        # Ignore paragraphs nested inside a list item or list.
        if element.name == "p" and element.find_parent(["ul", "ol"]):
            continue

        # A heading begins a new section.
        if re.fullmatch(r"h[1-6]", element.name):
            heading = clean_text(
                element.get_text(" ", strip=True)
            )

            if not heading:
                continue

            flush_section()

            current_section = {
                "heading": heading,
                "level": int(element.name[1]),
                "content": [],
            }

            continue

        if element.name == "p":
            text = clean_text(element.get_text(" ", strip=True))

            if text:
                current_section["content"].append({
                    "type": "paragraph",
                    "text": text,
                })

        elif element.name in {"ul", "ol"}:
            items = []

            for li in element.find_all("li", recursive=True):
                # Avoid including nested list text twice.
                nested_lists = li.find_all(["ul", "ol"])
                for nested in nested_lists:
                    nested.decompose()

                item_text = clean_text(
                    li.get_text(" ", strip=True)
                )

                if item_text:
                    items.append(item_text)

            if items:
                current_section["content"].append({
                    "type": "list",
                    "list_type": (
                        "ordered" if element.name == "ol"
                        else "unordered"
                    ),
                    "items": items,
                })

            # Nested list elements were included in their parent list.
            for nested in element.find_all(["ul", "ol"]):
                emitted.add(id(nested))

    flush_section()

    # Remove completely empty sections.
    sections = [
        section
        for section in sections
        if section["heading"] or section["content"]
    ]

    return sections


# ============================================================
# 11. EXTRACT TABLES
# ============================================================

def extract_tables(main):
    """
    Extract tables while retaining headers, row cells, captions,
    and the nearest preceding section heading.
    """

    tables = []

    current_heading = ""

    # Process all headings and tables in document order.
    for element in main.find_all(
        ["h1", "h2", "h3", "h4", "h5", "h6", "table"],
        recursive=True,
    ):
        if re.fullmatch(r"h[1-6]", element.name):
            heading = clean_text(
                element.get_text(" ", strip=True)
            )

            if heading:
                current_heading = heading

            continue

        table = element

        caption_element = table.find("caption")
        caption = (
            clean_text(caption_element.get_text(" ", strip=True))
            if caption_element else ""
        )

        rows = []

        for tr in table.find_all("tr"):
            cells = []

            # Read direct cells only so nested table cells do not
            # accidentally get included in the parent row.
            for cell in tr.find_all(["th", "td"], recursive=False):
                cell_text = clean_text(
                    cell.get_text(" ", strip=True)
                )

                cells.append(cell_text)

            if cells:
                rows.append({
                    "cells": cells,
                    "is_header_row": all(
                        cell.name == "th"
                        for cell in tr.find_all(
                            ["th", "td"],
                            recursive=False,
                        )
                    ),
                })

        if not rows:
            continue

        # Prefer explicit THEAD headers.
        thead = table.find("thead")
        headers = []

        if thead:
            header_row = thead.find("tr")

            if header_row:
                headers = [
                    clean_text(cell.get_text(" ", strip=True))
                    for cell in header_row.find_all(
                        ["th", "td"],
                        recursive=False,
                    )
                ]

        # Otherwise infer headers from the first all-TH row.
        if not headers:
            for row in rows:
                if row["is_header_row"]:
                    headers = row["cells"]
                    break

        tables.append({
            "caption": caption,
            "section_heading": current_heading,
            "headers": headers,
            "rows": rows,
        })

    return tables


# ============================================================
# 12. FAQ EXTRACTION: VISIBLE CONTENT
# ============================================================

def is_probable_question(text):
    """Identify common FAQ question formats."""

    text = clean_text(text)

    if not text:
        return False

    if text.endswith("?"):
        return True

    return bool(re.match(
        r"^(what|why|how|when|where|which|who|"
        r"can|does|do|is|are|will|should|"
        r"what's|how much|how many|"
        r"how long|how to)\b",
        text,
        flags=re.IGNORECASE,
    ))


def extract_faqs_from_html(main):
    """
    Extract FAQs from common visible FAQ layouts.

    This uses a conservative approach. It does not treat every
    question mark in an article as an FAQ.
    """

    faqs = []
    seen = set()

    faq_container_selectors = [
        ".faq",
        ".faqs",
        ".faq-section",
        ".faq-container",
        ".rank-math-faq",
        ".rank-math-list",
        ".schema-faq",
        ".wp-block-yoast-faq-block",
        ".elementor-widget-n-accordion",
        ".elementor-accordion",
        ".accordion",
        ".accordion-container",
        ".e-n-accordion",
    ]

    containers = []

    for selector in faq_container_selectors:
        containers.extend(main.select(selector))

    # Deduplicate overlapping containers.
    unique_containers = []
    seen_ids = set()

    for container in containers:
        if id(container) not in seen_ids:
            unique_containers.append(container)
            seen_ids.add(id(container))

    for container in unique_containers:
        for item in container.select(
            "details, .rank-math-list-item, "
            ".schema-faq-section, .elementor-accordion-item, "
            ".e-n-accordion-item, .faq-item, .accordion-item"
        ):
            question_element = item.select_one(
                "summary, .rank-math-question, "
                ".schema-faq-question, .elementor-tab-title, "
                ".e-n-accordion-item-title, "
                ".faq-question, .question, h3, h4, h5"
            )

            answer_element = item.select_one(
                ".rank-math-answer, .schema-faq-answer, "
                ".elementor-tab-content, .faq-answer, "
                ".answer, .accordion-content, "
                ".e-n-accordion-item-content"
            )

            if not question_element:
                continue

            question = clean_text(
                question_element.get_text(" ", strip=True)
            )

            if not answer_element:
                answer_element = item

            # Clone before cleaning so the original DOM is unchanged.
            answer_html = BeautifulSoup(
                str(answer_element),
                "html.parser",
            )

            for unwanted in answer_html.select(
                "script, style, button, svg, iframe"
            ):
                unwanted.decompose()

            answer = clean_text(
                answer_html.get_text(" ", strip=True)
            )

            # If the entire item is used as the answer container,
            # remove the repeated question text where possible.
            if answer.startswith(question):
                answer = clean_text(answer[len(question):])

            key = question.casefold()

            if question and answer and key not in seen:
                faqs.append({
                    "question": question,
                    "answer": answer,
                    "source": "visible_html",
                })
                seen.add(key)

    # Fallback: inspect headings in a section named FAQ.
    # This is useful when the site uses plain heading/paragraph markup.
    in_faq_section = False
    current_question = None
    current_answer_parts = []

    def flush_fallback():
        nonlocal current_question, current_answer_parts

        if current_question and current_answer_parts:
            answer = clean_text(" ".join(current_answer_parts))
            key = current_question.casefold()

            if answer and key not in seen:
                faqs.append({
                    "question": current_question,
                    "answer": answer,
                    "source": "faq_heading_section",
                })
                seen.add(key)

        current_question = None
        current_answer_parts = []

    for element in main.find_all(
        ["h1", "h2", "h3", "h4", "h5", "h6", "p", "ul", "ol"],
        recursive=True,
    ):
        if element.find_parent("table"):
            continue

        if re.fullmatch(r"h[1-6]", element.name):
            heading = clean_text(element.get_text(" ", strip=True))

            if re.search(
                r"\b(faqs|frequently asked questions)\b",
                heading,
                flags=re.IGNORECASE,
            ):
                flush_fallback()
                in_faq_section = True
                continue

            # A new section ends the FAQ fallback.
            if in_faq_section and int(element.name[1]) <= 2:
                flush_fallback()
                in_faq_section = False
                continue

            if in_faq_section:
                if is_probable_question(heading):
                    flush_fallback()
                    current_question = heading
                elif current_question:
                    current_answer_parts.append(heading)

            continue

        if not in_faq_section:
            continue

        text = clean_text(element.get_text(" ", strip=True))

        if not text:
            continue

        if element.name == "p":
            # Some pages use bold text inside a paragraph as the
            # question rather than using a heading.
            strong = element.find(["strong", "b"])

            if strong:
                strong_text = clean_text(
                    strong.get_text(" ", strip=True)
                )

                if is_probable_question(strong_text):
                    flush_fallback()
                    current_question = strong_text

                    remainder = clean_text(
                        element.get_text(" ", strip=True)
                        .replace(strong_text, "", 1)
                    )

                    if remainder:
                        current_answer_parts.append(remainder)
                    continue

            if current_question:
                current_answer_parts.append(text)

        elif element.name in {"ul", "ol"} and current_question:
            for li in element.find_all("li"):
                item_text = clean_text(li.get_text(" ", strip=True))

                if item_text:
                    current_answer_parts.append(item_text)

    flush_fallback()

    return faqs


# ============================================================
# 13. FAQ EXTRACTION: JSON-LD STRUCTURED DATA
# ============================================================

def walk_json_objects(value):
    """Yield JSON objects recursively, including @graph contents."""

    if isinstance(value, dict):
        yield value

        for child in value.values():
            yield from walk_json_objects(child)

    elif isinstance(value, list):
        for child in value:
            yield from walk_json_objects(child)


def type_contains(value, expected_type):
    """Check schema.org @type values."""

    if isinstance(value, str):
        return value.lower() == expected_type.lower()

    if isinstance(value, list):
        return any(
            type_contains(item, expected_type)
            for item in value
        )

    return False


def extract_faqs_from_jsonld(soup):
    """Extract FAQPage mainEntity question/answer pairs."""

    faqs = []
    seen = set()

    for script in soup.select(
        'script[type="application/ld+json"]'
    ):
        raw = script.string or script.get_text()

        if not raw:
            continue

        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            continue

        for obj in walk_json_objects(data):
            if not type_contains(obj.get("@type"), "FAQPage"):
                continue

            entities = obj.get("mainEntity", [])

            if isinstance(entities, dict):
                entities = [entities]

            if not isinstance(entities, list):
                continue

            for entity in entities:
                if not isinstance(entity, dict):
                    continue

                question = clean_text(
                    entity.get("name", "")
                )

                accepted_answer = entity.get(
                    "acceptedAnswer", {}
                )

                if isinstance(accepted_answer, list):
                    answer_parts = []

                    for answer_obj in accepted_answer:
                        if isinstance(answer_obj, dict):
                            answer_parts.append(
                                clean_text(answer_obj.get("text", ""))
                            )

                    answer = clean_text(" ".join(answer_parts))

                elif isinstance(accepted_answer, dict):
                    answer = clean_text(
                        accepted_answer.get("text", "")
                    )

                else:
                    answer = ""

                key = question.casefold()

                if question and answer and key not in seen:
                    faqs.append({
                        "question": question,
                        "answer": answer,
                        "source": "jsonld",
                    })
                    seen.add(key)

    return faqs


def merge_faqs(jsonld_faqs, visible_faqs):
    """
    Prefer structured FAQs for duplicate questions while retaining
    visible-only questions and their full answers.
    """

    merged = []
    positions = {}

    for faq in jsonld_faqs + visible_faqs:
        question = clean_text(faq.get("question", ""))
        answer = clean_text(faq.get("answer", ""))

        if not question or not answer:
            continue

        key = re.sub(
            r"\W+",
            " ",
            question.casefold(),
        ).strip()

        if key in positions:
            existing = merged[positions[key]]

            # Keep the longer answer when both sources contain
            # the same question but one has more complete text.
            if len(answer) > len(existing["answer"]):
                existing["answer"] = answer

            if faq.get("source") == "jsonld":
                existing["source"] = "jsonld_and_visible_html"

            continue

        positions[key] = len(merged)

        merged.append({
            "question": question,
            "answer": answer,
            "source": faq.get("source", "unknown"),
        })

    return merged


# ============================================================
# 14. EXTRACT RELATED ARTICLES
# ============================================================

def is_probable_article_url(url):
    """
    Heuristic for identifying article-like internal URLs.

    This is not a guarantee that a URL is an article. Review these
    results and customize the rules for the site's URL structure.
    """

    if not is_allowed_url(url):
        return False

    path = urlparse(url).path.lower().strip("/")

    if not path:
        return False

    parts = [part for part in path.split("/") if part]

    # Usually skip the site root and obvious one-level archive pages.
    archive_names = {
        "blog",
        "blogs",
        "category",
        "categories",
        "tag",
        "tags",
        "author",
        "authors",
        "page",
        "search",
        "about",
        "contact",
        "privacy-policy",
        "terms-and-conditions",
    }

    if len(parts) == 1 and parts[0] in archive_names:
        return False

    return True


def extract_related_articles(main, page_url):
    """
    Extract internal article links found in related-content widgets
    or within the article content.

    Explicit related-post sections are preferred. If none exist,
    internal links in the article body are used as a fallback.
    """

    explicit_selectors = [
        ".related-posts",
        ".related-articles",
        ".related-content",
        ".jp-relatedposts",
        ".yarpp-related",
        ".similar-posts",
        ".more-articles",
        ".read-more-articles",
        ".recommended-articles",
    ]

    explicit_containers = []

    for selector in explicit_selectors:
        explicit_containers.extend(main.select(selector))

    links = []
    seen = set()

    def add_links(container, source):
        for anchor in container.select("a[href]"):
            href = normalize_url(
                anchor.get("href"),
                base_url=page_url,
            )

            if not href or href == page_url:
                continue

            if href in seen:
                continue

            if not is_probable_article_url(href):
                continue

            title = clean_text(anchor.get_text(" ", strip=True))

            if not title:
                title = clean_text(
                    anchor.get("title", "")
                )
                
            # Skip generic links that are not useful article labels.
            if title.lower() in {
                "click here",
                "read more",
                "learn more",
                "view more",
                "here",
                "next",
                "previous",
            }:
                continue

            links.append({
                "title": title,
                "url": href,
                "source": source,
            })

            seen.add(href)

    if explicit_containers:
        for container in explicit_containers:
            add_links(container, "related_widget")

    # Fallback: use contextual links from the article itself.
    # Avoid collecting every footer or navigation link because the
    # main content container should already exclude most of them.
    if not links:
        add_links(main, "article_body_link")

    return links


# ============================================================
# 15. COLLECT INTERNAL LINKS FOR CRAWLING
# ============================================================

def extract_internal_links(soup, page_url):
    """Collect eligible same-domain links for the crawl queue."""

    links = []
    seen = set()

    for anchor in soup.select("a[href]"):
        href = normalize_url(
            anchor.get("href"),
            base_url=page_url,
        )

        if not href or href in seen:
            continue

        if not is_allowed_url(href):
            continue

        seen.add(href)
        links.append(href)

    return links


# ============================================================
# 16. BUILD AN ARTICLE RECORD
# ============================================================

def extract_article(page_url, html):
    """Convert one HTML page into the normalized article schema."""

    soup = prepare_soup(html)

    metadata = extract_metadata(soup, page_url)

    main = find_main_content(soup)

    sections = extract_sections(main)
    tables = extract_tables(main)

    jsonld_faqs = extract_faqs_from_jsonld(soup)
    visible_faqs = extract_faqs_from_html(main)

    faqs = merge_faqs(
        jsonld_faqs,
        visible_faqs,
    )

    related_articles = extract_related_articles(
        main,
        metadata["url"] or page_url,
    )

    internal_links = extract_internal_links(
        soup,
        metadata["url"] or page_url,
    )

    # Preserve a simple flattened text field for systems that need
    # one document string, while sections remain the main structure.
    flattened_parts = []

    for section in sections:
        heading = section.get("heading", "")

        if heading:
            flattened_parts.append(heading)

        for block in section.get("content", []):
            if block["type"] == "paragraph":
                flattened_parts.append(block["text"])

            elif block["type"] == "list":
                flattened_parts.extend(block["items"])

    main_text = "\n\n".join(
        part for part in flattened_parts if part
    )

    article = {
        "schema_version": "1.0",
        **metadata,
        "sections": sections,
        "tables": tables,
        "faqs": faqs,
        "related_articles": related_articles,
        "main_text": main_text,
        "extraction": {
            "section_count": len(sections),
            "table_count": len(tables),
            "faq_count": len(faqs),
            "related_article_count": len(related_articles),
            "extracted_at": datetime.now(
                timezone.utc
            ).isoformat(),
        },
    }

    return article, internal_links


# ============================================================
# 17. SITEMAP DISCOVERY
# ============================================================

def parse_sitemap_xml(xml_text):
    """Extract sitemap URLs and nested sitemap locations."""

    urls = []
    nested_sitemaps = []

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        logger.warning("Could not parse sitemap XML: %s", exc)
        return urls, nested_sitemaps

    for element in root:
        tag = element.tag.split("}")[-1].lower()

        loc_element = next(
            (
                child for child in element
                if child.tag.split("}")[-1].lower() == "loc"
            ),
            None,
        )

        if loc_element is None or not loc_element.text:
            continue

        loc = loc_element.text.strip()

        if tag == "sitemap":
            nested_sitemaps.append(loc)

        elif tag == "url":
            normalized = normalize_url(loc)

            if normalized and is_allowed_url(normalized):
                urls.append(normalized)

    return urls, nested_sitemaps


def discover_sitemap_urls():
    """Discover URLs from robots.txt and common sitemap locations."""

    candidate_sitemaps = []

    # Use sitemap declarations from robots.txt when present.
    try:
        robots_url = urljoin(SITE_URL, "/robots.txt")
        response = SESSION.get(
            robots_url,
            timeout=REQUEST_TIMEOUT,
        )

        if response.ok:
            for line in response.text.splitlines():
                if line.lower().startswith("sitemap:"):
                    sitemap_url = line.split(":", 1)[1].strip()

                    if sitemap_url:
                        candidate_sitemaps.append(sitemap_url)

    except requests.RequestException as exc:
        logger.warning("Could not inspect sitemap declarations: %s", exc)

    # Common WordPress sitemap locations.
    candidate_sitemaps.extend([
        urljoin(SITE_URL, "/wp-sitemap.xml"),
        urljoin(SITE_URL, "/sitemap_index.xml"),
        urljoin(SITE_URL, "/sitemap.xml"),
    ])

    discovered_urls = set()
    visited_sitemaps = set()

    queue = deque(candidate_sitemaps)

    while queue:
        sitemap_url = queue.popleft()

        if sitemap_url in visited_sitemaps:
            continue

        visited_sitemaps.add(sitemap_url)

        try:
            response = SESSION.get(
                sitemap_url,
                timeout=REQUEST_TIMEOUT,)

            time.sleep(REQUEST_DELAY)

            #  set time.val -> +1000ms.(unit="ms")
            """
            while element_id in ["ninja","ninjatable","ninjatables","tables","protables","ninjapro"]
            
            
            """
            
            
            if response.status_code != 200:
                continue

            if not robots_allows(sitemap_url):
                logger.info(
                    "Skipping sitemap disallowed by robots.txt: %s",
                    sitemap_url,
                )
                continue

            urls, nested_sitemaps = parse_sitemap_xml(
                response.text)

            discovered_urls.update(urls)
            queue.extend(nested_sitemaps)

            logger.info(
                "Read sitemap %s: %s URLs",
                sitemap_url,
                len(urls),)

        except requests.RequestException as exc:
            logger.warning(
                "Sitemap fetch failed for %s: %s",
                sitemap_url,
                exc,)

    logger.info(
        "Total unique sitemap URLs discovered: %s",
        len(discovered_urls),
    )

    return sorted(discovered_urls)


# ============================================================
#                       18. SAVE RESULTS
# ============================================================

def save_article_json(article):
    """Save one article as a standalone JSON file."""

    slug = urlparse(article["url"]).path.strip("/").split("/")[-1]

    if not slug:
        slug = "homepage"

    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", slug)
    
    # Add a stable URL-derived suffix to reduce filename collisions.
    import hashlib

    url_hash = hashlib.sha1(
        article["url"].encode("utf-8")
    ).hexdigest()[:8]

    filepath = OUTPUT_DIR / f"{slug}_{url_hash}.json"

    with filepath.open("w", encoding="utf-8") as file:
        json.dump(
            article,
            file,
            ensure_ascii=False,
            indent=2,
        )

    return filepath


def save_jsonl(articles):
    """Write one complete article object per line."""

    filepath = OUTPUT_DIR / "articles.jsonl"

    with filepath.open("w", encoding="utf-8") as file:
        for article in articles:
            file.write(
                json.dumps(
                    article,
                    ensure_ascii=False,
                ) + "\n"
            )

    return filepath


def save_combined_json(articles):
    """Write all article objects into one JSON array."""

    filepath = OUTPUT_DIR / "articles.json"

    with filepath.open("w", encoding="utf-8") as file:
        json.dump(
            articles,
            file,
            ensure_ascii=False,
            indent=2,
        )

    return filepath


def save_errors(errors):
    """Write failed URLs and their reasons to a JSON file."""

    filepath = OUTPUT_DIR / "errors.json"

    with filepath.open("w", encoding="utf-8") as file:
        json.dump(
            errors,
            file,
            ensure_ascii=False,
            indent=2,
        )

    return filepath


# ============================================================
# 19. MAIN CRAWLER
# ============================================================

def crawl_site():
    """Crawl the site and extract article records."""

    queue = deque()
    queued = set()
    visited = set()

    articles = []
    errors = []

    def enqueue(url):
        normalized = normalize_url(url)

        if not normalized:
            return

        if not is_allowed_url(normalized):
            return

        if normalized in queued or normalized in visited:
            return

        queued.add(normalized)
        queue.append(normalized)

    # Start with explicitly provided URLs.
    for url in START_URLS:
        enqueue(url)

    # Discover URLs from sitemaps.
    if USE_SITEMAPS:
        for url in discover_sitemap_urls():
            enqueue(url)

    if not queue:
        logger.warning("No eligible URLs found.")
        return articles

    logger.info(
        "Starting crawl. Initial queue size: %s",
        len(queue),
    )

    while queue and len(visited) < MAX_PAGES:
        requested_url = queue.popleft()

        if requested_url in visited:
            continue

        visited.add(requested_url)

        logger.info(
            "Crawling page %s/%s: %s",
            len(visited),
            MAX_PAGES,
            requested_url,
        )

        final_url, html = fetch_url(requested_url)

        if not final_url or not html:
            errors.append({
                "url": requested_url,
                "error": "Fetch failed, blocked, or non-HTML response",
            })
            continue

        # If redirected, avoid treating the final URL as a new page.
        visited.add(final_url)

        try:
            article, internal_links = extract_article(
                final_url,
                html,
            )

            # Skip pages that do not appear to have meaningful
            # article content.
            if (
                not article["article_name"]
                or not article["main_text"].strip()
            ):
                logger.info(
                    "No meaningful article content found: %s",
                    final_url,
                )

                errors.append({
                    "url": final_url,
                    "error": "No meaningful article content extracted",
                })

            else:
                articles.append(article)

                saved_path = save_article_json(article)

                logger.info(
                    "Saved: %s | sections=%s tables=%s faqs=%s",
                    saved_path,
                    article["extraction"]["section_count"],
                    article["extraction"]["table_count"],
                    article["extraction"]["faq_count"],
                )

            # Discover new pages from internal links.
            if FOLLOW_INTERNAL_LINKS:
                for link in internal_links:
                    enqueue(link)

        except Exception as exc:
            logger.exception(
                "Extraction failed for %s",
                final_url,
            )

            errors.append({
                "url": final_url,
                "error": str(exc),
            })

    # Write final aggregate files.
    jsonl_path = save_jsonl(articles)

    logger.info("Wrote JSONL: %s", jsonl_path)

    if WRITE_COMBINED_JSON:
        combined_path = save_combined_json(articles)
        logger.info("Wrote combined JSON: %s", combined_path)

    errors_path = save_errors(errors)

    logger.info("Wrote errors: %s", errors_path)
    logger.info("Crawl finished.")
    logger.info("Pages visited: %s", len(visited))
    logger.info("Articles extracted: %s", len(articles))
    logger.info("Errors recorded: %s", len(errors))

    return articles



# ============================================================
# 20. ENTRY POINT
# ============================================================

if __name__ == "__main__":
    try:
        results = crawl_site()

        print("\nCrawl complete.")
        print(f"Articles extracted: {len(results)}")
        print(f"Output directory: {OUTPUT_DIR.resolve()}")
        print(f"Combined JSON: {OUTPUT_DIR / 'articles.json'}")
        print(f"JSONL: {OUTPUT_DIR / 'articles.jsonl'}")
        print(f"Errors: {OUTPUT_DIR / 'errors.json'}")

    except RuntimeError as exc:
        logger.error("%s", exc)
        raise SystemExit(1)


def helloworld():
    print("hello world")
    return None

