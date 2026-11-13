
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse, unquote

import requests
from bs4 import BeautifulSoup, NavigableString, Tag
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "testlist.txt"
OUTPUT_DIR = BASE_DIR / "CM_COMBINED_JSON"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/130.0.0.0 Safari/537.36"
    )
}

TIMEOUT = (10, 30)
REQUEST_DELAY_SECONDS = 1.0

# Tags that should not contribute paragraph text.
# Anchor tags are deliberately NOT excluded:
# their visible text is meaningful webpage content.
EXCLUDED_TAGS = {
    "nav",
    "header",
    "footer",
    "aside",
    "script",
    "style",
    "noscript",
    "form",
    "button",
    "svg",
    "iframe",
    "img",
    "picture",
    "video",
    "audio"
}

# Common WordPress, Elementor and webpage structures
# that do not belong to the main article.
EXCLUDED_SELECTORS = [
    "nav",
    "header",
    "footer",
    "aside",
    ".sidebar",
    ".widget-area",
    ".comments-area",
    ".comment-respond",
    ".elementor-location-header",
    ".elementor-location-footer",
    ".breadcrumb",
    ".breadcrumbs",
    ".share-buttons",
    ".social-share",
    ".related-posts",
    ".advertisement",
    ".ads",
    "[role='navigation']",
    "[aria-hidden='true']"
]

# These selectors are checked in order.
CONTENT_SELECTORS = [
    ".entry-content",
    ".wp-block-post-content",
    ".post-content",
    ".elementor-widget-theme-post-content",
    "article",
    "main",
    ".elementor",
    "body"
]

# Table-of-contents containers are excluded separately.
TOC_SELECTORS = [
    ".ez-toc",
    ".lwptoc",
    ".toc",
    ".table-of-contents",
    ".elementor-toc",
    "[class*='table-of-contents']",
    "[id*='table-of-contents']"
]

HEADING_TAGS = [
    "h1", "h2", "h3", "h4", "h5", "h6"
]

SUBTITLE_TAGS = [
    "h2", "h3", "h4", "h5", "h6"
]

CONTENT_TAGS = [
    "p",
    "li",
    "blockquote"
]


# ============================================================
# HTTP SESSION
# ============================================================

def create_session():
    """Create a reusable HTTP session with retries."""

    session = requests.Session()
    session.headers.update(HEADERS)

    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=0.7,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"])
    )

    adapter = HTTPAdapter(max_retries=retry)

    session.mount("http://", adapter)
    session.mount("https://", adapter)

    return session


# ============================================================
# COMMON FUNCTIONS
# ============================================================

def clean_text(element):
    """Extract normalized readable text from an element."""

    if element is None:
        return ""

    return " ".join(
        element.get_text(" ", strip=True).split()
    )


def create_filename(title, used_names):
    """Generate a filesystem-safe unique filename."""

    name = re.sub(
        r'[<>:"/\\|?*\x00-\x1f]',
        "-",
        title
    )

    name = re.sub(r"\s+", "-", name)
    name = name.strip(". -")
    name = name[:150]

    if not name:
        name = "webpage"

    original_name = name
    counter = 2

    while name.casefold() in used_names:

        suffix = f"-{counter}"

        name = (
            original_name[:150 - len(suffix)]
            + suffix
        )

        counter += 1

    used_names.add(name.casefold())

    return f"{name}.json"


def is_valid_http_url(url):
    """Validate URL format."""

    try:
        parsed = urlparse(url)

        return (
            parsed.scheme.lower() in {"http", "https"}
            and bool(parsed.netloc)
            and bool(parsed.hostname)
        )

    except ValueError:
        return False


def is_inside_excluded_structure(element, stop_at=None):
    """Check whether an element belongs to excluded content."""

    current = element

    excluded_classes = {
        "sidebar",
        "widget-area",
        "comments-area",
        "comment-respond",
        "breadcrumb",
        "breadcrumbs",
        "share-buttons",
        "social-share",
        "related-posts",
        "advertisement",
        "ads"
    }

    while isinstance(current, Tag) and current is not stop_at:

        if current.name in EXCLUDED_TAGS:
            return True

        if current.get("aria-hidden") == "true":
            return True

        classes = set(current.get("class", []))

        if classes.intersection(excluded_classes):
            return True

        current = current.parent

    return False


def is_inside_toc(element, stop_at=None):
    """Check whether an element belongs to a table of contents."""

    current = element

    while isinstance(current, Tag) and current is not stop_at:

        classes = " ".join(current.get("class", [])).casefold()
        element_id = str(current.get("id", "")).casefold()

        if (
            "ez-toc" in classes
            or "lwptoc" in classes
            or "table-of-contents" in classes
            or "elementor-toc" in classes
            or "table-of-contents" in element_id
        ):
            return True

        current = current.parent

    return False


def clean_paragraph(element):
    """
    Extract readable text from a paragraph or list item.

    Keeps meaningful hyperlink text.
    Ignores images, scripts, styles and other unwanted
    nested elements.
    """

    if element is None:
        return ""

    text_parts = []

    for node in element.descendants:

        if not isinstance(node, NavigableString):
            continue

        parent = node.parent
        skip = False

        while isinstance(parent, Tag) and parent is not element:

            if parent.name in EXCLUDED_TAGS:
                skip = True
                break

            parent = parent.parent

        if not skip:
            text_parts.append(str(node))

    return " ".join(
        " ".join(text_parts).split()
    )


def remove_unwanted_elements(root):
    """Remove navigation and other unwanted structures."""

    for selector in EXCLUDED_SELECTORS + TOC_SELECTORS:

        for element in root.select(selector):
            element.decompose()


def is_valid_article_text(text):
    """Reject empty or navigation-like text."""

    if not text:
        return False

    normalized = text.strip()

    if not normalized:
        return False

    if normalized.casefold() in {
        "contents",
        "table of contents",
        "quick links"
    }:
        return False

    return True


# ============================================================
# CONTENT ROOT AND PAGE TITLE
# ============================================================

def find_content_root(soup):
    """Identify the main webpage content area."""

    for selector in CONTENT_SELECTORS:

        element = soup.select_one(selector)

        if element:
            return element

    return soup.body or soup


def get_page_title(soup, content_root, url):
    """Extract the article title."""

    # Prefer the H1 belonging to the article.
    h1 = content_root.find("h1")

    if h1:

        title = clean_text(h1)

        if title:
            return title

    # Fall back to the document title.
    if soup.title:

        title = clean_text(soup.title)

        if title:
            return title

    # Final fallback: URL path.
    path = unquote(
        urlparse(url).path
    ).strip("/")

    if path:
        return path.split("/")[-1]

    return "homepage"


# ============================================================
# MAIN STRUCTURED EXTRACTION
# ============================================================

def extract_subtitles_and_content(content_root, page_title):
    """
    Extract subtitles and the text belonging to each subtitle.

    Rules:

    1. H1 is treated as the page title, not a subtitle.
    2. H2-H6 are treated as subtitles.
    3. Paragraphs, list items and blockquotes are collected.
    4. Images are ignored.
    5. Tables are excluded from paragraph content.
    6. Each text element belongs to the most recent subtitle.
    7. TOC and excluded structures are ignored.
    """

    subtitles = []
    content = {}

    heading_counts = {}

    current_subtitle = None

    elements = content_root.find_all(
        HEADING_TAGS + CONTENT_TAGS
    )

    for element in elements:

        if is_inside_excluded_structure(
            element,
            stop_at=content_root
        ):
            continue

        if is_inside_toc(
            element,
            stop_at=content_root
        ):
            continue

        # Avoid duplicate paragraph extraction inside list items.
        if (
            element.name == "p"
            and element.find_parent("li")
        ):
            continue

        # Avoid extracting nested list items twice.
        if (
            element.name == "li"
            and element.find_parent("li")
        ):
            continue

        # ----------------------------------------------------
        # SUBTITLE
        # ----------------------------------------------------

        if element.name in SUBTITLE_TAGS:

            heading_text = clean_text(element)

            if not is_valid_article_text(heading_text):
                current_subtitle = None
                continue

            heading_counts[heading_text] = (
                heading_counts.get(heading_text, 0) + 1
            )

            # Avoid overwriting content when headings repeat.
            if heading_counts[heading_text] == 1:
                subtitle_key = heading_text
            else:
                subtitle_key = (
                    f"{heading_text} "
                    f"({heading_counts[heading_text]})"
                )

            subtitles.append(subtitle_key)

            content[subtitle_key] = []

            current_subtitle = subtitle_key

            continue

        # ----------------------------------------------------
        # PAGE TITLE
        # ----------------------------------------------------

        if element.name == "h1":

            heading_text = clean_text(element)

            # Do not treat the page title as a subtitle.
            if (
                heading_text.casefold()
                == str(page_title).strip().casefold()
            ):
                continue

            # Other H1 elements are unusual, but should not
            # silently become paragraph content.
            continue

        # ----------------------------------------------------
        # PARAGRAPHS, LIST ITEMS AND BLOCKQUOTES
        # ----------------------------------------------------

        if element.name in CONTENT_TAGS:

            # Text before the first subtitle is deliberately
            # not assigned to an arbitrary section.
            if current_subtitle is None:
                continue

            # Never extract table-cell content as paragraphs.
            if element.find_parent("table"):
                continue

            text = clean_paragraph(element)

            if not is_valid_article_text(text):
                continue

            content[current_subtitle].append(text)

    return subtitles, content


# ============================================================
# TABLE EXTRACTION
# ============================================================

def extract_table_rows(table):
    """
    Extract table content row-wise.

    Supports:
    - HTML table rows
    - TH and TD cells
    - Basic rowspan
    - Basic colspan

    Returns a list of rows, with each row represented
    as a list of cell strings.
    """

    output_rows = []
    pending_spans = {}

    max_columns = 0

    rows = [
        tr for tr in table.find_all("tr")
        if tr.find_parent("table") is table
    ]

    for tr in rows:

        row = []
        column = 0

        # Fill cells carried forward from previous rowspans.
        while column in pending_spans:

            value, remaining = pending_spans[column]

            row.append(value)

            if remaining <= 1:
                del pending_spans[column]
            else:
                pending_spans[column] = (
                    value,
                    remaining - 1
                )

            column += 1

        cells = tr.find_all(
            ["th", "td"],
            recursive=False
        )

        for cell in cells:

            while column in pending_spans:

                value, remaining = pending_spans[column]

                row.append(value)

                if remaining <= 1:
                    del pending_spans[column]
                else:
                    pending_spans[column] = (
                        value,
                        remaining - 1
                    )

                column += 1

            value = clean_text(cell)

            try:
                colspan = max(
                    1,
                    int(cell.get("colspan", 1))
                )
            except (TypeError, ValueError):
                colspan = 1

            try:
                rowspan = max(
                    1,
                    int(cell.get("rowspan", 1))
                )
            except (TypeError, ValueError):
                rowspan = 1

            for offset in range(colspan):

                row.append(value)

                if rowspan > 1:

                    pending_spans[column + offset] = (
                        value,
                        rowspan - 1
                    )

            column += colspan

        if row:

            output_rows.append(row)

            max_columns = max(
                max_columns,
                len(row)
            )

    # Ensure consistent column counts.
    for row in output_rows:

        row.extend(
            [""] * (max_columns - len(row))
        )

    return output_rows


def find_table_position(table, content_root):
    """
    Find the most recent subtitle before the table.

    The returned value identifies the section in which
    the table appears.
    """

    current_subtitle = None

    elements = content_root.find_all(
        SUBTITLE_TAGS + ["table"]
    )

    for element in elements:

        if element is table:
            break

        if element.name in SUBTITLE_TAGS:

            if is_inside_excluded_structure(
                element,
                stop_at=content_root
            ):
                continue

            if is_inside_toc(
                element,
                stop_at=content_root
            ):
                continue

            heading = clean_text(element)

            if is_valid_article_text(heading):
                current_subtitle = heading

    return current_subtitle


def find_table_heading(table):
    """
    Find a table-specific heading.

    Priority:
    1. Table caption.
    2. Explicit heading immediately preceding the table.

    The section position is stored separately.
    """

    caption = table.find(
        "caption",
        recursive=False
    )

    if caption:

        caption_text = clean_text(caption)

        if caption_text:
            return caption_text

    previous = table.find_previous_sibling()

    # Look through a small number of preceding siblings.
    # This avoids accidentally assigning a distant heading.
    checked = 0

    while previous is not None and checked < 4:

        if isinstance(previous, Tag):

            if previous.name in HEADING_TAGS:

                heading_text = clean_text(previous)

                if heading_text:
                    return heading_text

            # A paragraph immediately before a table can
            # sometimes act as its label.
            if previous.name == "p":

                paragraph_text = clean_text(previous)

                if paragraph_text and len(paragraph_text) < 180:
                    return paragraph_text

        previous = previous.find_previous_sibling()
        checked += 1

    return None


def extract_tables(content_root):
    """
    Extract all HTML tables.

    Each table contains:
    - heading
    - position
    - content (row-wise list)
    """

    tables = {}

    table_index = 0

    for table in content_root.find_all("table"):

        if is_inside_excluded_structure(
            table,
            stop_at=content_root
        ):
            continue

        if is_inside_toc(
            table,
            stop_at=content_root
        ):
            continue

        rows = extract_table_rows(table)

        if not rows:
            continue

        table_index += 1

        table_id = f"table{table_index}"

        table_heading = find_table_heading(table)

        table_position = find_table_position(
            table,
            content_root
        )

        tables[table_id] = {

            "heading": table_heading,

            "position": table_position,

            "content": rows

        }

    return tables


# ============================================================
# LEGACY COMPATIBILITY STRUCTURES
# ============================================================

def build_legacy_sections(subtitles, content):
    """
    Build the previous schema's sections list.

    This is retained temporarily so existing app.py
    code can continue reading the crawler output.
    """

    sections = []

    for index, subtitle in enumerate(subtitles, start=1):

        sections.append({

            "section_id": f"section_{index:04d}",

            "heading": subtitle,

            "heading_key": subtitle,

            "heading_level": None,

            "parent_heading": None,

            "content": content.get(subtitle, [])

        })

    return sections


def build_legacy_table_content(tables):
    """Convert the new table structure to the old structure."""

    legacy_tables = []

    for index, (table_id, table) in enumerate(
        tables.items(),
        start=1
    ):

        rows = table.get("content", [])

        legacy_tables.append({

            "table_id": table_id,

            "table_index": index,

            "table_title": (
                table.get("heading")
                or table.get("position")
                or f"Table {index}"
            ),

            "associated_section": table.get("position"),

            "table_header": False,

            "headers": [],

            "rows": [
                {"cells": row}
                for row in rows
            ],

            "table_values": rows

        })

    return {

        "table_count": len(legacy_tables),

        "tables": legacy_tables

    }


# ============================================================
# WEBPAGE PROCESSING
# ============================================================

def extract_webpage(url, session):
    """Download a webpage once and extract structured content."""

    response = session.get(
        url,
        timeout=TIMEOUT
    )

    response.raise_for_status()

    soup = BeautifulSoup(
        response.content,
        "lxml"
    )

    content_root_original = find_content_root(soup)

    # Work on a copy so the original soup remains available.
    content_root_soup = BeautifulSoup(
        str(content_root_original),
        "lxml"
    )

    # The copied document may contain html/body wrappers.
    content_root = (
        content_root_soup.body
        or content_root_soup
    )

    remove_unwanted_elements(content_root)

    page_title = get_page_title(
        soup,
        content_root,
        response.url
    )

    # --------------------------------------------------------
    # NEW STRUCTURE
    # --------------------------------------------------------

    subtitles, content = extract_subtitles_and_content(
        content_root,
        page_title
    )

    tables = extract_tables(
        content_root
    )

    paragraph_count = sum(
        len(paragraphs)
        for paragraphs in content.values()
    )

    # --------------------------------------------------------
    # LEGACY STRUCTURE
    # --------------------------------------------------------

    legacy_sections = build_legacy_sections(
        subtitles,
        content
    )

    legacy_table_content = build_legacy_table_content(
        tables
    )

    # --------------------------------------------------------
    # FINAL JSON
    # --------------------------------------------------------

    combined_data = {

        "schema_version": "3.0",

        # New primary structure.
        "title": page_title,

        "subtitles": subtitles,

        "content": content,

        "tables": tables,

        # Existing source metadata.
        "source_url": url,

        "final_url": response.url,

        "http_status": response.status_code,

        "fetched_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),

        # Compatibility aliases for the current app.py.
        "page_title": page_title,

        "sections": legacy_sections,

        "table_content": legacy_table_content,

        # Retained for compatibility with any code that
        # still expects the original text_content dictionary.
        "text_content": content,

        "extraction_stats": {

            "section_count": len(subtitles),

            "heading_count": len(subtitles),

            "paragraph_and_list_item_count": paragraph_count,

            "table_count": len(tables)

        }

    }

    return combined_data


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    if not INPUT_FILE.exists():

        print(
            f"ERROR: Input file not found: {INPUT_FILE}"
        )

        return

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Read URLs.
    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8-sig"
    ) as file:

        urls = [

            line.strip()

            for line in file

            if line.strip()
            and not line.strip().startswith("#")

        ]

    if not urls:

        print("No URLs found in testlist.txt.")

        return

    print(f"URLs found: {len(urls)}")

    print(f"Output folder: {OUTPUT_DIR}")

    print("-" * 60)

    successful = 0
    failed = 0

    used_names = set()

    session = create_session()

    try:

        for number, url in enumerate(
            urls,
            start=1
        ):

            print(
                f"[{number}/{len(urls)}] Processing: {url}"
            )

            if not is_valid_http_url(url):

                print("  ERROR: Invalid URL. Skipping.")

                failed += 1

                continue

            try:

                data = extract_webpage(
                    url,
                    session
                )

                filename = create_filename(
                    data["title"],
                    used_names
                )

                output_file = OUTPUT_DIR / filename

                with open(
                    output_file,
                    "w",
                    encoding="utf-8"
                ) as file:

                    json.dump(
                        data,
                        file,
                        indent=4,
                        ensure_ascii=False
                    )

                stats = data["extraction_stats"]

                print(f"  SUCCESS: {filename}")

                print(
                    f"  Subtitles: "
                    f"{stats['section_count']}"
                )

                print(
                    f"  Paragraphs/list items: "
                    f"{stats['paragraph_and_list_item_count']}"
                )

                print(
                    f"  Tables: "
                    f"{stats['table_count']}"
                )

                successful += 1

            except requests.RequestException as error:

                print(
                    f"  DOWNLOAD ERROR: {error}"
                )

                failed += 1

            except Exception as error:

                print(
                    f"  EXTRACTION ERROR: {error}"
                )

                failed += 1

            if (
                number < len(urls)
                and REQUEST_DELAY_SECONDS > 0
            ):

                time.sleep(
                    REQUEST_DELAY_SECONDS
                )

    finally:

        session.close()

    print("-" * 60)

    print("PROCESSING COMPLETE")

    print(f"Total URLs: {len(urls)}")

    print(f"Successful: {successful}")

    print(f"Failed: {failed}")

    print(f"JSON output folder: {OUTPUT_DIR}")


if __name__ == "__main__":

    main()
