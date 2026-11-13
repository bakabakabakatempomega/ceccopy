
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

EXCLUDED_TAGS = {
    "a", "nav", "header", "footer", "aside",
    "script", "style", "noscript", "form",
    "button", "svg", "iframe"
}

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


# ============================================================
# HTTP SESSION
# ============================================================

def create_session():

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
    """Extract normalized readable text."""

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


def clean_paragraph(element):
    """
    Extract text while excluding hyperlinks and unwanted
    nested structures.
    """

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
    """Remove navigation, sidebar and other unwanted elements."""

    for selector in EXCLUDED_SELECTORS:

        for element in root.select(selector):

            element.decompose()


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
    """Extract webpage title."""

    h1 = content_root.find("h1")

    if h1:

        title = clean_text(h1)

        if title:
            return title

    if soup.title:

        title = clean_text(soup.title)

        if title:
            return title

    path = unquote(
        urlparse(url).path
    ).strip("/")

    if path:
        return path.split("/")[-1]

    return "homepage"


# ============================================================
# TEXT AND SECTION EXTRACTION
# ============================================================

def extract_text_content(content_root):
    """
    Extract headings, paragraphs and list items.

    Maintains:
    1. Original dictionary structure.
    2. Structured sections with heading hierarchy.
    """

    sections_dict = {}
    sections_list = []

    heading_stack = []
    current_section = None

    heading_counts = {}

    elements = content_root.find_all(
        ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li"]
    )

    for element in elements:

        if is_inside_excluded_structure(
            element,
            stop_at=content_root
        ):
            continue

        # Avoid duplicate extraction of paragraphs inside lists.
        if element.name == "p" and element.find_parent("li"):
            continue

        # Avoid extracting nested list items twice.
        if element.name == "li" and element.find_parent("li"):
            continue

        # ----------------------------------------------------
        # HEADING
        # ----------------------------------------------------

        if element.name in [
            "h1", "h2", "h3", "h4", "h5", "h6"
        ]:

            heading_text = clean_text(element)

            if not heading_text:
                current_section = None
                continue

            level = int(element.name[1])

            heading_counts[heading_text] = (
                heading_counts.get(heading_text, 0) + 1
            )

            if heading_counts[heading_text] == 1:
                section_key = heading_text
            else:
                section_key = (
                    f"{heading_text} "
                    f"({heading_counts[heading_text]})"
                )

            # Remove headings at the same or deeper level.
            while (
                heading_stack
                and heading_stack[-1]["level"] >= level
            ):

                heading_stack.pop()

            parent_heading = (
                heading_stack[-1]["heading"]
                if heading_stack
                else None
            )

            section = {

                "section_id": (
                    f"section_{len(sections_list) + 1:04d}"
                ),

                "heading": heading_text,
                "heading_key": section_key,
                "heading_level": level,
                "parent_heading": parent_heading,
                "content": []

            }

            sections_list.append(section)

            sections_dict[section_key] = []

            heading_stack.append({

                "level": level,
                "heading": heading_text,
                "section_id": section["section_id"]

            })

            current_section = section

            continue

        # ----------------------------------------------------
        # PARAGRAPH OR LIST ITEM
        # ----------------------------------------------------

        if current_section is None:
            continue

        text = clean_paragraph(element)

        if not text:
            continue

        current_section["content"].append(text)

        sections_dict[
            current_section["heading_key"]
        ].append(text)

    return sections_dict, sections_list


# ============================================================
# TABLE EXTRACTION
# ============================================================

def extract_table_rows(table):
    """
    Extract table rows.

    Expands basic rowspan and colspan values into a
    rectangular structure.
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


def find_table_context(table, content_root):
    """Find the nearest preceding heading."""

    preceding_heading = None

    elements = content_root.find_all(
        ["h1", "h2", "h3", "h4", "h5", "h6", "table"]
    )

    for element in elements:

        if element is table:
            break

        if element.name.startswith("h"):

            heading = clean_text(element)

            if heading:
                preceding_heading = heading

    return preceding_heading


def infer_table_title(table, associated_section, table_index):
    """Determine a useful table title."""

    caption = table.find(
        "caption",
        recursive=False
    )

    if caption:

        caption_text = clean_text(caption)

        if caption_text:
            return caption_text

    previous_heading = table.find_previous(
        ["h2", "h3", "h4", "h5", "h6"]
    )

    if previous_heading:

        heading_text = clean_text(previous_heading)

        if heading_text:
            return heading_text

    if associated_section:
        return associated_section

    return f"Table {table_index}"


def split_headers_and_rows(table, table_values):
    """
    Identify explicit HTML header cells.

    When no TH cells exist, do not guess which row is the header.
    """

    header_rows = []

    for tr in table.find_all("tr"):

        if tr.find_parent("table") is not table:
            continue

        if tr.find_all("th", recursive=False):
            header_rows.append(tr)

    if not header_rows:
        return [], table_values

    headers = []

    for tr in header_rows:

        cells = tr.find_all(
            ["th", "td"],
            recursive=False
        )

        headers = [
            clean_text(cell)
            for cell in cells
        ]

    header_row_ids = {
        id(row) for row in header_rows
    }

    data_rows = []

    actual_rows = [
        row for row in table.find_all("tr")
        if row.find_parent("table") is table
    ]

    for tr, row in zip(actual_rows, table_values):

        if id(tr) not in header_row_ids:
            data_rows.append(row)

    return headers, data_rows


def extract_table_content(content_root):
    """
    Extract tables and associate each with its nearest
    preceding section heading.
    """

    tables = []

    for table_index, table in enumerate(
        content_root.find_all("table"),
        start=1
    ):

        if is_inside_excluded_structure(
            table,
            stop_at=content_root
        ):
            continue

        table_values = extract_table_rows(table)

        if not table_values:
            continue

        associated_section = find_table_context(
            table,
            content_root
        )

        table_title = infer_table_title(
            table,
            associated_section,
            table_index
        )

        headers, data_rows = split_headers_and_rows(
            table,
            table_values
        )

        tables.append({

            "table_id": f"table_{table_index:03d}",

            "table_index": table_index,

            "table_title": table_title,

            "associated_section": associated_section,

            "table_header": bool(headers),

            "headers": headers,

            "rows": [
                {"cells": row}
                for row in data_rows
            ],

            # Backward compatibility.
            "table_values": table_values

        })

    return {

        "table_count": len(tables),

        "tables": tables

    }


# ============================================================
# WEBPAGE PROCESSING
# ============================================================

def extract_webpage(url, session):
    """Download a webpage once and extract its content."""

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

    # Work on a copy so that the original soup remains available.
    content_root = BeautifulSoup(
        str(content_root_original),
        "lxml"
    )

    remove_unwanted_elements(content_root)

    page_title = get_page_title(
        soup,
        content_root,
        response.url
    )

    text_content, sections = extract_text_content(
        content_root
    )

    table_content = extract_table_content(
        content_root
    )

    paragraph_count = sum(
        len(section["content"])
        for section in sections
    )

    combined_data = {

        "schema_version": "2.0",

        "source_url": url,

        "final_url": response.url,

        "page_title": page_title,

        "http_status": response.status_code,

        "fetched_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),

        "text_content": text_content,

        "sections": sections,

        "table_content": table_content,

        "extraction_stats": {

            "section_count": len(sections),

            "heading_count": len(sections),

            "paragraph_and_list_item_count": paragraph_count,

            "table_count": table_content["table_count"]

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
                    data["page_title"],
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
                    f"  Sections/headings: "
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
