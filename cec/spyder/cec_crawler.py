
import requests
import json
import re

from pathlib import Path
from urllib.parse import urlparse, unquote
from bs4 import BeautifulSoup, NavigableString


# ---------------- CONFIGURATION ----------------

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "testlist.txt"
OUTPUT_DIR = BASE_DIR / "CM_COMBINED_JSON"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "Chrome/130.0 Safari/537.36"
    )
}

TIMEOUT = 30


# Structures whose content should be ignored by TEXT extraction
EXCLUDED_TAGS = {
    "table",
    "a",
    "ul",
    "ol",
    "nav",
    "header",
    "footer",
    "aside",
    "script",
    "style",
    "noscript",
    "form",
    "figure"
}


# ---------------- COMMON FUNCTIONS ----------------


def clean_text(element):
    """Extract readable text from an HTML element."""

    return " ".join(element.get_text(" ", strip=True).split())


def create_filename(title, used_names):
    """Create a readable and unique JSON filename."""

    name = re.sub(
        r'[<>:"/\\|?*\x00-\x1f]',
        "-",
        title
    )

    name = re.sub(r"\s+", "-", name)
    name = name.strip(". -")

    if not name:
        name = "webpage"

    original_name = name
    counter = 2

    while name.lower() in used_names:
        name = f"{original_name}-{counter}"
        counter += 1

    used_names.add(name.lower())

    return f"{name}.json"


# ---------------- TEXT EXTRACTION ----------------


def is_excluded(element):
    """Check whether an element is inside an excluded structure."""

    current = element

    while current is not None:

        if getattr(current, "name", None) in EXCLUDED_TAGS:
            return True

        current = current.parent

    return False


def clean_paragraph(element):
    """
    Extract paragraph text while excluding hyperlinks
    and unwanted nested structures.
    """

    text_parts = []

    for node in element.descendants:

        if isinstance(node, NavigableString):

            parent = node.parent
            skip = False

            while parent is not None and parent != element:

                if getattr(parent, "name", None) in EXCLUDED_TAGS:
                    skip = True
                    break

                parent = parent.parent

            if not skip:
                text_parts.append(str(node))

    return " ".join(" ".join(text_parts).split())


def find_content_root(soup):
    """Identify the main webpage content area."""

    selectors = [
        ".entry-content",
        ".wp-block-post-content",
        ".post-content",
        ".elementor-widget-theme-post-content",
        "article",
        "main",
        ".elementor",
        "body"
    ]

    for selector in selectors:

        element = soup.select_one(selector)

        if element:
            return element

    return soup


def get_page_title(soup, content_root, url):
    """Extract the webpage title."""

    for heading in content_root.find_all("h1"):

        if not is_excluded(heading):

            title = clean_text(heading)

            if title:
                return title

    if soup.title:

        title = clean_text(soup.title)

        if title:
            return title

    path = unquote(urlparse(url).path).strip("/")

    if path:
        return path.split("/")[-1]

    return "homepage"


def create_unique_heading_key(heading, sections):
    """Prevent repeated headings from overwriting each other."""

    if heading not in sections:
        return heading

    counter = 2

    while f"{heading} ({counter})" in sections:
        counter += 1

    return f"{heading} ({counter})"


def extract_text_content(content_root):
    """
    Extract headings and their respective paragraphs
    in HTML document order.
    """

    sections = {}
    current_heading = None

    elements = content_root.find_all(
        ["h1", "h2", "h3", "h4", "h5", "h6", "p"]
    )

    for element in elements:

        if is_excluded(element):
            continue

        # Start a new section when a heading appears
        if element.name in [
            "h1", "h2", "h3", "h4", "h5", "h6"
        ]:

            heading_text = clean_text(element)

            if not heading_text:
                current_heading = None
                continue

            current_heading = create_unique_heading_key(
                heading_text,
                sections
            )

            sections[current_heading] = []

        # Add paragraphs to the current heading
        elif element.name == "p":

            if current_heading is None:
                continue

            paragraph_text = clean_paragraph(element)

            if paragraph_text:
                sections[current_heading].append(
                    paragraph_text
                )

    return sections


# ---------------- TABLE EXTRACTION ----------------


def extract_table_content(soup):
    """
    Extract all HTML tables from the webpage.

    This preserves the original table extraction logic.
    """

    tables = []

    for table_index, table in enumerate(
        soup.find_all("table"),
        start=1
    ):

        table_values = []

        for tr in table.find_all("tr"):

            cells = tr.find_all(
                ["th", "td"],
                recursive=False
            )

            if cells:

                row = [
                    clean_text(cell)
                    for cell in cells
                ]

                table_values.append(row)

        if not table_values:
            continue

        tables.append({
            "table_index": table_index,
            "table_header": bool(table.find("th")),
            "table_values": table_values
        })

    return {
        "table_count": len(tables),
        "tables": tables
    }


# ---------------- WEBPAGE PROCESSING ----------------


def extract_webpage(url):
    """
    Download each webpage once and extract both
    text content and table content.
    """

    response = requests.get(
        url,
        headers=HEADERS,
        timeout=TIMEOUT
    )

    response.raise_for_status()

    soup = BeautifulSoup(
        response.text,
        "lxml"
    )

    content_root = find_content_root(soup)

    page_title = get_page_title(
        soup,
        content_root,
        url
    )

    # FIRST: Extract all text
    text_content = extract_text_content(
        content_root
    )

    # SECOND: Extract all tables
    table_content = extract_table_content(
        soup
    )

    # Combine both extraction results
    combined_data = {
        "source_url": url,
        "page_title": page_title,
        "text_content": text_content,
        "table_content": table_content
    }

    return combined_data


# ---------------- MAIN PROGRAM ----------------


def main():

    if not INPUT_FILE.exists():

        print(f"ERROR: Input file not found: {INPUT_FILE}")
        return

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Read URLs from testlist.txt
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
    print("-" * 50)

    successful = 0
    failed = 0

    used_names = set()

    # Process URLs sequentially
    for number, url in enumerate(urls, start=1):

        print(
            f"[{number}/{len(urls)}] Processing: {url}"
        )

        parsed_url = urlparse(url)

        # Validate URL
        if (
            parsed_url.scheme not in ("http", "https")
            or not parsed_url.netloc
        ):

            print("  ERROR: Invalid URL. Skipping.")
            failed += 1
            continue

        try:

            # Extract text and tables
            data = extract_webpage(url)

            # Generate unique filename
            filename = create_filename(
                data["page_title"],
                used_names
            )

            output_file = OUTPUT_DIR / filename

            # Save combined JSON
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

            # Calculate extraction statistics
            heading_count = len(
                data["text_content"]
            )

            paragraph_count = sum(
                len(paragraphs)
                for paragraphs in data["text_content"].values()
            )

            table_count = data["table_content"]["table_count"]

            print(f"  SUCCESS: {filename}")
            print(f"  Headings extracted: {heading_count}")
            print(f"  Paragraphs extracted: {paragraph_count}")
            print(f"  Tables extracted: {table_count}")

            successful += 1

        except requests.RequestException as error:

            print(f"  DOWNLOAD ERROR: {error}")
            failed += 1

        except Exception as error:

            print(f"  ERROR: {error}")
            failed += 1

    # Final report
    print("-" * 50)
    print("PROCESSING COMPLETE")
    print(f"Total URLs: {len(urls)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"JSON output folder: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
