
"""
query_spider.py

Search CareerMudhra and crawl relevant pages into temporary JSON files.

Location:
    CEC/spyder/query_spider.py

Crawler:
    CEC/spyder/cec_crawler_v2.py

Temporary output:
    CEC/spyder/cec_temp_kb/<request_id>/

The crawler delegates webpage extraction to cec_crawler_v2.py.
All saved JSON files use schema_version 2.0.

Public API:
    search_and_crawl()
    cleanup_request()
    cleanup_stale_temp()
"""

import json
import re
import time
import uuid
import shutil
import logging
import ipaddress

from datetime import datetime, timezone, timedelta
from pathlib import Path
from urllib.parse import (
    urljoin,
    urlparse,
    urlunparse,
    urlencode,
    parse_qsl,
)
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup


# -------------------------------------------------------------------
# Crawler import
# -------------------------------------------------------------------

try:
    # When imported through app.py
    from .cec_crawler_v2 import create_session, extract_webpage

except ImportError:
    # When query_spider.py is executed directly
    from cec_crawler_v2 import create_session, extract_webpage


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

SPYDER_DIR = Path(__file__).resolve().parent
TEMP_KB_DIR = SPYDER_DIR / "cec_temp_kb"

BASE_URL = "https://careermudhra.com/"
SITE_HOST = "careermudhra.com"
ROBOTS_URL = urljoin(BASE_URL, "robots.txt")

DEFAULT_MAX_PAGES = 5
DEFAULT_TIMEOUT = (10, 30)
DEFAULT_DELAY_SECONDS = 1.0
DEFAULT_MAX_HTML_BYTES = 5 * 1024 * 1024
DEFAULT_STALE_HOURS = 2

MAX_SEARCH_QUERIES = 5
MAX_SEARCH_RESULTS_PER_QUERY = 40
MAX_TOTAL_SEARCH_RESULTS = 150

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Common functions
# -------------------------------------------------------------------

def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_iso() -> str:
    return _utc_now().isoformat()


def _clean_query(value) -> str:
    """
    Convert a value into a normalized search string.

    Supports strings and lists.
    """

    if isinstance(value, list):

        value = " ".join(
            str(item).strip()
            for item in value
            if item
        )

    return re.sub(
        r"\s+",
        " ",
        str(value or "")
    ).strip()


def _tokenize(value: str) -> set[str]:
    """
    Extract normalized words from a string.
    """

    stopwords = {
        "a", "an", "the", "of", "in", "at",
        "and", "or", "for", "to", "by",
        "with", "on", "from", "about"
    }

    return {
        token
        for token in re.findall(
            r"[a-z0-9]+",
            str(value).casefold()
        )
        if len(token) > 1 and token not in stopwords
    }


def _flatten_text(value) -> list[str]:
    """
    Recursively extract readable strings from nested JSON data.
    """

    if isinstance(value, str):

        text = value.strip()

        return [text] if text else []

    if isinstance(value, (int, float)):

        return [str(value)]

    if isinstance(value, list):

        result = []

        for item in value:
            result.extend(
                _flatten_text(item)
            )

        return result

    if isinstance(value, dict):

        result = []

        ignored_keys = {
            "section_id",
            "heading_key",
            "heading_level",
            "table_id",
            "table_index",
            "table_header",
        }

        for key, item in value.items():

            if key in ignored_keys:
                continue

            result.extend(
                _flatten_text(item)
            )

        return result

    return []


# -------------------------------------------------------------------
# Search query generation
# -------------------------------------------------------------------

def _generate_search_queries(query) -> list[str]:
    """
    Generate multiple targeted CareerMudhra search queries.

    Accepts:
        Plain search string
        OR dictionary returned by query_extractor.

    Search priority:
        1. Original extracted search query.
        2. College + degree + specialization.
        3. College + specialization + requested information.
        4. College + degree + requested information.
        5. College name alone.

    Does not modify query_extractor.py.
    """

    if isinstance(query, dict):

        original_question = _clean_query(
            query.get("original_question")
        )

        primary_query = _clean_query(
            query.get("search_query")
        )

        colleges = [
            _clean_query(item)
            for item in query.get("college_names", [])
            if _clean_query(item)
        ]

        degrees = [
            _clean_query(item)
            for item in query.get("degree_names", [])
            if _clean_query(item)
        ]

        fields = [
            _clean_query(item)
            for item in query.get("fields", [])
            if _clean_query(item)
        ]

        requested_data = [
            _clean_query(item)
            for item in query.get("requested_data", [])
            if _clean_query(item)
        ]

        locations = [
            _clean_query(item)
            for item in query.get("locations", [])
            if _clean_query(item)
        ]

        keywords = [
            _clean_query(item)
            for item in query.get("keywords", [])
            if _clean_query(item)
        ]

    else:

        primary_query = _clean_query(query)
        original_question = primary_query

        colleges = []
        degrees = []
        fields = []
        requested_data = []
        locations = []
        keywords = []

    generated = []
    seen = set()

    def add(value):

        value = _clean_query(value)

        if not value:
            return

        normalized = value.casefold()

        if normalized not in seen:

            seen.add(normalized)
            generated.append(value)

    # 1. Original extracted query.
    add(primary_query)

    # 2. College-focused queries.
    for college in colleges:

        degree_part = " ".join(degrees)
        field_part = " ".join(fields)
        requested_part = " ".join(requested_data)
        location_part = " ".join(locations)

        # College + degree + specialization.
        add(
            f"{college} {degree_part} {field_part}"
        )

        # College + specialization + requested information.
        add(
            f"{college} {field_part} {requested_part}"
        )

        # College + degree + requested information.
        add(
            f"{college} {degree_part} {requested_part}"
        )

        # College + degree + location.
        add(
            f"{college} {degree_part} {location_part}"
        )

        # College name alone.
        add(college)

    # 3. Degree and specialization fallback.
    if degrees or fields:

        add(
            f"{' '.join(degrees)} "
            f"{' '.join(fields)} "
            f"{' '.join(locations)}"
        )

    # 4. Keyword fallback.
    if keywords:

        add(
            " ".join(keywords[:8])
        )

    # 5. Original question fallback.
    add(original_question)

    return generated[:MAX_SEARCH_QUERIES]


# -------------------------------------------------------------------
# URL and filesystem safety
# -------------------------------------------------------------------

def _is_allowed_url(url: str) -> bool:
    """
    Permit only HTTPS URLs on careermudhra.com or its subdomains.

    Reject credentials, non-HTTPS schemes, IP literals,
    localhost and unrelated domains.
    """

    try:

        parsed = urlparse(url)

        if parsed.scheme.lower() != "https":
            return False

        if not parsed.hostname:
            return False

        host = parsed.hostname.lower().rstrip(".")

        if not (
            host == SITE_HOST
            or host.endswith("." + SITE_HOST)
        ):
            return False

        if parsed.username or parsed.password:
            return False

        try:

            ipaddress.ip_address(host)
            return False

        except ValueError:
            pass

        return True

    except Exception:

        return False


def _normalize_url(url: str) -> str:
    """
    Remove fragments and common tracking parameters.
    """

    parsed = urlparse(url)

    query_items = []

    for key, value in parse_qsl(
        parsed.query,
        keep_blank_values=False
    ):

        if key.lower().startswith("utm_"):
            continue

        if key.lower() in {
            "fbclid",
            "gclid",
            "mc_cid",
            "mc_eid"
        }:
            continue

        query_items.append(
            (key, value)
        )

    clean_query = urlencode(
        query_items,
        doseq=True
    )

    normalized = parsed._replace(
        scheme="https",
        netloc=parsed.netloc.lower(),
        query=clean_query,
        fragment=""
    )

    result = urlunparse(normalized)

    if not parsed.path:

        result = result.replace(
            parsed.netloc.lower(),
            parsed.netloc.lower() + "/",
            1
        )

    return result


def _make_request_directory(
    request_id: str | None = None
) -> tuple[str, Path]:
    """
    Create a request-specific directory inside cec_temp_kb.
    """

    TEMP_KB_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    if request_id is None:

        request_id = uuid.uuid4().hex

    if not re.fullmatch(
        r"[A-Za-z0-9_-]{1,80}",
        request_id
    ):

        raise ValueError(
            "Invalid request_id."
        )

    request_dir = (
        TEMP_KB_DIR / request_id
    ).resolve()

    if request_dir.parent != TEMP_KB_DIR.resolve():

        raise ValueError(
            "Invalid temporary directory path."
        )

    request_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    return request_id, request_dir


# -------------------------------------------------------------------
# HTTP session and robots.txt
# -------------------------------------------------------------------

def _new_session() -> requests.Session:
    """
    Use the retry-enabled session from cec_crawler_v2.py.
    """

    return create_session()


def _load_robots(
    session: requests.Session,
    timeout=DEFAULT_TIMEOUT
):
    """
    Download and parse robots.txt.

    Fail closed if robots.txt cannot be retrieved.
    """

    with session.get(
        ROBOTS_URL,
        timeout=timeout,
        allow_redirects=True
    ) as response:

        response.raise_for_status()

        if response.status_code != 200:

            raise RuntimeError(
                f"Cannot verify robots.txt "
                f"(HTTP {response.status_code})."
            )

        if not _is_allowed_url(response.url):

            raise RuntimeError(
                "robots.txt redirected outside CareerMudhra."
            )

        parser = RobotFileParser()

        parser.set_url(ROBOTS_URL)

        parser.parse(
            response.text.splitlines()
        )

        return parser


def _robots_allows(
    robots: RobotFileParser,
    url: str,
    session: requests.Session
) -> bool:
    """
    Check robots.txt using the same User-Agent as the HTTP session.
    """

    user_agent = session.headers.get(
        "User-Agent",
        "*"
    )

    return robots.can_fetch(
        user_agent,
        url
    )


def _get_html(
    session: requests.Session,
    url: str,
    timeout=DEFAULT_TIMEOUT,
    max_bytes=DEFAULT_MAX_HTML_BYTES
) -> tuple[str, str, int]:
    """
    Fetch HTML from an allowed URL.

    Returns:
        (html_text, final_url, status_code)
    """

    if not _is_allowed_url(url):

        raise ValueError(
            f"Blocked URL outside CareerMudhra: {url}"
        )

    with session.get(
        url,
        timeout=timeout,
        allow_redirects=True,
        stream=True
    ) as response:

        final_url = response.url

        if not _is_allowed_url(final_url):

            raise ValueError(
                "The request redirected outside CareerMudhra."
            )

        response.raise_for_status()

        content_type = response.headers.get(
            "Content-Type",
            ""
        ).lower()

        if (
            "text/html" not in content_type
            and "application/xhtml+xml" not in content_type
        ):

            raise ValueError(
                f"Skipping non-HTML response: {content_type}"
            )

        content_length = response.headers.get(
            "Content-Length"
        )

        if content_length:

            try:

                if int(content_length) > max_bytes:

                    raise ValueError(
                        "HTML response exceeds the configured size limit."
                    )

            except ValueError as exc:

                if "exceeds" in str(exc):
                    raise

        chunks = []
        total_bytes = 0

        for chunk in response.iter_content(
            chunk_size=65536
        ):

            if not chunk:
                continue

            total_bytes += len(chunk)

            if total_bytes > max_bytes:

                raise ValueError(
                    "HTML response exceeds the configured size limit."
                )

            chunks.append(chunk)

        raw = b"".join(chunks)

        encoding = response.encoding or "utf-8"

        html = raw.decode(
            encoding,
            errors="replace"
        )

        return html, final_url, response.status_code


# -------------------------------------------------------------------
# Search result processing
# -------------------------------------------------------------------

def _is_probable_article_url(url: str) -> bool:
    """
    Filter out common WordPress utility pages and non-article links.
    """

    if not _is_allowed_url(url):
        return False

    parsed = urlparse(url)

    path = parsed.path.lower().strip("/")

    if not path:
        return False

    excluded_segments = {
        "category",
        "tag",
        "author",
        "page",
        "feed",
        "wp-admin",
        "wp-content",
        "wp-includes",
        "search",
        "cart",
        "checkout",
        "my-account",
        "privacy-policy"
    }

    segments = [
        segment
        for segment in path.split("/")
        if segment
    ]

    if any(
        segment in excluded_segments
        for segment in segments
    ):
        return False

    if path.endswith((
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".webp",
        ".pdf",
        ".zip"
    )):
        return False

    if "s=" in parsed.query.lower():
        return False

    return True


def _extract_search_result_urls(
    html: str,
    search_page_url: str,
    limit: int = 20
) -> list[dict]:
    """
    Extract likely article links and their visible titles.

    Returns:
        [
            {
                "url": "...",
                "title": "..."
            }
        ]
    """

    soup = BeautifulSoup(
        html,
        "html.parser"
    )

    selectors = [
        "article h2 a",
        "article h3 a",
        ".entry-title a",
        ".post-title a",
        ".search-result a",
        "main article a",
        "article a"
    ]

    anchors = []
    seen_anchor_ids = set()

    for selector in selectors:

        for anchor in soup.select(selector):

            identity = id(anchor)

            if identity not in seen_anchor_ids:

                seen_anchor_ids.add(identity)
                anchors.append(anchor)

    if not anchors:

        for selector in [
            "main a",
            "#content a",
            ".site-content a"
        ]:

            anchors.extend(
                soup.select(selector)
            )

    results = []
    seen_urls = set()

    for anchor in anchors:

        href = anchor.get("href")

        if not href:
            continue

        absolute_url = urljoin(
            search_page_url,
            href
        )

        normalized_url = _normalize_url(
            absolute_url
        )

        if not _is_probable_article_url(
            normalized_url
        ):
            continue

        if normalized_url == _normalize_url(
            search_page_url
        ):
            continue

        if normalized_url in seen_urls:
            continue

        seen_urls.add(normalized_url)

        title = anchor.get_text(
            " ",
            strip=True
        )

        if not title:

            parent = anchor.find_parent("article")

            if parent:

                title = parent.get_text(
                    " ",
                    strip=True
                )

        results.append({

            "url": normalized_url,

            "title": title,

            "search_hits": 1

        })

        if len(results) >= limit:
            break

    return results


# -------------------------------------------------------------------
# Search result relevance ranking
# -------------------------------------------------------------------

def _score_search_result(
    candidate: dict,
    query_data
) -> tuple[int, list[str]]:
    """
    Calculate search result relevance.

    College name receives the highest weight.

    Ranking uses:
        - College name
        - Degree
        - Specialization
        - Requested information
        - Location
        - Number of searches returning the result
    """

    title = _clean_query(
        candidate.get("title")
    )

    url = candidate.get(
        "url",
        ""
    )

    searchable_text = (
        title + " " + url.replace("-", " ")
    ).casefold()

    score = 0
    matched_terms = []

    if not isinstance(query_data, dict):
        query_data = {}

    colleges = query_data.get(
        "college_names",
        []
    )

    degrees = query_data.get(
        "degree_names",
        []
    )

    fields = query_data.get(
        "fields",
        []
    )

    requested_data = query_data.get(
        "requested_data",
        []
    )

    locations = query_data.get(
        "locations",
        []
    )

    result_tokens = _tokenize(searchable_text)

    # ---------------------------------------------------------------
    # 1. College name
    # ---------------------------------------------------------------

    college_match_found = False

    for college in colleges:

        college = _clean_query(college)

        if not college:
            continue

        college_tokens = _tokenize(college)

        if college.casefold() in searchable_text:

            score += 100

            college_match_found = True

            matched_terms.append(college)

        elif college_tokens:

            matching_tokens = (
                college_tokens & result_tokens
            )

            coverage = (
                len(matching_tokens)
                / len(college_tokens)
            )

            if coverage >= 0.5:

                score += int(65 * coverage)

                college_match_found = True

                matched_terms.append(college)

            elif matching_tokens:

                score += int(30 * coverage)

    # ---------------------------------------------------------------
    # 2. Degree
    # ---------------------------------------------------------------

    for degree in degrees:

        degree = _clean_query(degree)

        if not degree:
            continue

        degree_tokens = _tokenize(degree)

        if degree.casefold() in searchable_text:

            score += 20

            matched_terms.append(degree)

        elif degree_tokens:

            matching_tokens = (
                degree_tokens & result_tokens
            )

            if matching_tokens:

                score += int(
                    15 * len(matching_tokens)
                    / len(degree_tokens)
                )

    # ---------------------------------------------------------------
    # 3. Specialization / field
    # ---------------------------------------------------------------

    for field in fields:

        field = _clean_query(field)

        if not field:
            continue

        field_tokens = _tokenize(field)

        if field.casefold() in searchable_text:

            score += 30

            matched_terms.append(field)

        elif field_tokens:

            matching_tokens = (
                field_tokens & result_tokens
            )

            if matching_tokens:

                score += int(
                    20 * len(matching_tokens)
                    / len(field_tokens)
                )

    # ---------------------------------------------------------------
    # 4. Requested information
    # ---------------------------------------------------------------

    for item in requested_data:

        item = _clean_query(item)

        if not item:
            continue

        item_tokens = _tokenize(item)

        if item.casefold() in searchable_text:

            score += 20

            matched_terms.append(item)

        elif item_tokens:

            matching_tokens = (
                item_tokens & result_tokens
            )

            score += (
                5 * len(matching_tokens)
            )

    # ---------------------------------------------------------------
    # 5. Location
    # ---------------------------------------------------------------

    for location in locations:

        location = _clean_query(location)

        if not location:
            continue

        location_tokens = _tokenize(location)

        if location.casefold() in searchable_text:

            score += 10

            matched_terms.append(location)

        elif location_tokens:

            matching_tokens = (
                location_tokens & result_tokens
            )

            if matching_tokens:

                score += int(
                    8 * len(matching_tokens)
                    / len(location_tokens)
                )

    # ---------------------------------------------------------------
    # 6. Penalize unrelated colleges
    # ---------------------------------------------------------------

    if colleges and not college_match_found:

        score -= 50

    # ---------------------------------------------------------------
    # 7. Small boost for results found in multiple searches
    # ---------------------------------------------------------------

    search_hits = candidate.get(
        "search_hits",
        1
    )

    score += min(
        max(search_hits - 1, 0) * 3,
        12
    )

    return score, matched_terms


def _rank_search_results(
    candidates: list[dict],
    query_data
) -> list[dict]:
    """
    Rank search results from most to least relevant.
    """

    ranked = []

    for candidate in candidates:

        score, matched_terms = _score_search_result(
            candidate,
            query_data
        )

        ranked_candidate = dict(candidate)

        ranked_candidate["relevance_score"] = score

        ranked_candidate["matched_terms"] = matched_terms

        ranked.append(ranked_candidate)

    ranked.sort(
        key=lambda item: (
            item["relevance_score"],
            item.get("search_hits", 1)
        ),
        reverse=True
    )

    return ranked


# -------------------------------------------------------------------
# New crawler integration
# -------------------------------------------------------------------

def _page_has_useful_content(page: dict) -> bool:
    """
    Check whether the new-schema page contains useful extracted text.
    """

    if not isinstance(page, dict):
        return False

    title = page.get(
        "page_title",
        ""
    )

    sections = page.get(
        "sections",
        []
    )

    table_content = page.get(
        "table_content",
        {}
    )

    text_content = page.get(
        "text_content",
        {}
    )

    tables = (
        table_content.get("tables", [])
        if isinstance(table_content, dict)
        else []
    )

    section_text = _flatten_text(sections)

    table_text = _flatten_text(tables)

    fallback_text = _flatten_text(text_content)

    combined_text = (
        _flatten_text(title)
        + section_text
        + table_text
    )

    # If sections and tables are empty, use text_content.
    if not section_text and not table_text:

        combined_text.extend(
            fallback_text
        )

    readable_text = " ".join(
        combined_text
    ).strip()

    return bool(
        title
        and len(readable_text) >= 100
    )


def _save_page_json(
    page: dict,
    request_dir: Path,
    page_number: int
) -> Path:
    """
    Save one new-schema page JSON file.
    """

    filename = f"page_{page_number:03d}.json"

    file_path = request_dir / filename

    with file_path.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            page,
            file,
            ensure_ascii=False,
            indent=2
        )

    return file_path


# -------------------------------------------------------------------
# Main public API
# -------------------------------------------------------------------

def search_and_crawl(
    query,
    request_id: str | None = None,
    max_pages: int = DEFAULT_MAX_PAGES,
    timeout=DEFAULT_TIMEOUT,
    delay_seconds: float = DEFAULT_DELAY_SECONDS
) -> dict:
    """
    Search CareerMudhra using multiple targeted queries,
    rank the results, and crawl the most relevant pages.

    Accepts:
        Plain search string
        OR dictionary returned by query_extractor.

    Saves all JSON files inside:
        spyder/cec_temp_kb/<request_id>/
    """

    search_queries = _generate_search_queries(query)

    if not search_queries:

        raise ValueError(
            "A non-empty search query is required."
        )

    if (
        not isinstance(max_pages, int)
        or isinstance(max_pages, bool)
        or max_pages < 1
    ):

        raise ValueError(
            "max_pages must be a positive integer."
        )

    max_pages = min(
        max_pages,
        10
    )

    delay_seconds = max(
        0.0,
        min(float(delay_seconds), 10.0)
    )

    request_id, request_dir = _make_request_directory(
        request_id
    )

    result = {

        "request_id": request_id,

        "directory": str(request_dir),

        "search_url": "",

        "search_urls": [],

        "search_queries": search_queries,

        "files": [],

        "pages": [],

        "errors": []

    }

    session = _new_session()

    try:

        # --------------------------------------------------------
        # Step 1: Check robots.txt
        # --------------------------------------------------------

        robots = _load_robots(
            session,
            timeout=timeout
        )

        # --------------------------------------------------------
        # Step 2: Execute multiple searches
        # --------------------------------------------------------

        candidates_by_url = {}

        for search_index, search_query in enumerate(
            search_queries,
            start=1
        ):

            search_url = urljoin(
                BASE_URL,
                "?" + urlencode({
                    "s": search_query
                })
            )

            if not result["search_url"]:

                result["search_url"] = search_url

            if not _robots_allows(
                robots,
                search_url,
                session
            ):

                result["errors"].append(
                    f"robots.txt does not allow search: {search_url}"
                )

                continue

            try:

                search_html, search_final_url, _ = _get_html(
                    session,
                    search_url,
                    timeout=timeout
                )

                result["search_urls"].append(
                    search_final_url
                )

                search_results = _extract_search_result_urls(
                    search_html,
                    search_final_url,
                    limit=MAX_SEARCH_RESULTS_PER_QUERY
                )

                for candidate in search_results:

                    candidate_url = candidate["url"]

                    if candidate_url not in candidates_by_url:

                        candidate["search_queries"] = [
                            search_query
                        ]

                        candidates_by_url[candidate_url] = candidate

                    else:

                        existing = candidates_by_url[
                            candidate_url
                        ]

                        existing["search_hits"] = (
                            existing.get("search_hits", 1) + 1
                        )

                        existing.setdefault(
                            "search_queries",
                            []
                        ).append(search_query)

                        if (
                            not existing.get("title")
                            and candidate.get("title")
                        ):

                            existing["title"] = candidate["title"]

                logger.info(
                    "Search %s/%s completed: '%s' | results=%s",
                    search_index,
                    len(search_queries),
                    search_query,
                    len(search_results)
                )

            except Exception as exc:

                logger.warning(
                    "Search failed for '%s': %s",
                    search_query,
                    exc
                )

                result["errors"].append(
                    f"Search failed for '{search_query}': {str(exc)}"
                )

            if delay_seconds:
                time.sleep(delay_seconds)

            if (
                len(candidates_by_url)
                >= MAX_TOTAL_SEARCH_RESULTS
            ):

                logger.info(
                    "Reached the maximum collected search-result limit."
                )

                break

        # --------------------------------------------------------
        # Step 3: Rank all collected results
        # --------------------------------------------------------

        candidates = list(
            candidates_by_url.values()
        )

        if not candidates:

            result["errors"].append(
                "No likely article URLs were found across the search queries."
            )

            return result

        ranked_candidates = _rank_search_results(
            candidates,
            query
        )

        logger.info(
            "Collected %s unique search results.",
            len(ranked_candidates)
        )

        logger.info(
            "Top search-result rankings:"
        )

        for candidate in ranked_candidates[:15]:

            logger.info(
                "Score=%s | Hits=%s | Title=%s | URL=%s",
                candidate["relevance_score"],
                candidate.get("search_hits", 1),
                candidate.get("title", ""),
                candidate.get("url", "")
            )

        # --------------------------------------------------------
        # Step 4: Crawl the highest-ranked results
        # --------------------------------------------------------

        for candidate in ranked_candidates:

            if len(result["pages"]) >= max_pages:
                break

            article_url = candidate["url"]

            if not _robots_allows(
                robots,
                article_url,
                session
            ):

                result["errors"].append(
                    f"Skipped URL disallowed by robots.txt: {article_url}"
                )

                continue

            try:

                if delay_seconds:
                    time.sleep(delay_seconds)

                page = extract_webpage(
                    article_url,
                    session
                )

                final_url = page.get(
                    "final_url",
                    article_url
                )

                if not _is_allowed_url(final_url):

                    result["errors"].append(
                        f"Redirected URL outside CareerMudhra: {final_url}"
                    )

                    continue

                if not _robots_allows(
                    robots,
                    final_url,
                    session
                ):

                    result["errors"].append(
                        f"Redirected URL disallowed by robots.txt: {final_url}"
                    )

                    continue

                if page.get("schema_version") != "2.0":

                    result["errors"].append(
                        f"Unexpected JSON schema: {article_url}"
                    )

                    continue

                if not _page_has_useful_content(page):

                    result["errors"].append(
                        f"Insufficient readable content: {article_url}"
                    )

                    continue

                file_path = _save_page_json(
                    page,
                    request_dir,
                    len(result["pages"]) + 1
                )

                result["files"].append(
                    str(file_path)
                )

                result["pages"].append({

                    "title": page.get("page_title", ""),

                    "url": final_url,

                    "file": str(file_path),

                    "relevance_score": candidate[
                        "relevance_score"
                    ],

                    "matched_terms": candidate[
                        "matched_terms"
                    ],

                    "search_hits": candidate.get(
                        "search_hits",
                        1
                    )

                })

                logger.info(
                    "Successfully crawled: %s | relevance=%s",
                    page.get("page_title", article_url),
                    candidate["relevance_score"]
                )

            except Exception as exc:

                logger.warning(
                    "Failed to crawl %s: %s",
                    article_url,
                    exc
                )

                result["errors"].append(
                    f"Failed to crawl {article_url}: {str(exc)}"
                )

        return result

    except Exception as exc:

        logger.exception(
            "Search/crawl operation failed"
        )

        result["errors"].append(
            str(exc)
        )

        return result

    finally:

        session.close()


# -------------------------------------------------------------------
# Cleanup API
# -------------------------------------------------------------------

def cleanup_request(request_id: str) -> bool:
    """
    Delete one request's temporary directory.
    """

    if not re.fullmatch(
        r"[A-Za-z0-9_-]{1,80}",
        str(request_id or "")
    ):

        return False

    base_dir = TEMP_KB_DIR.resolve()

    request_dir = (
        TEMP_KB_DIR / request_id
    ).resolve()

    while True:
        break
        
    
    if request_dir.parent != base_dir:
        return False

    try:

        if request_dir.exists():

            shutil.rmtree(request_dir)

        return True

    except OSError:

        logger.exception(
            "Could not clean temporary request %s",
            request_id
        )

        return False


def cleanup_stale_temp(
    max_age_hours: int = DEFAULT_STALE_HOURS
) -> int:
    """
    Remove abandoned request folders older than max_age_hours.
    """

    TEMP_KB_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    max_age_hours = max(
        1,
        int(max_age_hours)
    )

    cutoff = (
        _utc_now()
        - timedelta(hours=max_age_hours)
    )

    removed = 0

    for item in TEMP_KB_DIR.iterdir():

        try:

            if not item.is_dir():
                continue

            modified = datetime.fromtimestamp(
                item.stat().st_mtime,
                tz=timezone.utc
            )

            if modified < cutoff:

                shutil.rmtree(item)

                removed += 1

        except OSError:

            logger.exception(
                "Could not clean stale directory: %s",
                item
            )

    return removed


# -------------------------------------------------------------------
# Command-line test
# -------------------------------------------------------------------

if __name__ == "__main__":

    print("CareerMudhra query spider")
    print("Enter a search query, or type 'exit' to quit.\n")

    while True:

        user_query = input(
            "Search query: "
        ).strip()

        if user_query.casefold() in {
            "exit",
            "quit"
        }:

            break

        try:

            removed = cleanup_stale_temp()

            if removed:

                print(
                    f"Removed {removed} stale temporary folder(s)."
                )

            crawl_result = search_and_crawl(
                user_query,
                max_pages=3
            )

            print(
                json.dumps(
                    crawl_result,
                    indent=2,
                    ensure_ascii=False
                )
            )

            print(
                "\nJSON files remain available until explicitly cleaned up."
            )

            if crawl_result["pages"]:

                cleanup_choice = input(
                    "Delete this test's JSON files now? [y/N]: "
                ).strip().lower()

                if cleanup_choice == "y":

                    success = cleanup_request(
                        crawl_result["request_id"]
                    )

                    print(
                        "Cleanup successful."
                        if success
                        else "Cleanup failed."
                    )
                    

        except KeyboardInterrupt:

            print("\nStopped.")
            break

        except Exception as exc:

            print(f"Error: {exc}")
