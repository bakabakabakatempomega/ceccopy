
"""
query_spider.py

Search CareerMudhra once, collect the top 10 website search results,
rank those results using the existing relevance-ranking system,
and crawl the top 3 relevant pages into temporary JSON files.

Location:
    CEC/spyder/query_spider.py

Crawler:
    CEC/spyder/cec_crawler_v3.py

Temporary output:
    CEC/spyder/cec_temp_kb/<request_id>/

All saved JSON files use schema_version 3.0.

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
from urllib.parse import urljoin, urlparse, urlunparse, urlencode, parse_qsl
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup


# -------------------------------------------------------------------
# Crawler import
# -------------------------------------------------------------------

try:
    from .cec_crawler_v3 import create_session, extract_webpage
except ImportError:
    from cec_crawler_v3 import create_session, extract_webpage


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

SPYDER_DIR = Path(__file__).resolve().parent
TEMP_KB_DIR = SPYDER_DIR / "cec_temp_kb"

BASE_URL = "https://careermudhra.com/"
SITE_HOST = "careermudhra.com"
ROBOTS_URL = urljoin(BASE_URL, "robots.txt")

DEFAULT_MAX_PAGES = 3
DEFAULT_TIMEOUT = (10, 30)
DEFAULT_DELAY_SECONDS = 1.0
DEFAULT_MAX_HTML_BYTES = 5 * 1024 * 1024
DEFAULT_STALE_HOURS = 2

# Exactly 10 website search results are collected.
MAX_SEARCH_RESULTS = 10

# At most 3 relevant pages are crawled.
MAX_CRAWL_PAGES = 3

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Common functions
# -------------------------------------------------------------------

def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _clean_query(value) -> str:
    """Convert strings, lists or other values into normalized text."""

    if isinstance(value, (list, tuple, set)):
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
    """Extract normalized words and remove common conversational words."""

    stopwords = {
        "a", "an", "the", "of", "in", "at",
        "and", "or", "for", "to", "by",
        "with", "on", "from", "about",
        "tell", "me", "what", "which",
        "who", "where", "when", "how",
        "can", "could", "would", "please",
        "give", "show", "explain", "describe",
        "needed", "required", "is", "are",
        "was", "were", "it", "this", "that"
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
    """Recursively extract readable strings from nested JSON data."""

    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []

    if isinstance(value, (int, float)):
        return [str(value)]

    if isinstance(value, list):
        result = []

        for item in value:
            result.extend(_flatten_text(item))

        return result

    if isinstance(value, dict):
        result = []

        ignored_keys = {
            "section_id",
            "heading_key",
            "heading_level",
            "table_id",
            "table_index",
            "table_header"
        }

        for key, item in value.items():
            if key in ignored_keys:
                continue

            result.extend(_flatten_text(item))

        return result

    return []


# -------------------------------------------------------------------
# Query data preparation
# -------------------------------------------------------------------

def _prepare_query_data(query) -> dict:
    """
    Normalize query input for the ranking system.

    Supports:
        1. Dictionary returned by query_extractor.py.
        2. Plain text entered directly.
    """

    if isinstance(query, dict):

        return {
            "original_question": _clean_query(
                query.get("original_question")
            ),

            "search_query": _clean_query(
                query.get("search_query")
                or query.get("original_question")
            ),

            "college_names": [
                _clean_query(item)
                for item in query.get("college_names", [])
                if _clean_query(item)
            ],

            "degree_names": [
                _clean_query(item)
                for item in query.get("degree_names", [])
                if _clean_query(item)
            ],

            "fields": [
                _clean_query(item)
                for item in query.get("fields", [])
                if _clean_query(item)
            ],

            "requested_data": [
                _clean_query(item)
                for item in query.get("requested_data", [])
                if _clean_query(item)
            ],

            "locations": [
                _clean_query(item)
                for item in query.get("locations", [])
                if _clean_query(item)
            ],

            "keywords": [
                _clean_query(item)
                for item in query.get("keywords", [])
                if _clean_query(item)
            ],

            "raw_query": _clean_query(
                query.get("search_query")
                or query.get("original_question")
            )
        }

    raw_query = _clean_query(query)

    return {
        "original_question": raw_query,
        "search_query": raw_query,
        "college_names": [],
        "degree_names": [],
        "fields": [],
        "requested_data": [],
        "locations": [],
        "keywords": [],
        "raw_query": raw_query
    }


# -------------------------------------------------------------------
# Search query generation
# -------------------------------------------------------------------

def _generate_search_queries(query) -> list[str]:
    """
    Generate exactly one search query.

    The extracted search_query is preferred.
    The original question is used as a fallback.

    No additional searches are generated.
    """

    data = _prepare_query_data(query)

    search_query = (
        data["search_query"]
        or data["original_question"]
        or data["raw_query"]
    )

    search_query = _clean_query(search_query)

    return [search_query] if search_query else []


# -------------------------------------------------------------------
# URL and filesystem safety
# -------------------------------------------------------------------

def _is_allowed_url(url: str) -> bool:
    """Permit only HTTPS URLs on careermudhra.com or its subdomains."""

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
    """Remove fragments and common tracking parameters."""

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

        query_items.append((key, value))

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
        raise ValueError("Invalid request_id.")

    base_dir = TEMP_KB_DIR.resolve()

    request_dir = (
        TEMP_KB_DIR / request_id
    ).resolve()

    if request_dir.parent != base_dir:
        raise ValueError("Invalid temporary directory path.")

    request_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    return request_id, request_dir


# -------------------------------------------------------------------
# HTTP session and robots.txt
# -------------------------------------------------------------------

def _new_session() -> requests.Session:
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

        content_length = response.headers.get("Content-Length")

        if content_length:
            try:
                if int(content_length) > max_bytes:
                    raise ValueError(
                        "HTML response exceeds the configured size limit."
                    )
            except (TypeError, ValueError) as exc:
                if "exceeds the configured size limit" in str(exc):
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
    limit: int = MAX_SEARCH_RESULTS
) -> list[dict]:
    """
    Extract the first 10 likely article links and their visible titles
    from the CareerMudhra website search-results page.
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

        if not _is_probable_article_url(normalized_url):
            continue

        if normalized_url == _normalize_url(search_page_url):
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

    Existing ranking weights are preserved.
    """

    data = _prepare_query_data(query_data)

    title = _clean_query(
        candidate.get("title")
    )

    url = _clean_query(
        candidate.get("url")
    )

    parsed_url = urlparse(url)

    url_text = (
        parsed_url.path
        .replace("-", " ")
        .replace("/", " ")
    )

    searchable_text = (
        title + " " + url_text
    ).casefold()

    result_tokens = _tokenize(searchable_text)

    score = 0
    matched_terms = []

    colleges = data["college_names"]
    degrees = data["degree_names"]
    fields = data["fields"]
    requested_data = data["requested_data"]
    locations = data["locations"]

    # ---------------------------------------------------------------
    # 1. College name matching
    # ---------------------------------------------------------------

    college_match_found = False

    for college in colleges:

        college = _clean_query(college)

        if not college:
            continue

        college_tokens = _tokenize(college)
        normalized_college = college.casefold()

        if normalized_college in searchable_text:

            score += 120
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

                score += int(90 * coverage)
                college_match_found = True
                matched_terms.append(college)

            elif matching_tokens:

                score += int(20 * coverage)

    # ---------------------------------------------------------------
    # 2. Degree matching
    # ---------------------------------------------------------------

    for degree in degrees:

        degree = _clean_query(degree)

        if not degree:
            continue

        degree_tokens = _tokenize(degree)

        if degree.casefold() in searchable_text:

            score += 25
            matched_terms.append(degree)

        elif degree_tokens:

            matching_tokens = (
                degree_tokens & result_tokens
            )

            if matching_tokens:

                score += int(
                    20 * len(matching_tokens)
                    / len(degree_tokens)
                )

    # ---------------------------------------------------------------
    # 3. Specialization / field matching
    # ---------------------------------------------------------------

    for field in fields:

        field = _clean_query(field)

        if not field:
            continue

        field_tokens = _tokenize(field)

        if field.casefold() in searchable_text:

            score += 35
            matched_terms.append(field)

        elif field_tokens:

            matching_tokens = (
                field_tokens & result_tokens
            )

            if matching_tokens:

                score += int(
                    25 * len(matching_tokens)
                    / len(field_tokens)
                )

    # ---------------------------------------------------------------
    # 4. Requested information matching
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

            score += 8 * len(matching_tokens)

    # ---------------------------------------------------------------
    # 5. Location matching
    # ---------------------------------------------------------------

    for location in locations:

        location = _clean_query(location)

        if not location:
            continue

        location_tokens = _tokenize(location)

        if location.casefold() in searchable_text:

            score += 12
            matched_terms.append(location)

        elif location_tokens:

            matching_tokens = (
                location_tokens & result_tokens
            )

            if matching_tokens:

                score += int(
                    10 * len(matching_tokens)
                    / len(location_tokens)
                )

    # ---------------------------------------------------------------
    # 6. Plain-text query matching
    # ---------------------------------------------------------------

    if not colleges and not degrees and not fields and not requested_data:

        raw_query = data["raw_query"]

        query_tokens = _tokenize(raw_query)

        generic_tokens = {
            "college",
            "colleges",
            "fee",
            "fees",
            "admission",
            "admissions",
            "course",
            "courses",
            "structure",
            "details",
            "information",
            "review"
        }

        matched_tokens = query_tokens & result_tokens

        for token in matched_tokens:

            if token in generic_tokens:
                score += 1

            elif token in {
                "east",
                "point",
                "radiology",
                "radiodiagnosis",
                "nri",
                "management",
                "quota"
            }:
                score += 5

            elif token in {
                "medical",
                "engineering",
                "dental",
                "mbbs",
                "md",
                "ms",
                "pg",
                "ug"
            }:
                score += 3

            else:
                score += 2

            matched_terms.append(token)

        normalized_query = re.sub(
            r"[^a-z0-9]+",
            " ",
            raw_query.casefold()
        ).strip()

        normalized_title = re.sub(
            r"[^a-z0-9]+",
            " ",
            searchable_text
        ).strip()

        if normalized_query and normalized_query in normalized_title:
            score += 20

    # ---------------------------------------------------------------
    # 7. Penalize unrelated colleges
    # ---------------------------------------------------------------

    if colleges and not college_match_found:
        score -= 80

    # ---------------------------------------------------------------
    # 8. Search-hit bonus
    # ---------------------------------------------------------------

    search_hits = candidate.get(
        "search_hits",
        1
    )

    score += min(
        max(search_hits - 1, 0) * 3,
        12
    )

    matched_terms = list(dict.fromkeys(matched_terms))

    return score, matched_terms


def _rank_search_results(
    candidates: list[dict],
    query_data
) -> list[dict]:
    """Rank the collected search results from most to least relevant."""

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
# Page validation and saving
# -------------------------------------------------------------------

def _page_has_useful_content(page: dict) -> bool:
    """
    Validate a schema 3.0 page.

    Expected primary fields:
        schema_version
        title
        subtitles
        content
        tables
    """

    if not isinstance(page, dict):
        return False

    if page.get("schema_version") != "3.0":
        return False

    title = page.get("title", "")

    if not isinstance(title, str) or not title.strip():
        return False

    subtitles = page.get("subtitles", [])
    content = page.get("content", {})
    tables = page.get("tables", {})

    if not isinstance(subtitles, list):
        return False

    if not isinstance(content, dict):
        return False

    if not isinstance(tables, (dict, list)):
        return False

    readable_parts = []

    readable_parts.extend(_flatten_text(title))
    readable_parts.extend(_flatten_text(subtitles))
    readable_parts.extend(_flatten_text(content))
    readable_parts.extend(_flatten_text(tables))

    readable_text = " ".join(readable_parts).strip()

    return len(readable_text) >= 100


def _save_page_json(
    page: dict,
    request_dir: Path,
    page_number: int
) -> Path:

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
    Search CareerMudhra exactly once.

    Workflow:

        1. Prepare one search query.
        2. Search the CareerMudhra website.
        3. Collect the first 10 search results.
        4. Rank those 10 results using the existing ranking system.
        5. Crawl the top 3 relevant pages.
        6. Save successfully crawled pages as JSON.

    Accepts:
        Plain search string
        OR dictionary returned by query_extractor.py

    Saves JSON files inside:
        spyder/cec_temp_kb/<request_id>/
    """

    query_data = _prepare_query_data(query)

    search_queries = _generate_search_queries(query_data)

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

    # Never crawl more than 3 pages.
    max_pages = min(
        max_pages,
        MAX_CRAWL_PAGES
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
        "search_results": [],
        "ranked_results": [],
        "files": [],
        "pages": [],
        "errors": []
    }

    session = _new_session()

    try:

        # ------------------------------------------------------------
        # Step 1: Check robots.txt
        # ------------------------------------------------------------

        robots = _load_robots(
            session,
            timeout=timeout
        )

        # ------------------------------------------------------------
        # Step 2: Search the website ONCE
        # ------------------------------------------------------------

        search_query = search_queries[0]

        search_url = urljoin(
            BASE_URL,
            "?" + urlencode({
                "s": search_query
            })
        )

        result["search_url"] = search_url

        if not _robots_allows(
            robots,
            search_url,
            session
        ):

            result["errors"].append(
                f"robots.txt does not allow search: {search_url}"
            )

            return result

        logger.info(
            "Searching CareerMudhra once: '%s'",
            search_query
        )

        search_html, search_final_url, _ = _get_html(
            session,
            search_url,
            timeout=timeout
        )

        result["search_urls"].append(
            search_final_url
        )

        if delay_seconds:
            time.sleep(delay_seconds)

        # ------------------------------------------------------------
        # Step 3: Collect the top 10 website search results
        # ------------------------------------------------------------

        candidates = _extract_search_result_urls(
            search_html,
            search_final_url,
            limit=MAX_SEARCH_RESULTS
        )

        logger.info(
            "Collected %s search results (maximum %s).",
            len(candidates),
            MAX_SEARCH_RESULTS
        )

        result["search_results"] = candidates

        if not candidates:

            result["errors"].append(
                "No likely article URLs were found on the search-results page."
            )

            return result

        # ------------------------------------------------------------
        # Step 4: Rank the 10 results using the existing ranking system
        # ------------------------------------------------------------

        ranked_candidates = _rank_search_results(
            candidates,
            query_data
        )

        result["ranked_results"] = ranked_candidates

        logger.info(
            "Ranking query data: %s",
            json.dumps(
                {
                    "college_names": query_data["college_names"],
                    "degree_names": query_data["degree_names"],
                    "fields": query_data["fields"],
                    "requested_data": query_data["requested_data"],
                    "locations": query_data["locations"],
                    "raw_query": query_data["raw_query"]
                },
                ensure_ascii=False
            )
        )

        logger.info("Ranked search results:")

        for rank, candidate in enumerate(
            ranked_candidates,
            start=1
        ):

            logger.info(
                "Rank=%s | Score=%s | Matches=%s | Title=%s | URL=%s",
                rank,
                candidate["relevance_score"],
                candidate["matched_terms"],
                candidate.get("title", ""),
                candidate.get("url", "")
            )

        # ------------------------------------------------------------
        # Step 5: Crawl the top 3 relevant results
        # ------------------------------------------------------------

        for rank, candidate in enumerate(
            ranked_candidates,
            start=1
        ):

            if len(result["pages"]) >= max_pages:
                break

            article_url = candidate["url"]

            if candidate["relevance_score"] <= 0:

                logger.info(
                    "Skipping low-relevance result: %s | score=%s",
                    candidate.get("title", article_url),
                    candidate["relevance_score"]
                )

                continue

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

                if not isinstance(page, dict):
                    raise ValueError(
                        "Crawler returned an invalid page object."
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

                if page.get("schema_version") != "3.0":

                    result["errors"].append(
                        f"Unexpected JSON schema for: {article_url}"
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

                    "title": page.get("title", ""),

                    "url": final_url,

                    "file": str(file_path),

                    "search_rank": rank,

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
                    "Successfully crawled: %s | search_rank=%s | relevance=%s | matches=%s",
                    page.get("title", article_url),
                    rank,
                    candidate["relevance_score"],
                    candidate["matched_terms"]
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

        logger.info(
            "Search and crawl completed: "
            "search_results=%s | ranked_results=%s | crawled_pages=%s",
            len(result["search_results"]),
            len(result["ranked_results"]),
            len(result["pages"])
        )

        return result

    except Exception as exc:

        logger.exception(
            "Search/crawl operation failed"
        )

        result["errors"].append(str(exc))

        return result

    finally:
        session.close()


# -------------------------------------------------------------------
# Cleanup API
# -------------------------------------------------------------------

def cleanup_request(request_id: str) -> bool:
    """Delete one request's temporary directory."""

    if not re.fullmatch(
        r"[A-Za-z0-9_-]{1,80}",
        str(request_id or "")
    ):
        return False

    base_dir = TEMP_KB_DIR.resolve()

    request_dir = (
        TEMP_KB_DIR / request_id
    ).resolve()

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
    """Remove abandoned request folders older than max_age_hours."""

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
