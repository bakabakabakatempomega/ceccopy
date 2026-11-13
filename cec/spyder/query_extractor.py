
"""
query_extractor.py

Extract structured search terms from a user's question using Groq.

Location:
    CEC/spyder/query_extractor.py

Dependencies:
    pip install groq python-dotenv
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from groq import Groq


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

# Project layout:
# CEC/
#   .env
#   spyder/
#       query_extractor.py

PROJECT_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_DIR / ".env"

load_dotenv(ENV_FILE)

GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")


def _get_client() -> Groq:
    """Create a Groq client using the project's environment variables."""
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            f"GROQ_API_KEY was not found. Check your .env file at: {ENV_FILE}"
        )

    return Groq(api_key=api_key)


def _clean_string(value: Any) -> str:
    """Convert a value to a safe, trimmed string."""
    if value is None:
        return ""

    if not isinstance(value, str):
        value = str(value)

    return re.sub(r"\s+", " ", value).strip()


def _clean_list(value: Any, limit: int = 10) -> list[str]:
    """Normalize a model-generated list into unique strings."""
    if isinstance(value, str):
        value = [value]

    if not isinstance(value, list):
        return []

    results = []
    seen = set()

    for item in value:
        item = _clean_string(item)

        if not item:
            continue

        key = item.casefold()

        if key not in seen:
            seen.add(key)
            results.append(item)

        if len(results) >= limit:
            break

    return results


def _fallback_result(question: str) -> dict:
    """
    Provide a basic result when the Groq call fails or returns invalid JSON.

    This fallback does not try to guess college names or degrees. It uses
    the user's question itself as the search query.
    """
    question = _clean_string(question)

    # Remove common question words for a compact keyword list.
    stopwords = {
        "a", "an", "and", "are", "at", "can", "do", "does", "for",
        "from", "give", "how", "i", "in", "is", "it", "me", "of",
        "on", "please", "show", "tell", "the", "their", "there",
        "to", "what", "which", "with", "would", "you", "your",
    }

    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9.+&'-]*", question)
    keywords = []
    seen = set()

    for word in words:
        normalized = word.casefold()

        if normalized in stopwords or len(normalized) < 2:
            continue

        if normalized not in seen:
            seen.add(normalized)
            keywords.append(word)

    return {
        "original_question": question,
        "degree_names": [], 
        "college_names": [],
        "fields": [],
        "locations": [],
        "requested_data": [],
        "keywords": keywords[:15],
        "search_query": question,
        "used_fallback": True,
    }


# -------------------------------------------------------------------
# Main extraction function
# -------------------------------------------------------------------

def extract_keywords(
    question: str,
    conversation_context: Optional[str] = None,
    client: Optional[Groq] = None,
) -> dict:
    """
    Extract search terms from a user question.

    Args:
        question:
            The latest user message.

        conversation_context:
            Optional short conversation history. Useful for follow-ups
            such as "What about its fees?"

        client:
            Optional existing Groq client, useful for testing or reuse.

    Returns:
        A dictionary with:
            original_question
            degree_names
            college_names
            fields
            locations
            requested_data
            keywords
            search_query
            used_fallback

    Raises:
        ValueError: if the question is empty.
        RuntimeError: if Groq is unavailable and no fallback is possible.
    """
    question = _clean_string(question)

    if not question:
        raise ValueError("The question cannot be empty.")

    context = _clean_string(conversation_context)

    if client is None:
        client = _get_client()

    system_prompt = """
You are a search-query extraction component for CareerMudhra,
an education information website.

Your task is to identify information explicitly requested by the user
and construct a concise search query for searching:
https://careermudhra.com/?s=

Return ONLY a valid JSON object with these keys:

{
  "degree_names": [],
  "college_names": [],
  "fields": [],
  "locations": [],
  "requested_data": [],
  "keywords": [],
  "search_query": ""
}

Rules:
1. Do not answer the user's question.
2. Do not invent college names, degrees, locations, fees, or requirements.
3. Extract names only when stated in the user's message or clearly
   established by the supplied conversation context.
4. degree_names may include MBBS, BDS, MD, MS, MDS, BAMS, BPT, MBA,
   nursing degrees, and other academic qualifications.
5. fields may include specializations, disciplines, or topics such as
   cardiology, dentistry, management quota, or medical admissions.
6. requested_data should identify the type of information requested,
   such as fees, eligibility, admission process, cutoff, seats, rankings,
   placements, hostel fees, or documents required.
7. keywords should contain useful search terms, without generic
   conversational words.
8. search_query should be a short, natural phrase suitable for a
   website search box. Prefer roughly 2-8 meaningful words when possible.
9. Resolve pronouns from the conversation context only when the
   reference is clear. If it is unclear, preserve the ambiguity rather
   than guessing.
10. Use empty arrays for categories with no identified values.
11. Output valid JSON only. Do not use Markdown fences.
"""

    user_payload = {
        "conversation_context": context,
        "latest_question": question,
    }

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(user_payload, ensure_ascii=False),
                },
            ],
            temperature=0,
            max_tokens=700,
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content or "{}"
        data = json.loads(content)

        if not isinstance(data, dict):
            raise ValueError("The model response was not a JSON object.")

        degree_names = _clean_list(data.get("degree_names"))
        college_names = _clean_list(data.get("college_names"))
        fields = _clean_list(data.get("fields"))
        locations = _clean_list(data.get("locations"))
        requested_data = _clean_list(data.get("requested_data"))
        keywords = _clean_list(data.get("keywords"), limit=15)
        search_query = _clean_string(data.get("search_query"))

        # Never send an empty query to the search spider.
        if not search_query:
            search_query = question

        # Add important extracted entities if the model omitted them
        # from its search query.
        query_parts = [search_query]

        for item in degree_names + college_names + fields + locations:
            if item.casefold() not in search_query.casefold():
                query_parts.append(item)

        search_query = " ".join(query_parts)
        search_query = re.sub(r"\s+", " ", search_query).strip()

        # Keep the query at a manageable length.
        search_query = search_query[:250]

        return {
            "original_question": question,
            "degree_names": degree_names,
            "college_names": college_names,
            "fields": fields,
            "locations": locations,
            "requested_data": requested_data,
            "keywords": keywords,
            "search_query": search_query,
            "used_fallback": False,
        }

    except Exception as exc:
        # Keep the application usable even if keyword extraction fails.
        # The caller can log this flag and decide whether to retry.
        result = _fallback_result(question)
        result["extraction_error"] = str(exc)
        return result


if __name__ == "__main__":
    print("CareerMudhra query extractor")
    print("Enter a question, or type 'exit' to quit.\n")

    while True:
        user_question = input("Question: ").strip()

        if user_question.casefold() in {"exit", "quit"}:
            break

        try:
            result = extract_keywords(user_question)
            print(json.dumps(result, indent=2, ensure_ascii=False))
        except Exception as exc:
            print(f"Error: {exc}")
