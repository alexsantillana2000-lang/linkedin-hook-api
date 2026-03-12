import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field, HttpUrl

APP_TITLE = "LinkedIn Hook Research API"
APP_VERSION = "1.0.0"

API_BEARER_TOKEN = os.getenv("API_BEARER_TOKEN", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "20"))
USER_AGENT = os.getenv(
    "USER_AGENT",
    "Mozilla/5.0 (compatible; HookResearchBot/1.0; +https://example.com/bot)"
)

app = FastAPI(
    title=APP_TITLE,
    version=APP_VERSION,
    description="Research public-web hook structures and normalize them into reusable templates."
)

bearer_scheme = HTTPBearer(auto_error=False)


def verify_bearer(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(bearer_scheme),
) -> None:
    if not API_BEARER_TOKEN:
        return

    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Missing bearer token")

    if credentials.credentials != API_BEARER_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid bearer token")


class ResearchHooksRequest(BaseModel):
    topic: str = Field(..., min_length=2)
    intent: str = Field(default="thought_leadership")
    audience: Optional[str] = None
    maxSources: int = Field(default=10, ge=1, le=50)
    allowedDomains: Optional[List[str]] = None
    blockedDomains: Optional[List[str]] = None
    includeExamples: bool = True
    includeTemplates: bool = True
    language: str = "en"


class AnalyzeUrlsRequest(BaseModel):
    urls: List[HttpUrl] = Field(..., min_length=1, max_length=20)
    extractMode: str = Field(default="hooks_examples_and_templates")
    language: str = "en"


def normalize_domain(domain: str) -> str:
    return domain.lower().replace("www.", "").strip()


def get_domain(url: str) -> str:
    return normalize_domain(urlparse(url).netloc)


def looks_like_blocked(url: str) -> bool:
    domain = get_domain(url)
    blocked = {
        "linkedin.com",
        "m.linkedin.com",
        "facebook.com",
        "instagram.com",
        "x.com",
        "twitter.com",
        "tiktok.com",
    }
    return domain in blocked


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def split_sentences(text: str) -> List[str]:
    text = clean_text(text)
    if not text:
        return []
    return [c.strip() for c in re.split(r"(?<=[.!?])\s+", text) if c.strip()]


def safe_truncate(text: str, n: int = 200) -> str:
    text = clean_text(text)
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


async def tavily_search(
    query: str,
    max_results: int,
    include_domains: Optional[List[str]] = None,
    exclude_domains: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    url = "https://api.tavily.com/search"
    payload: Dict[str, Any] = {
        "api_key": TAVILY_API_KEY,
        "query": query,
        "search_depth": "advanced",
        "max_results": max_results,
        "include_answer": False,
        "include_raw_content": False,
    }
    if include_domains:
        payload["include_domains"] = include_domains
    if exclude_domains:
        payload["exclude_domains"] = exclude_domains

    async with httpx.AsyncClient(
        timeout=REQUEST_TIMEOUT,
        headers={"User-Agent": USER_AGENT}
    ) as client:
        r = await client.post(url, json=payload)
        if r.status_code >= 400:
            raise HTTPException(status_code=500, detail=f"Tavily search failed: {r.text}")
        data = r.json()

    return [
        {
            "title": item.get("title") or "",
            "url": item.get("url") or "",
            "snippet": item.get("content") or "",
            "domain": get_domain(item.get("url") or ""),
        }
        for item in data.get("results", [])
    ]


async def fetch_html(url: str) -> str:
    async with httpx.AsyncClient(
        timeout=REQUEST_TIMEOUT,
        follow_redirects=True,
        headers={"User-Agent": USER_AGENT},
    ) as client:
        r = await client.get(url)
        r.raise_for_status()
        return r.text


def html_to_text(html: str) -> Dict[str, Any]:
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript", "svg", "header", "footer", "nav", "form"]):
        tag.decompose()

    title = clean_text(soup.title.get_text(" ", strip=True) if soup.title else "")
    meta_desc = ""
    meta = soup.find("meta", attrs={"name": "description"})
    if meta and meta.get("content"):
        meta_desc = clean_text(meta.get("content"))

    headings = []
    for tag_name in ["h1", "h2", "h3"]:
        for tag in soup.find_all(tag_name):
            txt = clean_text(tag.get_text(" ", strip=True))
            if txt:
                headings.append(txt)

    paragraphs = []
    for p in soup.find_all(["p", "li"]):
        txt = clean_text(p.get_text(" ", strip=True))
        if txt and len(txt.split()) >= 5:
            paragraphs.append(txt)

    return {
        "title": title,
        "meta_description": meta_desc,
        "headings": headings[:40],
        "paragraphs": paragraphs[:200],
        "body_text": "\n".join(paragraphs[:200]),
    }


HOOK_PATTERNS = {
    "Contrarian claim": [
        r"\b(everyone|most people|people)\b.*\b(but|wrong|actually|instead)\b",
        r"^stop\b",
        r"^don['’]?t\b",
    ],
    "Curiosity gap": [
        r"\bhere['’]s what\b",
        r"\bthe reason\b",
        r"\bno one tells you\b",
        r"\bwhat happened next\b",
    ],
    "Personal confession": [
        r"^i\b.*\b(used to|thought|believed|was wrong|learned)\b",
        r"^i made\b",
        r"^i tried\b",
    ],
    "Numbered lesson": [
        r"^\d+\s+(lessons?|mistakes?|ways?|rules?|truths?)\b",
        r"^\d+\b.*:",
    ],
    "Pain point": [
        r"\b(struggling|stuck|failing|overwhelmed|frustrated)\b",
    ],
    "Result / proof": [
        r"\b(i|we)\b.*\b(grew|hit|generated|made|earned|increased|cut|reduced)\b",
        r"\bfrom\b.*\bto\b",
    ],
    "Question hook": [
        r".*\?$",
        r"^(what|why|how|when)\b.*\?$",
    ],
}

GENERIC_TEMPLATES = {
    "Contrarian claim": "Everyone says [common belief], but [unexpected truth].",
    "Curiosity gap": "The reason [audience] struggle with [problem] is not [obvious cause]. It is [real cause].",
    "Personal confession": "I used to think [old belief] until [turning point].",
    "Numbered lesson": "[number] lessons I learned from [experience/result].",
    "Pain point": "If you are struggling with [problem], this is probably why.",
    "Result / proof": "We went from [before] to [after] by changing [specific thing].",
    "Question hook": "Why are [audience] still [undesired behavior] when [alternative] works better?",
    "Insight statement": "[unexpected observation]. Here is what that means for [audience].",
}


def sentence_score(sentence: str) -> float:
    s = sentence.strip()
    words = s.split()
    n = len(words)
    score = 0.0

    if 6 <= n <= 22:
        score += 2.0
    elif 4 <= n <= 30:
        score += 1.0

    if "?" in s:
        score += 1.0
    if ":" in s:
        score += 0.5

    lowered = s.lower()
    for cue in ["actually", "wrong", "truth", "mistake", "lesson", "stop", "don't", "used to", "learned", "why"]:
        if cue in lowered:
            score += 0.4

    if n > 30:
        score -= 1.0

    return score


def classify_structure(sentence: str) -> str:
    lowered = sentence.lower()
    for structure, patterns in HOOK_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, lowered, flags=re.IGNORECASE):
                return structure
    return "Insight statement"


def why_it_works(structure_name: str) -> str:
    reasons = {
        "Contrarian claim": "It interrupts scrolling by challenging common assumptions.",
        "Curiosity gap": "It creates an information gap the reader wants to close.",
        "Personal confession": "It builds trust through lived experience.",
        "Numbered lesson": "It promises concrete takeaways and structure.",
        "Pain point": "It signals relevance by naming a problem the reader already feels.",
        "Result / proof": "It increases credibility through evidence or outcomes.",
        "Question hook": "It invites the reader into an internal conversation.",
        "Insight statement": "It offers a sharp point of view in a compact format.",
    }
    return reasons.get(structure_name, "It captures attention quickly.")


def best_for(structure_name: str) -> List[str]:
    mapping = {
        "Contrarian claim": ["thought leadership", "opinion posts"],
        "Curiosity gap": ["educational posts", "story-led posts"],
        "Personal confession": ["personal brand", "founder stories"],
        "Numbered lesson": ["educational posts", "framework posts"],
        "Pain point": ["consulting", "service sales"],
        "Result / proof": ["case studies", "sales"],
        "Question hook": ["engagement", "discussion posts"],
        "Insight statement": ["thought leadership", "commentary"],
    }
    return mapping.get(structure_name, ["thought leadership"])


def extract_candidate_hooks(parsed: Dict[str, Any]) -> List[str]:
    candidates: List[str] = []

    for h in parsed.get("headings", []):
        if 4 <= len(h.split()) <= 20:
            candidates.append(h)

    intro_lines = []
    intro_lines.extend(parsed.get("paragraphs", [])[:10])
    intro_lines.extend(split_sentences(parsed.get("meta_description", "")))

    for line in intro_lines:
        line = clean_text(line)
        if 4 <= len(line.split()) <= 30:
            candidates.append(line)

    seen = set()
    deduped = []
    for c in candidates:
        k = c.lower()
        if k not in seen:
            seen.add(k)
            deduped.append(c)

    deduped.sort(key=sentence_score, reverse=True)
    return deduped[:12]


def normalize_structures(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for c in candidates:
        grouped[classify_structure(c["text"])].append(c)

    out = []
    for structure, items in grouped.items():
        items = sorted(items, key=lambda x: x["score"], reverse=True)
        sample_hooks = []
        source_urls = []
        seen = set()

        for item in items:
            hook = safe_truncate(item["text"], 180)
            key = hook.lower()
            if key not in seen:
                seen.add(key)
                sample_hooks.append(hook)
                source_urls.append(item["source_url"])
            if len(sample_hooks) >= 5:
                break

        out.append({
            "name": structure,
            "formula": GENERIC_TEMPLATES.get(structure, "[hook setup] + [specific insight]"),
            "whyItWorks": why_it_works(structure),
            "bestFor": best_for(structure),
            "sampleHooks": sample_hooks,
            "sourceUrls": list(dict.fromkeys(source_urls)),
            "count": len(items),
        })

    out.sort(key=lambda x: x["count"], reverse=True)
    return out


def structure_templates(structures: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    templates = []
    for s in structures:
        template = s["formula"]
        placeholders = re.findall(r"\[([^\]]+)\]", template)
        templates.append({
            "structureName": s["name"],
            "template": template,
            "placeholders": placeholders,
        })
    return templates


async def analyze_public_url(url: str) -> Dict[str, Any]:
    if looks_like_blocked(url):
        raise ValueError(f"Blocked or unsupported domain: {get_domain(url)}")

    html = await fetch_html(url)
    parsed = html_to_text(html)
    candidates = extract_candidate_hooks(parsed)

    return {
        "url": url,
        "title": parsed.get("title", ""),
        "snippet": safe_truncate(parsed.get("meta_description") or parsed.get("body_text", ""), 220),
        "candidates": [
            {
                "text": c,
                "score": sentence_score(c),
                "structure": classify_structure(c),
                "source_url": url,
            }
            for c in candidates
        ],
    }


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/analyze-urls")
async def analyze_urls(payload: AnalyzeUrlsRequest, _: None = Security(verify_bearer)) -> Dict[str, Any]:
    candidate_pool: List[Dict[str, Any]] = []
    warnings: List[str] = []
    analyzed_count = 0

    for url in payload.urls:
        try:
            result = await analyze_public_url(str(url))
            candidate_pool.extend(result["candidates"])
            analyzed_count += 1
        except Exception as e:
            warnings.append(f"{url}: {str(e)}")

    structures = normalize_structures(candidate_pool)
    templates = structure_templates(structures)

    response: Dict[str, Any] = {
        "analyzedCount": analyzed_count,
        "structures": [
            {
                "name": s["name"],
                "formula": s["formula"],
                "sampleHooks": s["sampleHooks"],
                "sourceUrls": s["sourceUrls"],
            }
            for s in structures
        ],
    }

    if payload.extractMode == "hooks_examples_and_templates":
        response["templates"] = templates
    elif payload.extractMode == "hooks_and_examples":
        response["templates"] = []

    if warnings:
        response["warnings"] = warnings

    return response


@app.post("/research-hooks")
async def research_hooks(payload: ResearchHooksRequest, _: None = Security(verify_bearer)) -> Dict[str, Any]:
    if not TAVILY_API_KEY:
        raise HTTPException(status_code=500, detail="TAVILY_API_KEY is not configured")

    include_domains = [normalize_domain(d) for d in (payload.allowedDomains or [])]
    exclude_domains = [normalize_domain(d) for d in (payload.blockedDomains or [])]
    if "linkedin.com" not in exclude_domains:
        exclude_domains.append("linkedin.com")

    query = f'{payload.topic} LinkedIn hook examples OR post hooks OR copywriting hooks'
    if payload.audience:
        query += f" for {payload.audience}"

    results = await tavily_search(
        query=query,
        max_results=payload.maxSources,
        include_domains=include_domains or None,
        exclude_domains=exclude_domains or None,
    )

    candidate_pool: List[Dict[str, Any]] = []
    warnings: List[str] = []
    sources: List[Dict[str, str]] = []

    for item in results:
        url = item["url"]
        if not url:
            continue

        try:
            analyzed = await analyze_public_url(url)
            candidate_pool.extend(analyzed["candidates"])
            sources.append({
                "title": item.get("title") or analyzed.get("title") or "",
                "url": url,
                "domain": item.get("domain") or get_domain(url),
                "snippet": safe_truncate(item.get("snippet") or analyzed.get("snippet") or "", 220),
            })
        except Exception as e:
            warnings.append(f"{url}: {str(e)}")

    structures = normalize_structures(candidate_pool)
    templates = structure_templates(structures)

    summary = (
        f'Top patterns for "{payload.topic}"'
        + (f" targeting {payload.audience}" if payload.audience else "")
        + " emphasize tension, specificity, and fast curiosity."
    )

    response: Dict[str, Any] = {
        "topic": payload.topic,
        "intent": payload.intent,
        "audience": payload.audience,
        "summary": summary,
        "structures": [
            {
                "name": s["name"],
                "formula": s["formula"],
                "whyItWorks": s["whyItWorks"],
                "bestFor": s["bestFor"],
                "sampleHooks": s["sampleHooks"] if payload.includeExamples else [],
            }
            for s in structures
        ],
        "sources": sources,
        "warnings": warnings,
    }

    if payload.includeTemplates:
        response["templates"] = templates

    return response
