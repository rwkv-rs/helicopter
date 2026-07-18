from __future__ import annotations

import base64
import gzip
import html
import ipaddress
import json
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlencode, unquote, urlparse
from urllib.request import Request, urlopen

from .protocol import Action


USER_AGENT = "rwkv-web-harness/0.1 (+https://github.com/rwkv/helicopter)"


@dataclass(frozen=True)
class Source:
    source_id: str
    title: str
    url: str
    snippet: str = ""


@dataclass(frozen=True)
class Page:
    source_id: str
    title: str
    url: str
    text: str


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    tool: str
    message: str
    data: dict[str, Any] | None = None

    def observation(self) -> str:
        payload = {"ok": self.ok, "tool": self.tool, "message": self.message}
        if self.data:
            payload["data"] = self.data
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)


class WebToolkit:
    """Read-only search and web fetching tools with no vendor API dependency."""

    def __init__(
        self,
        *,
        search_url: str | None = None,
        search_backend: str = "searxng",
        timeout: float = 20.0,
        max_page_chars: int = 6000,
        user_agent: str = USER_AGENT,
    ) -> None:
        if search_backend not in {"searxng", "html"}:
            raise ValueError("search_backend must be 'searxng' or 'html'")
        self.search_url = search_url or "https://lite.duckduckgo.com/lite/"
        self.search_backend = search_backend
        self.timeout = timeout
        self.max_page_chars = max_page_chars
        self.user_agent = user_agent
        self.sources: dict[str, Source] = {}
        self.pages: dict[str, Page] = {}
        self._next_source = 1

    @property
    def tool_descriptions(self) -> str:
        return (
            "Available tools:\n"
            '- web_search: {"query": string, "top_k": integer}\n'
            '- open_url: {"source_id": string} or {"url": string}\n'
            '- find_in_page: {"source_id": string, "pattern": string}\n'
            "Only use these read-only tools. Cite source ids in the final answer."
        )

    @property
    def tool_schemas(self) -> list[dict[str, Any]]:
        """OpenAI tool schemas consumed by vLLM-RWKV's native tool parser."""

        return [
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the public web and return source ids, titles, URLs, and snippets.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "The search query."},
                            "top_k": {"type": "integer", "minimum": 1, "maximum": 10},
                        },
                        "required": ["query"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "open_url",
                    "description": "Open a public search result or URL and extract readable page text.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "source_id": {"type": "string"},
                            "url": {"type": "string"},
                        },
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "find_in_page",
                    "description": "Find a phrase in a page that was already opened.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "source_id": {"type": "string"},
                            "pattern": {"type": "string"},
                        },
                        "required": ["source_id", "pattern"],
                        "additionalProperties": False,
                    },
                },
            },
        ]

    @property
    def g1h_tool_catalog(self) -> str:
        """Flat JSON catalog used in the g1h prompt instead of native chat tools."""

        catalog: list[dict[str, Any]] = []
        for schema in self.tool_schemas:
            function = schema.get("function", {})
            if isinstance(function, dict):
                catalog.append(
                    {
                        "name": function.get("name", ""),
                        "description": function.get("description", ""),
                        "parameters": function.get("parameters", {}),
                    }
                )
        catalog.append(
            {
                "name": "final_answer",
                "description": "Finish the task using the gathered evidence.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "answer": {"type": "string"},
                        "citations": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["answer"],
                    "additionalProperties": False,
                },
            }
        )
        return "Tools:\n" + json.dumps(catalog, ensure_ascii=False, separators=(",", ":"))

    def execute(self, action: Action) -> ToolResult:
        try:
            if action.name == "web_search":
                return self._search(action.arguments)
            if action.name == "open_url":
                return self._open_url(action.arguments)
            if action.name == "find_in_page":
                return self._find_in_page(action.arguments)
            return ToolResult(False, action.name, f"unknown tool: {action.name}")
        except (ValueError, WebToolError) as exc:
            return ToolResult(False, action.name, str(exc))

    def _search(self, arguments: dict[str, Any]) -> ToolResult:
        query = arguments.get("query")
        top_k = arguments.get("top_k", 5)
        if not isinstance(query, str) or not query.strip():
            raise ValueError("web_search requires a non-empty query")
        if not isinstance(top_k, int):
            raise ValueError("web_search top_k must be an integer")
        top_k = min(max(top_k, 1), 10)
        params = {"q": query.strip()}
        if self.search_backend == "searxng":
            params["format"] = "json"
        errors: list[str] = []
        results: list[Source] = []
        provider = self.search_url
        for provider_url in self._search_candidates():
            try:
                raw = _http_get(provider_url, params=params, timeout=self.timeout, user_agent=self.user_agent)
                if self.search_backend == "searxng" and provider_url == self.search_url:
                    results = _parse_searxng_results(raw, top_k)
                    if not results:
                        results = _parse_html_results(raw, top_k)
                else:
                    results = _parse_html_results(raw, top_k)
            except WebToolError as exc:
                errors.append(str(exc))
                continue
            if results:
                provider = provider_url
                break
        if not results:
            if errors:
                raise WebToolError(
                    "all search providers failed: "
                    + " | ".join(errors[:3])
                )
            return ToolResult(True, "web_search", "no search results", {"query": query, "results": []})
        normalized_results: list[Source] = []
        for result in results:
            source = Source(self._allocate_source_id(), result.title, result.url, result.snippet)
            self.sources[source.source_id] = source
            normalized_results.append(source)
        formatted = [
            {
                "source_id": item.source_id,
                "title": item.title,
                "url": item.url,
                "snippet": item.snippet,
            }
            for item in normalized_results
        ]
        return ToolResult(
            True,
            "web_search",
            f"found {len(normalized_results)} results",
            {"query": query, "provider": provider, "results": formatted},
        )

    def _search_candidates(self) -> list[str]:
        if self.search_backend == "searxng":
            return [self.search_url]
        candidates = [
            self.search_url,
            "https://html.duckduckgo.com/html/",
            "https://www.google.com/search",
            "https://www.bing.com/search",
        ]
        return list(dict.fromkeys(candidates))

    def _open_url(self, arguments: dict[str, Any]) -> ToolResult:
        source_id = arguments.get("source_id")
        url = arguments.get("url")
        source: Source | None = None
        if isinstance(source_id, str):
            source = self.sources.get(source_id)
            if source is None:
                raise ValueError(f"unknown source_id: {source_id}")
            url = source.url
        if not isinstance(url, str) or not url.strip():
            raise ValueError("open_url requires source_id or url")
        url = url.strip()
        _validate_public_url(url)
        raw = _http_get(url, timeout=self.timeout, user_agent=self.user_agent)
        title, text = _extract_html_text(raw, url)
        if not text:
            text = raw.decode("utf-8", errors="replace")
        text = _clean_text(text)[: self.max_page_chars]
        page_source_id = source.source_id if source else self._allocate_source_id()
        page = Page(source_id=page_source_id, title=title or (source.title if source else url), url=url, text=text)
        self.pages[page_source_id] = page
        return ToolResult(
            True,
            "open_url",
            f"opened {url}",
            {"source_id": page_source_id, "title": page.title, "url": url, "content": text},
        )

    def _find_in_page(self, arguments: dict[str, Any]) -> ToolResult:
        source_id = arguments.get("source_id")
        pattern = arguments.get("pattern")
        if not isinstance(source_id, str) or not source_id:
            raise ValueError("find_in_page requires source_id")
        if not isinstance(pattern, str) or not pattern.strip():
            raise ValueError("find_in_page requires a non-empty pattern")
        page = self.pages.get(source_id)
        if page is None:
            raise ValueError(f"source {source_id} has not been opened")
        text = page.text
        match = re.search(re.escape(pattern.strip()), text, flags=re.IGNORECASE)
        if not match:
            return ToolResult(True, "find_in_page", "pattern not found", {"source_id": source_id, "matches": []})
        start = max(0, match.start() - 500)
        end = min(len(text), match.end() + 1000)
        return ToolResult(
            True,
            "find_in_page",
            "pattern found",
            {"source_id": source_id, "matches": [text[start:end]]},
        )

    def _allocate_source_id(self) -> str:
        source_id = f"source_{self._next_source:03d}"
        self._next_source += 1
        return source_id


class WebToolError(RuntimeError):
    """Raised for network or content failures visible to the agent."""


def _http_get(
    url: str,
    *,
    params: dict[str, str] | None = None,
    timeout: float,
    user_agent: str,
    max_bytes: int = 2_000_000,
) -> bytes:
    if params:
        separator = "&" if "?" in url else "?"
        url = f"{url}{separator}{urlencode(params)}"
    request = Request(
        url,
        headers={
            "User-Agent": user_agent,
            "Accept": "text/html,application/json,text/plain",
            "Accept-Encoding": "gzip",
        },
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            content = response.read(max_bytes + 1)
            content_encoding = response.headers.get("Content-Encoding", "").lower()
    except HTTPError as exc:
        raise WebToolError(f"HTTP {exc.code} while fetching {url}") from exc
    except URLError as exc:
        raise WebToolError(f"network error while fetching {url}: {exc.reason}") from exc
    except TimeoutError as exc:
        raise WebToolError(f"request timed out while fetching {url}") from exc
    if "gzip" in content_encoding:
        try:
            content = gzip.decompress(content)
        except OSError as exc:
            raise WebToolError(f"invalid gzip response while fetching {url}") from exc
    if len(content) > max_bytes:
        raise WebToolError(f"response exceeded {max_bytes} bytes: {url}")
    return content


def _validate_public_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("only http and https URLs are allowed")
    host = parsed.hostname.lower()
    if host in {"localhost", "localhost.localdomain"}:
        raise ValueError("localhost URLs are not allowed by the web fetch tool")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return
    if address.is_private or address.is_loopback or address.is_link_local or address.is_reserved:
        raise ValueError("private, loopback, link-local, and reserved IPs are not allowed")


class _TextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []
        self.skip_depth = 0
        self.in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript", "svg", "template"}:
            self.skip_depth += 1
        elif tag == "title":
            self.in_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "svg", "template"} and self.skip_depth:
            self.skip_depth -= 1
        elif tag == "title":
            self.in_title = False

    def handle_data(self, data: str) -> None:
        if self.skip_depth:
            return
        if self.in_title:
            self.title_parts.append(data)
        self.text_parts.append(data)


class _SearchResultParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[tuple[str, str, str]] = []
        self._active_title: tuple[str, str] | None = None
        self._active_snippet = False
        self._snippet_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = dict(attrs)
        classes = set((attrs_dict.get("class") or "").split())
        if tag == "a" and ("result__a" in classes or "result-title" in classes or "result-link" in classes):
            href = attrs_dict.get("href") or ""
            self._active_title = (href, "")
        elif classes.intersection({"result__snippet", "result-snippet"}):
            self._active_snippet = True
            self._snippet_parts = []

    def handle_endtag(self, tag: str) -> None:
        if self._active_snippet and tag == "td":
            self._active_snippet = False
            if self.results:
                title, href, _ = self.results[-1]
                self.results[-1] = (title, href, _clean_text(" ".join(self._snippet_parts)))
        if tag == "a" and self._active_title:
            href, title = self._active_title
            if href and title:
                self.results.append((title, href, ""))
            self._active_title = None

    def handle_data(self, data: str) -> None:
        if self._active_title:
            href, title = self._active_title
            self._active_title = (href, f"{title}{data}")
        if self._active_snippet:
            self._snippet_parts.append(data)


def _parse_searxng_results(raw: bytes, top_k: int) -> list[Source]:
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return []
    rows = data.get("results") if isinstance(data, dict) else None
    if not isinstance(rows, list):
        return []
    results: list[Source] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        url = row.get("url")
        title = row.get("title")
        if not isinstance(url, str) or not isinstance(title, str):
            continue
        results.append(Source(f"source_{len(results) + 1:03d}", title.strip(), url.strip(), str(row.get("content") or "").strip()))
        if len(results) >= top_k:
            break
    return results


def _parse_html_results(raw: bytes, top_k: int) -> list[Source]:
    parser = _SearchResultParser()
    decoded = raw.decode("utf-8", errors="replace")
    if "anomaly-modal" in decoded or "bots use DuckDuckGo" in decoded:
        raise WebToolError("search provider requested a bot challenge; use SearXNG or another search backend")
    parser.feed(decoded)
    results: list[Source] = []
    for title, raw_url, snippet in parser.results:
        url = _normalize_search_url(raw_url)
        if not url.startswith(("http://", "https://")):
            continue
        results.append(Source(f"source_{len(results) + 1:03d}", _clean_text(title), url, _clean_text(snippet)))
        if len(results) >= top_k:
            break
    if results:
        return results
    generic_parser = _GenericSearchResultParser()
    generic_parser.feed(decoded)
    for title, raw_url in generic_parser.results:
        url = _normalize_search_url(raw_url)
        if not url.startswith(("http://", "https://")):
            continue
        if _is_navigation_result(title, url):
            continue
        results.append(Source(f"source_{len(results) + 1:03d}", _clean_text(title), url))
        if len(results) >= top_k:
            break
    return results


def _is_navigation_result(title: str, url: str) -> bool:
    normalized_title = _clean_text(title).lower()
    host = (urlparse(url).hostname or "").lower()
    return normalized_title in {"google", "bing", "images", "videos", "maps", "news"} or host in {
        "google.com",
        "www.google.com",
        "bing.com",
        "www.bing.com",
    }


class _GenericSearchResultParser(HTMLParser):
    """Small parser for public Google/Bing result pages, without an API."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[tuple[str, str]] = []
        self._active_href: str | None = None
        self._active_text: list[str] = []
        self._heading_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "a" and self._active_href is None:
            href = dict(attrs).get("href") or ""
            if href:
                self._active_href = href
                self._active_text = []
        if tag in {"h2", "h3", "h4"}:
            self._heading_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"h2", "h3", "h4"} and self._heading_depth:
            self._heading_depth -= 1
        if tag == "a" and self._active_href is not None:
            title = _clean_text(" ".join(self._active_text))
            if title and len(title) >= 3:
                self.results.append((title, self._active_href))
            self._active_href = None
            self._active_text = []

    def handle_data(self, data: str) -> None:
        if self._active_href is not None and self._heading_depth:
            self._active_text.append(data)


def _normalize_search_url(raw_url: str) -> str:
    if raw_url.startswith("//"):
        raw_url = f"https:{raw_url}"
    parsed = urlparse(raw_url)
    query = parse_qs(parsed.query)
    if "uddg" in query and query["uddg"]:
        return unquote(query["uddg"][0])
    if "q" in query and query["q"] and parsed.path.rstrip("/") in {"/url", "/link"}:
        return unquote(query["q"][0])
    if "u" in query and query["u"]:
        encoded = query["u"][0]
        if encoded.startswith("a1"):
            try:
                decoded = base64.urlsafe_b64decode(encoded[2:] + "=" * (-len(encoded[2:]) % 4)).decode("utf-8")
            except (ValueError, UnicodeDecodeError):
                decoded = ""
            if decoded.startswith(("http://", "https://")):
                return decoded
    return raw_url


def _extract_html_text(raw: bytes, url: str) -> tuple[str, str]:
    parser = _TextParser()
    parser.feed(raw.decode("utf-8", errors="replace"))
    title = _clean_text(" ".join(parser.title_parts))
    text = _clean_text(" ".join(parser.text_parts))
    return title, text


def _clean_text(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
