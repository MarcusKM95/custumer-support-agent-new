from html.parser import HTMLParser
from urllib.parse import urldefrag

import httpx


class HTMLTextExtractor(HTMLParser):
    BLOCK_TAGS = {
        "article",
        "br",
        "div",
        "h1",
        "h2",
        "h3",
        "h4",
        "li",
        "main",
        "p",
        "section",
        "tr",
    }
    SKIP_TAGS = {"script", "style", "noscript", "svg"}

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self.SKIP_TAGS:
            self._skip_depth += 1
            return
        if tag in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self.SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
            return
        if tag in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        text = " ".join(data.split())
        if text:
            self.parts.append(f"{text} ")

    def get_text(self) -> str:
        lines = []
        for part in "".join(self.parts).splitlines():
            line = " ".join(part.split())
            if line:
                lines.append(line)
        return "\n".join(lines)


def fetch_html_text(url: str, timeout: float = 20.0) -> str:
    """
    Fetch an HTML page and return visible text.

    URL fragments are browser-only, so the fragment is stripped before the HTTP
    request and handled by source-specific extraction code.
    """
    request_url, _fragment = urldefrag(url)
    response = httpx.get(
        request_url,
        timeout=timeout,
        follow_redirects=True,
        headers={"User-Agent": "customer-support-agent-ingester/0.1"},
    )
    response.raise_for_status()

    parser = HTMLTextExtractor()
    parser.feed(response.text)
    return parser.get_text()
