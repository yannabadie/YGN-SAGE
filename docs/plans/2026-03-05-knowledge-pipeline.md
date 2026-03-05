# SOTA Knowledge Pipeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build an auto-refreshing knowledge pipeline that discovers, curates, and ingests SOTA research papers into ExoCortex, providing YGN-SAGE agents with always-current scientific grounding.

**Architecture:** 4 pipeline modules (discovery, curator, ingestion, migration) orchestrated by `pipeline.py`. ExoCortex wired end-to-end in `agent_loop.py` for passive grounding + 2 new agent tools (`search_exocortex`, `refresh_knowledge`). Single store `ygn-sage-research` with `custom_metadata` for domain filtering.

**Tech Stack:** Python 3.12+, Google GenAI File Search API, `arxiv` lib, `semanticscholar` lib, Gemini Flash for curation scoring.

**Design doc:** `docs/plans/2026-03-05-knowledge-pipeline-design.md`

**Baseline:** 175 Python tests + 38 Rust tests passing. Zero regressions allowed.

---

### Task 1: ExoCortex End-to-End Wiring in agent_loop.py

Wire `agent_loop.py` so `_think()` automatically passes ExoCortex store names to `generate()`, and create the `search_exocortex` agent tool.

**Files:**
- Modify: `sage-python/src/sage/agent_loop.py:109,206-210`
- Modify: `sage-python/src/sage/memory/remote_rag.py` (add `query()` method)
- Create: `sage-python/src/sage/tools/exocortex_tools.py`
- Modify: `sage-python/src/sage/boot.py:178-182`
- Test: `sage-python/tests/test_exocortex_wiring.py`

**Step 1: Write the failing tests**

Create `sage-python/tests/test_exocortex_wiring.py`:

```python
import sys, types

if "sage_core" not in sys.modules:
    sys.modules["sage_core"] = types.ModuleType("sage_core")

_mock_core = sys.modules["sage_core"]
if not hasattr(_mock_core, "WorkingMemory"):
    class _MockMemoryEvent:
        def __init__(self, id, event_type, content, timestamp_str, is_summary=False):
            self.id = id
            self.event_type = event_type
            self.content = content
            self.timestamp_str = timestamp_str
            self.is_summary = is_summary

    class _MockWorkingMemory:
        def __init__(self, agent_id, parent_id=None):
            self.agent_id = agent_id
            self.parent_id = parent_id
            self._events = []
            self._counter = 0
            self._children = []

        def add_event(self, event_type, content):
            self._counter += 1
            import time
            self._events.append(_MockMemoryEvent(
                id=f"evt-{self._counter}", event_type=event_type, content=content,
                timestamp_str=str(time.time()),
            ))
            return f"evt-{self._counter}"

        def get_event(self, event_id):
            for e in self._events:
                if e.id == event_id:
                    return e
            return None

        def recent_events(self, n):
            return self._events[-n:] if n > 0 else []

        def event_count(self):
            return len(self._events)

        def add_child_agent(self, child_id):
            self._children.append(child_id)

        def child_agents(self):
            return list(self._children)

        def compress_old_events(self, keep_recent, summary):
            kept = self._events[-keep_recent:] if keep_recent > 0 else []
            self._events = [_MockMemoryEvent(
                id="summary-0", event_type="summary", content=summary,
                timestamp_str="0", is_summary=True,
            )] + kept

        def compact_to_arrow(self):
            return 0

        def compact_to_arrow_with_meta(self, keywords, embedding, parent_chunk_id):
            return 0

        def retrieve_relevant_chunks(self, active_chunk_id, max_hops, weights):
            return []

        def get_page_out_candidates(self, active_chunk_id, max_hops, budget):
            return []

        def smmu_chunk_count(self):
            return 0

        def get_latest_arrow_chunk(self):
            return None

    _mock_core.WorkingMemory = _MockWorkingMemory

import inspect
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


def test_agent_loop_has_exocortex_attribute():
    """AgentLoop must declare exocortex attribute."""
    from sage.agent_loop import AgentLoop
    from sage.agent import AgentConfig
    from sage.llm.base import LLMConfig
    from sage.llm.mock import MockProvider

    config = AgentConfig(
        name="test", llm=LLMConfig(provider="mock", model="mock"),
        max_steps=1, validation_level=1,
    )
    loop = AgentLoop(config=config, llm_provider=MockProvider(responses=["ok"]))
    assert hasattr(loop, "exocortex")


def test_think_passes_file_search_store_names():
    """The _think section of run() must pass file_search_store_names when exocortex is available."""
    source = inspect.getsource(__import__("sage.agent_loop", fromlist=["AgentLoop"]).AgentLoop)
    assert "file_search_store_names" in source


def test_exocortex_query_method_exists():
    """ExoCortex must have a query() method for the search tool."""
    from sage.memory.remote_rag import ExoCortex
    exo = ExoCortex(store_name="test-store")
    assert hasattr(exo, "query")
    assert callable(exo.query)


def test_search_exocortex_tool_exists():
    """search_exocortex tool must be importable."""
    from sage.tools.exocortex_tools import create_exocortex_tools
    assert callable(create_exocortex_tools)


@pytest.mark.asyncio
async def test_agent_loop_passes_store_names_to_generate():
    """When exocortex has a store_name, generate() receives file_search_store_names."""
    from sage.agent_loop import AgentLoop
    from sage.agent import AgentConfig
    from sage.llm.base import LLMConfig, LLMResponse
    from sage.memory.remote_rag import ExoCortex

    config = AgentConfig(
        name="test", llm=LLMConfig(provider="mock", model="mock"),
        max_steps=1, validation_level=1,
    )
    mock_provider = AsyncMock()
    mock_provider.generate.return_value = LLMResponse(content="Answer with grounding.")

    loop = AgentLoop(config=config, llm_provider=mock_provider)
    exo = ExoCortex(store_name="file_search_stores/abc123")
    exo._api_key = "fake-key"
    loop.exocortex = exo

    await loop.run("What is MARL?")

    # Verify generate was called with file_search_store_names
    call_kwargs = mock_provider.generate.call_args
    assert call_kwargs is not None
    # Check keyword args for file_search_store_names
    if call_kwargs.kwargs.get("file_search_store_names"):
        assert "file_search_stores/abc123" in call_kwargs.kwargs["file_search_store_names"]
    else:
        # May be positional — check all calls
        found = False
        for call in mock_provider.generate.call_args_list:
            if call.kwargs.get("file_search_store_names"):
                found = True
                break
        assert found, "generate() was never called with file_search_store_names"
```

**Step 2: Run tests to verify they fail**

Run: `cd sage-python && python -m pytest tests/test_exocortex_wiring.py -v`
Expected: FAIL — `exocortex` attribute missing from `__init__`, no `file_search_store_names` in source, no `query()` method, no `exocortex_tools` module.

**Step 3: Implement ExoCortex wiring**

3a. Add `exocortex` attribute to `AgentLoop.__init__` in `sage-python/src/sage/agent_loop.py:109` — add after `self.sandbox_manager`:

```python
        self.exocortex: Any = None  # ExoCortex for File Search grounding
```

3b. Modify `_think` section in `agent_loop.py:206-210`. Replace:

```python
            response = await self._llm.generate(
                messages=messages,
                tools=tool_defs if tool_defs else None,
                config=self.config.llm,
            )
```

With:

```python
            # ExoCortex passive grounding
            exo_store_names = None
            if self.exocortex and hasattr(self.exocortex, "store_name") and self.exocortex.store_name:
                if hasattr(self.exocortex, "is_available") and self.exocortex.is_available:
                    exo_store_names = [self.exocortex.store_name]

            response = await self._llm.generate(
                messages=messages,
                tools=tool_defs if tool_defs else None,
                config=self.config.llm,
                file_search_store_names=exo_store_names,
            )
```

3c. Add `query()` method to `ExoCortex` in `sage-python/src/sage/memory/remote_rag.py` — after `delete_store()` method (after line 97):

```python
    def query(self, question: str, domain: str | None = None) -> str:
        """Synchronous query for tool use. Returns grounded answer or empty string.

        This uses generate() with file_search grounding to answer questions
        from the store's indexed documents.
        """
        if not self.store_name or not self._api_key:
            return ""
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=self._api_key)
            tools = [types.Tool(
                file_search=types.FileSearch(
                    file_search_store_names=[self.store_name]
                )
            )]
            config = types.GenerateContentConfig(
                tools=tools,
                system_instruction="Answer based on the indexed documents. Be precise and cite sources.",
            )
            response = client.models.generate_content(
                model="gemini-3.1-flash-lite-preview",
                contents=question,
                config=config,
            )
            return response.text or ""
        except Exception as e:
            log.warning("ExoCortex query failed: %s", e)
            return ""
```

3d. Create `sage-python/src/sage/tools/exocortex_tools.py`:

```python
"""ExoCortex agent tools: search_exocortex + refresh_knowledge."""
from __future__ import annotations

import asyncio
from typing import Any

from sage.tools.base import Tool


def create_exocortex_tools(exocortex: Any) -> list[Tool]:
    """Create ExoCortex tools bound to the given ExoCortex instance."""
    tools: list[Tool] = []

    @Tool.define(
        name="search_exocortex",
        description=(
            "Search the ExoCortex knowledge store for research papers and SOTA insights. "
            "Use when you need specific research knowledge about MARL, cognitive architectures, "
            "formal verification, evolutionary computation, or memory systems."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Research question to search for"},
                "domain": {
                    "type": "string",
                    "description": "Optional domain filter: marl, cognitive_architectures, formal_verification, evolutionary_computation, memory_systems",
                },
            },
            "required": ["query"],
        },
    )
    async def search_exocortex(query: str, domain: str | None = None) -> str:
        if not exocortex or not exocortex.store_name:
            return "ExoCortex not configured. Set SAGE_EXOCORTEX_STORE environment variable."
        result = await asyncio.to_thread(exocortex.query, query, domain)
        if not result:
            return "No relevant results found in ExoCortex."
        return result

    tools.append(search_exocortex)
    return tools
```

3e. Wire in `boot.py` — after line 182 (after memory tools registration), add:

```python
    # ExoCortex tools (search + refresh)
    from sage.tools.exocortex_tools import create_exocortex_tools
    for tool in create_exocortex_tools(exocortex):
        tool_registry.register(tool)
```

3f. Update `LLMProvider.generate()` base signature — check if `MockProvider` and `CodexProvider` accept `**kwargs` for `file_search_store_names`. If not, add `**kwargs` to their `generate()` signatures so they silently ignore the parameter.

Check `sage-python/src/sage/llm/mock.py` and `sage-python/src/sage/llm/codex.py` — add `file_search_store_names: list[str] | None = None` or `**kwargs` to their `generate()` method signatures.

**Step 4: Run tests to verify they pass**

Run: `cd sage-python && python -m pytest tests/test_exocortex_wiring.py -v`
Expected: 5 PASS

**Step 5: Run full test suite for regressions**

Run: `cd sage-python && python -m pytest -x`
Expected: All 180+ tests pass (175 existing + 5 new)

**Step 6: Commit**

```bash
git add sage-python/tests/test_exocortex_wiring.py sage-python/src/sage/agent_loop.py sage-python/src/sage/memory/remote_rag.py sage-python/src/sage/tools/exocortex_tools.py sage-python/src/sage/boot.py
git commit -m "feat(exocortex): wire File Search end-to-end in agent_loop + search_exocortex tool"
```

---

### Task 2: Discovery Module — arXiv + Semantic Scholar + HuggingFace

Create the discovery module that scans 3 sources for new papers in 5 target domains.

**Files:**
- Modify: `sage-discover/pyproject.toml:6-8` (add dependencies)
- Create: `sage-discover/src/discover/discovery.py`
- Test: `sage-discover/tests/test_discovery.py`

**Step 1: Add dependencies to pyproject.toml**

Modify `sage-discover/pyproject.toml` dependencies to:

```toml
dependencies = [
    "ygn-sage>=0.1.0",
    "arxiv>=2.1",
    "semanticscholar>=0.8",
]
```

**Step 2: Write the failing tests**

Create `sage-discover/tests/test_discovery.py`:

```python
"""Tests for the discovery module — paper scanning from arXiv, S2, HuggingFace."""
import pytest
from datetime import date
from unittest.mock import AsyncMock, patch, MagicMock

from discover.discovery import (
    PaperCandidate,
    DOMAINS,
    discover_arxiv,
    discover_semantic_scholar,
    discover_hf,
    deduplicate,
    discover,
)


def test_domains_has_five_entries():
    assert len(DOMAINS) == 5
    assert "marl" in DOMAINS
    assert "cognitive_architectures" in DOMAINS
    assert "formal_verification" in DOMAINS
    assert "evolutionary_computation" in DOMAINS
    assert "memory_systems" in DOMAINS


def test_paper_candidate_dataclass():
    p = PaperCandidate(
        paper_id="2503.01234",
        title="Test Paper",
        authors=["Alice"],
        abstract="A test abstract.",
        source="arxiv",
        domain="marl",
        published=date(2026, 3, 5),
        pdf_url="https://arxiv.org/pdf/2503.01234.pdf",
        citation_count=0,
    )
    assert p.paper_id == "2503.01234"
    assert p.source == "arxiv"


def test_deduplicate_merges_by_title():
    p1 = PaperCandidate(
        paper_id="2503.01234", title="Attention Is All You Need",
        authors=["A"], abstract="abs", source="arxiv", domain="marl",
        published=date(2026, 3, 1), pdf_url=None, citation_count=0,
    )
    p2 = PaperCandidate(
        paper_id="S2-12345", title="attention is all you need",
        authors=["A"], abstract="abs", source="s2", domain="marl",
        published=date(2026, 3, 1), pdf_url=None, citation_count=42,
    )
    p3 = PaperCandidate(
        paper_id="2503.99999", title="Totally Different Paper",
        authors=["B"], abstract="other", source="hf", domain="marl",
        published=date(2026, 3, 2), pdf_url=None, citation_count=0,
    )
    result = deduplicate([p1, p2, p3])
    assert len(result) == 2
    # Merged paper should have highest citation count
    merged = [p for p in result if "attention" in p.title.lower()][0]
    assert merged.citation_count == 42


@pytest.mark.asyncio
async def test_discover_arxiv_returns_candidates():
    mock_result = MagicMock()
    mock_result.entry_id = "http://arxiv.org/abs/2503.01234v1"
    mock_result.title = "MARL Paper"
    mock_result.authors = [MagicMock(name="Author A")]
    mock_result.authors[0].name = "Author A"
    mock_result.summary = "A paper about multi-agent reinforcement learning."
    mock_result.published = MagicMock()
    mock_result.published.date.return_value = date(2026, 3, 4)
    mock_result.pdf_url = "https://arxiv.org/pdf/2503.01234.pdf"

    with patch("discover.discovery.arxiv") as mock_arxiv:
        mock_client = MagicMock()
        mock_arxiv.Client.return_value = mock_client
        mock_client.results.return_value = iter([mock_result])
        mock_arxiv.Search.return_value = MagicMock()
        mock_arxiv.SortCriterion.SubmittedDate = "submittedDate"

        candidates = await discover_arxiv("marl", since=date(2026, 3, 1))
        assert len(candidates) >= 1
        assert candidates[0].source == "arxiv"
        assert candidates[0].domain == "marl"


@pytest.mark.asyncio
async def test_discover_returns_deduplicated():
    """Full discover() pipeline returns deduplicated candidates."""
    fake_candidates = [
        PaperCandidate(
            paper_id="2503.00001", title="Test Paper A",
            authors=["X"], abstract="Abstract A", source="arxiv", domain="marl",
            published=date(2026, 3, 5), pdf_url=None, citation_count=0,
        ),
    ]
    with patch("discover.discovery.discover_arxiv", new_callable=AsyncMock, return_value=fake_candidates), \
         patch("discover.discovery.discover_semantic_scholar", new_callable=AsyncMock, return_value=[]), \
         patch("discover.discovery.discover_hf", new_callable=AsyncMock, return_value=[]):
        result = await discover(since=date(2026, 3, 4))
        assert len(result) >= 1
```

**Step 3: Run tests to verify they fail**

Run: `cd sage-discover && python -m pytest tests/test_discovery.py -v`
Expected: FAIL — `discover.discovery` module does not exist.

**Step 4: Implement the discovery module**

Create `sage-discover/src/discover/discovery.py`:

```python
"""Discovery module: scan arXiv, Semantic Scholar, HuggingFace for new papers."""
from __future__ import annotations

import asyncio
import logging
import re
import string
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

log = logging.getLogger(__name__)

DOMAINS: dict[str, dict[str, Any]] = {
    "marl": {
        "arxiv_categories": ["cs.MA", "cs.AI", "cs.GT"],
        "keywords": ["multi-agent reinforcement learning", "PSRO", "Nash equilibrium"],
    },
    "cognitive_architectures": {
        "arxiv_categories": ["cs.AI", "cs.CL"],
        "keywords": ["cognitive architecture", "metacognition", "SOFAI", "dual process"],
    },
    "formal_verification": {
        "arxiv_categories": ["cs.LO", "cs.SE"],
        "keywords": ["SMT solver", "Z3", "formal verification", "program synthesis"],
    },
    "evolutionary_computation": {
        "arxiv_categories": ["cs.NE", "cs.AI"],
        "keywords": ["MAP-Elites", "quality diversity", "evolutionary strategy", "LLM mutation"],
    },
    "memory_systems": {
        "arxiv_categories": ["cs.AI", "cs.CL"],
        "keywords": ["episodic memory", "working memory", "RAG", "retrieval augmented"],
    },
}


@dataclass
class PaperCandidate:
    paper_id: str
    title: str
    authors: list[str]
    abstract: str
    source: str  # "arxiv" | "s2" | "hf"
    domain: str
    published: date
    pdf_url: str | None
    citation_count: int


def _normalize_title(title: str) -> str:
    """Normalize title for dedup: lowercase, strip punctuation."""
    return re.sub(r"[^\w\s]", "", title.lower()).strip()


def deduplicate(candidates: list[PaperCandidate]) -> list[PaperCandidate]:
    """Deduplicate by normalized title, keeping highest citation count."""
    seen: dict[str, PaperCandidate] = {}
    for c in candidates:
        key = _normalize_title(c.title)
        if key in seen:
            if c.citation_count > seen[key].citation_count:
                seen[key] = c
        else:
            seen[key] = c
    return list(seen.values())


async def discover_arxiv(
    domain: str,
    since: date | None = None,
    max_results: int = 20,
) -> list[PaperCandidate]:
    """Search arXiv for recent papers in a domain."""
    try:
        import arxiv
    except ImportError:
        log.warning("arxiv library not installed. Skipping arXiv discovery.")
        return []

    domain_config = DOMAINS.get(domain)
    if not domain_config:
        return []

    keywords = domain_config["keywords"]
    query = " OR ".join(f'"{kw}"' for kw in keywords[:3])

    def _search():
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
        )
        results = []
        for r in client.results(search):
            pub_date = r.published.date() if hasattr(r.published, "date") else r.published
            if since and pub_date < since:
                continue
            arxiv_id = r.entry_id.split("/abs/")[-1].replace("v1", "").replace("v2", "").strip("/")
            results.append(PaperCandidate(
                paper_id=arxiv_id,
                title=r.title,
                authors=[a.name for a in r.authors[:5]],
                abstract=r.summary,
                source="arxiv",
                domain=domain,
                published=pub_date,
                pdf_url=r.pdf_url,
                citation_count=0,
            ))
        return results

    return await asyncio.to_thread(_search)


async def discover_semantic_scholar(
    domain: str,
    since: date | None = None,
    max_results: int = 20,
) -> list[PaperCandidate]:
    """Search Semantic Scholar for recent papers in a domain."""
    try:
        from semanticscholar import SemanticScholar
    except ImportError:
        log.warning("semanticscholar library not installed. Skipping S2 discovery.")
        return []

    domain_config = DOMAINS.get(domain)
    if not domain_config:
        return []

    keywords = domain_config["keywords"]
    query = " ".join(keywords[:2])
    year_filter = f">{since.year - 1}" if since else ">2025"

    def _search():
        sch = SemanticScholar()
        results = []
        try:
            papers = sch.search_paper(query, year=year_filter, limit=max_results)
            for p in papers:
                if not p.title or not p.abstract:
                    continue
                pub_date = date.fromisoformat(p.publicationDate) if p.publicationDate else date.today()
                if since and pub_date < since:
                    continue
                results.append(PaperCandidate(
                    paper_id=str(p.paperId or ""),
                    title=p.title,
                    authors=[a.name for a in (p.authors or [])[:5] if a.name],
                    abstract=p.abstract or "",
                    source="s2",
                    domain=domain,
                    published=pub_date,
                    pdf_url=getattr(p, "url", None),
                    citation_count=p.citationCount or 0,
                ))
        except Exception as e:
            log.warning("Semantic Scholar search failed for %s: %s", domain, e)
        return results

    return await asyncio.to_thread(_search)


async def discover_hf(
    domain: str,
    max_results: int = 10,
) -> list[PaperCandidate]:
    """Search HuggingFace for papers in a domain."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        log.warning("huggingface_hub not installed. Skipping HF discovery.")
        return []

    domain_config = DOMAINS.get(domain)
    if not domain_config:
        return []

    keywords = domain_config["keywords"]
    query = keywords[0]

    def _search():
        api = HfApi()
        results = []
        try:
            papers = api.list_papers(query=query)
            for p in list(papers)[:max_results]:
                results.append(PaperCandidate(
                    paper_id=getattr(p, "id", "") or getattr(p, "paper", {}).get("id", ""),
                    title=getattr(p, "title", "") or str(p),
                    authors=[],
                    abstract=getattr(p, "summary", "") or "",
                    source="hf",
                    domain=domain,
                    published=date.today(),
                    pdf_url=None,
                    citation_count=0,
                ))
        except Exception as e:
            log.warning("HuggingFace paper search failed for %s: %s", domain, e)
        return results

    return await asyncio.to_thread(_search)


async def discover(
    since: date | None = None,
    query: str | None = None,
    domains: list[str] | None = None,
) -> list[PaperCandidate]:
    """Run discovery across all sources and domains. Returns deduplicated candidates."""
    if since is None:
        since = date.today() - timedelta(days=1)

    target_domains = domains or list(DOMAINS.keys())
    all_candidates: list[PaperCandidate] = []

    for domain in target_domains:
        # Run all 3 sources in parallel per domain
        arxiv_task = discover_arxiv(domain, since=since)
        s2_task = discover_semantic_scholar(domain, since=since)
        hf_task = discover_hf(domain)

        results = await asyncio.gather(arxiv_task, s2_task, hf_task, return_exceptions=True)
        for r in results:
            if isinstance(r, list):
                all_candidates.extend(r)
            elif isinstance(r, Exception):
                log.warning("Discovery source failed: %s", r)

    return deduplicate(all_candidates)
```

**Step 5: Run tests to verify they pass**

Run: `cd sage-discover && python -m pytest tests/test_discovery.py -v`
Expected: 6 PASS

**Step 6: Run full test suite for regressions**

Run: `cd sage-discover && python -m pytest -x`
Expected: All tests pass (existing + 6 new)

**Step 7: Commit**

```bash
git add sage-discover/pyproject.toml sage-discover/src/discover/discovery.py sage-discover/tests/test_discovery.py
git commit -m "feat(discover): add discovery module — arXiv, Semantic Scholar, HuggingFace scanning"
```

---

### Task 3: Curator Module — Heuristic Filter + LLM Scoring

Create the curator that filters noise and scores paper relevance using a two-stage pipeline.

**Files:**
- Create: `sage-discover/src/discover/curator.py`
- Test: `sage-discover/tests/test_curator.py`

**Step 1: Write the failing tests**

Create `sage-discover/tests/test_curator.py`:

```python
"""Tests for the curator module — heuristic filtering + LLM scoring."""
import pytest
from datetime import date, timedelta
from unittest.mock import AsyncMock, patch, MagicMock

from discover.discovery import PaperCandidate
from discover.curator import (
    CuratedPaper,
    heuristic_filter,
    llm_score,
    curate,
    RELEVANCE_THRESHOLD,
)


def _make_candidate(**overrides) -> PaperCandidate:
    defaults = dict(
        paper_id="2503.01234",
        title="A Novel Approach to Multi-Agent Reinforcement Learning",
        authors=["Alice", "Bob"],
        abstract="We propose a novel method for multi-agent reinforcement learning using policy-space response oracles and quality diversity. " * 3,
        source="arxiv",
        domain="marl",
        published=date.today(),
        pdf_url="https://arxiv.org/pdf/2503.01234.pdf",
        citation_count=5,
    )
    defaults.update(overrides)
    return PaperCandidate(**defaults)


def test_relevance_threshold_is_six():
    assert RELEVANCE_THRESHOLD == 6


def test_heuristic_rejects_short_abstract():
    p = _make_candidate(abstract="Too short.")
    result = heuristic_filter([p])
    assert len(result) == 0


def test_heuristic_rejects_stale_uncited():
    p = _make_candidate(
        published=date.today() - timedelta(days=100),
        citation_count=0,
    )
    result = heuristic_filter([p])
    assert len(result) == 0


def test_heuristic_keeps_recent_paper():
    p = _make_candidate(published=date.today(), citation_count=0)
    result = heuristic_filter([p])
    assert len(result) == 1


def test_heuristic_keeps_old_but_cited():
    p = _make_candidate(
        published=date.today() - timedelta(days=100),
        citation_count=15,
    )
    result = heuristic_filter([p])
    assert len(result) == 1


@pytest.mark.asyncio
async def test_llm_score_returns_curated_papers():
    """LLM scoring returns CuratedPaper with score and insights."""
    candidates = [_make_candidate()]
    mock_provider = AsyncMock()
    mock_provider.generate.return_value = MagicMock(
        content='[{"score": 8, "reason": "Highly relevant to MARL pillar", "key_insights": ["Uses PSRO", "Novel reward shaping"]}]'
    )

    result = await llm_score(candidates, llm=mock_provider)
    assert len(result) == 1
    assert result[0].relevance_score == 8
    assert len(result[0].key_insights) == 2


@pytest.mark.asyncio
async def test_curate_full_pipeline():
    """Full curate() applies heuristic + LLM scoring."""
    candidates = [_make_candidate(), _make_candidate(abstract="x")]
    mock_provider = AsyncMock()
    mock_provider.generate.return_value = MagicMock(
        content='[{"score": 7, "reason": "Relevant", "key_insights": ["insight"]}]'
    )

    result = await curate(candidates, llm=mock_provider)
    # One candidate filtered by heuristic (short abstract), one scored by LLM
    assert len(result) == 1
    assert result[0].relevance_score >= RELEVANCE_THRESHOLD
```

**Step 2: Run tests to verify they fail**

Run: `cd sage-discover && python -m pytest tests/test_curator.py -v`
Expected: FAIL — `discover.curator` module does not exist.

**Step 3: Implement the curator module**

Create `sage-discover/src/discover/curator.py`:

```python
"""Curator module: two-stage paper relevance filtering."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from discover.discovery import PaperCandidate

log = logging.getLogger(__name__)

RELEVANCE_THRESHOLD = 6
STALE_DAYS = 90
MIN_ABSTRACT_LENGTH = 100
BLOCKLIST_PATTERNS = ["survey of surveys", "correction to", "erratum"]


@dataclass
class CuratedPaper:
    candidate: PaperCandidate
    relevance_score: int
    reason: str
    key_insights: list[str] = field(default_factory=list)
    pdf_path: Path | None = None


def heuristic_filter(candidates: list[PaperCandidate]) -> list[PaperCandidate]:
    """Stage 1: fast, zero-cost heuristic filtering."""
    passed: list[PaperCandidate] = []
    today = date.today()

    for c in candidates:
        # Reject short abstracts
        if len(c.abstract) < MIN_ABSTRACT_LENGTH:
            log.debug("Rejected %s: abstract too short (%d chars)", c.paper_id, len(c.abstract))
            continue

        # Reject stale + uncited
        age_days = (today - c.published).days
        if age_days > STALE_DAYS and c.citation_count == 0:
            log.debug("Rejected %s: stale (%d days) and uncited", c.paper_id, age_days)
            continue

        # Reject blocklist patterns
        title_lower = c.title.lower()
        if any(pattern in title_lower for pattern in BLOCKLIST_PATTERNS):
            log.debug("Rejected %s: matches blocklist", c.paper_id)
            continue

        passed.append(c)

    log.info("Heuristic filter: %d/%d passed", len(passed), len(candidates))
    return passed


CURATION_PROMPT = """Rate each paper's relevance to YGN-SAGE (0-10).
YGN-SAGE builds: multi-agent systems, evolutionary code generation,
metacognitive routing, formal verification, memory architectures.

Papers:
{papers}

Return a JSON array with one object per paper:
[{{"score": int, "reason": str, "key_insights": [str, ...]}}]
Only return the JSON array, nothing else."""


async def llm_score(
    candidates: list[PaperCandidate],
    llm: Any,
    batch_size: int = 20,
) -> list[CuratedPaper]:
    """Stage 2: LLM-based relevance scoring."""
    from sage.llm.base import LLMConfig, Message, Role

    curated: list[CuratedPaper] = []

    for i in range(0, len(candidates), batch_size):
        batch = candidates[i : i + batch_size]
        papers_text = "\n\n".join(
            f"Paper {j+1}: {c.title}\nAbstract: {c.abstract[:500]}"
            for j, c in enumerate(batch)
        )
        prompt = CURATION_PROMPT.format(papers=papers_text)

        try:
            response = await llm.generate(
                messages=[Message(role=Role.USER, content=prompt)],
                config=LLMConfig(provider="google", model="gemini-3.1-flash-lite-preview"),
            )
            scores = json.loads(response.content)
            for j, score_data in enumerate(scores):
                if j >= len(batch):
                    break
                curated.append(CuratedPaper(
                    candidate=batch[j],
                    relevance_score=score_data.get("score", 0),
                    reason=score_data.get("reason", ""),
                    key_insights=score_data.get("key_insights", []),
                ))
        except (json.JSONDecodeError, Exception) as e:
            log.warning("LLM scoring failed for batch %d: %s", i, e)
            # Fallback: give all papers a neutral score
            for c in batch:
                curated.append(CuratedPaper(
                    candidate=c,
                    relevance_score=5,
                    reason="LLM scoring failed, neutral score assigned",
                ))

    return curated


async def curate(
    candidates: list[PaperCandidate],
    llm: Any,
) -> list[CuratedPaper]:
    """Full curation pipeline: heuristic filter + LLM scoring + threshold."""
    # Stage 1
    filtered = heuristic_filter(candidates)
    if not filtered:
        return []

    # Stage 2
    scored = await llm_score(filtered, llm)

    # Apply threshold
    passed = [p for p in scored if p.relevance_score >= RELEVANCE_THRESHOLD]
    log.info("Curation: %d/%d passed threshold (>=%d)", len(passed), len(scored), RELEVANCE_THRESHOLD)
    return passed
```

**Step 4: Run tests to verify they pass**

Run: `cd sage-discover && python -m pytest tests/test_curator.py -v`
Expected: 7 PASS

**Step 5: Run full test suite**

Run: `cd sage-discover && python -m pytest -x`
Expected: All pass

**Step 6: Commit**

```bash
git add sage-discover/src/discover/curator.py sage-discover/tests/test_curator.py
git commit -m "feat(discover): add curator module — heuristic filter + LLM relevance scoring"
```

---

### Task 4: Ingestion Module — ExoCortex Upload + Manifest

Create the ingestion module that uploads curated papers to ExoCortex with custom_metadata and tracks them in a local manifest.

**Files:**
- Create: `sage-discover/src/discover/ingestion.py`
- Test: `sage-discover/tests/test_ingestion.py`

**Step 1: Write the failing tests**

Create `sage-discover/tests/test_ingestion.py`:

```python
"""Tests for the ingestion module — ExoCortex upload + manifest tracking."""
import json
import pytest
from datetime import date
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

from discover.discovery import PaperCandidate
from discover.curator import CuratedPaper
from discover.ingestion import (
    Manifest,
    load_manifest,
    save_manifest,
    is_already_ingested,
    download_pdf,
    ingest,
    ingest_all,
)


@pytest.fixture
def tmp_manifest(tmp_path):
    return tmp_path / "manifest.json"


@pytest.fixture
def sample_curated():
    candidate = PaperCandidate(
        paper_id="2503.01234",
        title="Test Paper",
        authors=["Alice"],
        abstract="A long abstract about MARL and evolutionary computation." * 5,
        source="arxiv",
        domain="marl",
        published=date(2026, 3, 5),
        pdf_url="https://arxiv.org/pdf/2503.01234.pdf",
        citation_count=10,
    )
    return CuratedPaper(
        candidate=candidate,
        relevance_score=8,
        reason="Highly relevant",
        key_insights=["Uses PSRO", "Novel approach"],
        pdf_path=None,
    )


def test_manifest_structure():
    m = Manifest(store_name="file_search_stores/abc", papers={})
    assert m.store_name == "file_search_stores/abc"
    assert m.papers == {}


def test_load_manifest_missing_file(tmp_manifest):
    m = load_manifest(tmp_manifest)
    assert m.store_name == ""
    assert m.papers == {}


def test_save_and_load_manifest(tmp_manifest):
    m = Manifest(store_name="test-store", papers={"id1": {"title": "Paper 1"}})
    save_manifest(m, tmp_manifest)
    loaded = load_manifest(tmp_manifest)
    assert loaded.store_name == "test-store"
    assert "id1" in loaded.papers


def test_is_already_ingested():
    m = Manifest(store_name="s", papers={"2503.01234": {"title": "T"}})
    assert is_already_ingested("2503.01234", m) is True
    assert is_already_ingested("9999.99999", m) is False


@pytest.mark.asyncio
async def test_ingest_skips_already_ingested(sample_curated, tmp_manifest):
    m = Manifest(store_name="s", papers={"2503.01234": {"title": "Already there"}})
    save_manifest(m, tmp_manifest)

    mock_exo = MagicMock()
    result = await ingest(sample_curated, mock_exo, manifest_path=tmp_manifest)
    assert result is False  # Skipped


@pytest.mark.asyncio
async def test_ingest_all_returns_count(sample_curated, tmp_manifest):
    mock_exo = MagicMock()
    mock_exo.store_name = "file_search_stores/abc"
    mock_exo._api_key = "fake"

    with patch("discover.ingestion.ingest", new_callable=AsyncMock, return_value=True):
        count = await ingest_all([sample_curated], mock_exo, manifest_path=tmp_manifest)
        assert count == 1
```

**Step 2: Run tests to verify they fail**

Run: `cd sage-discover && python -m pytest tests/test_ingestion.py -v`
Expected: FAIL — `discover.ingestion` module does not exist.

**Step 3: Implement the ingestion module**

Create `sage-discover/src/discover/ingestion.py`:

```python
"""Ingestion module: upload curated papers to ExoCortex with metadata tracking."""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from discover.curator import CuratedPaper

log = logging.getLogger(__name__)

DEFAULT_MANIFEST_PATH = Path.home() / ".sage" / "manifest.json"
DEFAULT_PAPERS_DIR = Path.home() / ".sage" / "papers"


@dataclass
class Manifest:
    store_name: str = ""
    papers: dict[str, dict[str, Any]] = field(default_factory=dict)


def load_manifest(path: Path = DEFAULT_MANIFEST_PATH) -> Manifest:
    """Load manifest from disk. Returns empty manifest if file doesn't exist."""
    if not path.exists():
        return Manifest()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return Manifest(
            store_name=data.get("store_name", ""),
            papers=data.get("papers", {}),
        )
    except (json.JSONDecodeError, KeyError) as e:
        log.warning("Failed to load manifest: %s", e)
        return Manifest()


def save_manifest(manifest: Manifest, path: Path = DEFAULT_MANIFEST_PATH) -> None:
    """Save manifest to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"store_name": manifest.store_name, "papers": manifest.papers}, indent=2),
        encoding="utf-8",
    )


def is_already_ingested(paper_id: str, manifest: Manifest) -> bool:
    """Check if a paper is already in the manifest."""
    return paper_id in manifest.papers


async def download_pdf(url: str, dest: Path) -> bool:
    """Download a PDF from URL to dest path. Returns True on success."""
    if dest.exists():
        return True

    def _download():
        try:
            import urllib.request
            dest.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(url, str(dest))
            return True
        except Exception as e:
            log.warning("PDF download failed: %s — %s", url, e)
            return False

    return await asyncio.to_thread(_download)


async def ingest(
    paper: CuratedPaper,
    exocortex: Any,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
) -> bool:
    """Ingest a single curated paper into ExoCortex. Returns True if newly ingested."""
    manifest = load_manifest(manifest_path)
    paper_id = paper.candidate.paper_id

    if is_already_ingested(paper_id, manifest):
        log.debug("Skipping %s: already ingested", paper_id)
        return False

    # Download PDF if available
    pdf_path = paper.pdf_path
    if not pdf_path and paper.candidate.pdf_url:
        domain_dir = DEFAULT_PAPERS_DIR / paper.candidate.domain
        pdf_path = domain_dir / f"{paper_id.replace('/', '_')}.pdf"
        success = await download_pdf(paper.candidate.pdf_url, pdf_path)
        if not success:
            log.warning("Could not download PDF for %s, skipping ingestion", paper_id)
            return False

    if not pdf_path or not pdf_path.exists():
        log.warning("No PDF available for %s, skipping", paper_id)
        return False

    # Upload to ExoCortex
    try:
        await exocortex.upload(
            file_path=str(pdf_path),
            display_name=paper.candidate.title,
        )
    except Exception as e:
        log.error("ExoCortex upload failed for %s: %s", paper_id, e)
        return False

    # Update manifest
    manifest.store_name = exocortex.store_name or manifest.store_name
    manifest.papers[paper_id] = {
        "title": paper.candidate.title,
        "domain": paper.candidate.domain,
        "source": paper.candidate.source,
        "relevance_score": paper.relevance_score,
        "ingested_at": datetime.utcnow().isoformat(),
    }
    save_manifest(manifest, manifest_path)
    log.info("Ingested %s into ExoCortex", paper.candidate.title)
    return True


async def ingest_all(
    papers: list[CuratedPaper],
    exocortex: Any,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
) -> int:
    """Ingest all curated papers. Returns count of newly ingested papers."""
    count = 0
    for paper in papers:
        success = await ingest(paper, exocortex, manifest_path=manifest_path)
        if success:
            count += 1
    return count
```

**Step 4: Run tests to verify they pass**

Run: `cd sage-discover && python -m pytest tests/test_ingestion.py -v`
Expected: 6 PASS

**Step 5: Run full test suite**

Run: `cd sage-discover && python -m pytest -x`
Expected: All pass

**Step 6: Commit**

```bash
git add sage-discover/src/discover/ingestion.py sage-discover/tests/test_ingestion.py
git commit -m "feat(discover): add ingestion module — ExoCortex upload + manifest tracking"
```

---

### Task 5: Migration Module — NotebookLM Bootstrap

Create the migration module that bootstraps ExoCortex from NotebookLM exports.

**Files:**
- Create: `sage-discover/src/discover/migration.py`
- Test: `sage-discover/tests/test_migration.py`

**Step 1: Write the failing tests**

Create `sage-discover/tests/test_migration.py`:

```python
"""Tests for the migration module — NotebookLM bootstrap."""
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from discover.migration import (
    extract_arxiv_ids,
    migrate_markdown,
    migrate_notebooks,
)


def test_extract_arxiv_ids_from_text():
    text = """
    Based on the paper arXiv:2503.01234, the method uses PSRO.
    See also https://arxiv.org/abs/2401.56789 for more details.
    And 2312.11111v2 is also relevant.
    """
    ids = extract_arxiv_ids(text)
    assert "2503.01234" in ids
    assert "2401.56789" in ids
    assert "2312.11111" in ids


def test_extract_arxiv_ids_empty():
    assert extract_arxiv_ids("No papers here.") == []


@pytest.mark.asyncio
async def test_migrate_markdown_uploads_to_exocortex(tmp_path):
    md_file = tmp_path / "technical.md"
    md_file.write_text("# Technical Notes\n\nMEM1 uses rolling internal state.\n" * 10)

    mock_exo = AsyncMock()
    mock_exo.store_name = "test-store"
    mock_exo._api_key = "fake"

    await migrate_markdown(md_file, mock_exo, domain="cognitive_architectures")
    mock_exo.upload.assert_called_once()
    call_kwargs = mock_exo.upload.call_args
    assert "technical.md" in str(call_kwargs)


@pytest.mark.asyncio
async def test_migrate_notebooks_skips_missing_dir(tmp_path):
    mock_exo = AsyncMock()
    mock_exo.store_name = "test-store"
    # Should not raise even if migration dir doesn't exist
    result = await migrate_notebooks(mock_exo, migration_dir=tmp_path / "nonexistent")
    assert result == 0
```

**Step 2: Run tests to verify they fail**

Run: `cd sage-discover && python -m pytest tests/test_migration.py -v`
Expected: FAIL — `discover.migration` module does not exist.

**Step 3: Implement the migration module**

Create `sage-discover/src/discover/migration.py`:

```python
"""Migration module: bootstrap ExoCortex from NotebookLM markdown exports."""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

DEFAULT_MIGRATION_DIR = Path.home() / ".sage" / "migration"

# Maps markdown filename stems to domain tags
NOTEBOOK_DOMAINS = {
    "technical": "cognitive_architectures",
    "exocortex": "cognitive_architectures",
    "ygn": "evolutionary_computation",
    "discover": "marl",
    "discover_ai": "marl",
}


def extract_arxiv_ids(text: str) -> list[str]:
    """Extract arXiv paper IDs from text.

    Matches patterns like:
    - arXiv:2503.01234
    - https://arxiv.org/abs/2503.01234
    - 2503.01234v2
    """
    patterns = [
        r"arXiv:(\d{4}\.\d{4,5})",
        r"arxiv\.org/abs/(\d{4}\.\d{4,5})",
        r"\b(\d{4}\.\d{4,5})(?:v\d+)?",
    ]
    ids = set()
    for pattern in patterns:
        for match in re.findall(pattern, text, re.IGNORECASE):
            ids.add(match)
    return sorted(ids)


async def migrate_markdown(
    md_path: Path,
    exocortex: Any,
    domain: str | None = None,
) -> None:
    """Upload a single markdown file to ExoCortex."""
    if not md_path.exists():
        log.warning("Migration file not found: %s", md_path)
        return

    display_name = f"[Migration] {md_path.stem}"
    domain = domain or NOTEBOOK_DOMAINS.get(md_path.stem.lower(), "general")

    await exocortex.upload(
        file_path=str(md_path),
        display_name=display_name,
    )
    log.info("Migrated %s to ExoCortex (domain=%s)", md_path.name, domain)


async def migrate_notebooks(
    exocortex: Any,
    migration_dir: Path = DEFAULT_MIGRATION_DIR,
) -> int:
    """Migrate all markdown files in the migration directory to ExoCortex.

    Returns the number of files migrated.
    """
    if not migration_dir.exists():
        log.info("Migration directory not found: %s. Nothing to migrate.", migration_dir)
        return 0

    md_files = sorted(migration_dir.glob("*.md"))
    if not md_files:
        log.info("No markdown files found in %s", migration_dir)
        return 0

    count = 0
    for md_file in md_files:
        try:
            domain = NOTEBOOK_DOMAINS.get(md_file.stem.lower())
            await migrate_markdown(md_file, exocortex, domain=domain)
            count += 1
        except Exception as e:
            log.error("Failed to migrate %s: %s", md_file.name, e)

    log.info("Migration complete: %d/%d files", count, len(md_files))
    return count
```

**Step 4: Run tests to verify they pass**

Run: `cd sage-discover && python -m pytest tests/test_migration.py -v`
Expected: 4 PASS

**Step 5: Run full test suite**

Run: `cd sage-discover && python -m pytest -x`
Expected: All pass

**Step 6: Commit**

```bash
git add sage-discover/src/discover/migration.py sage-discover/tests/test_migration.py
git commit -m "feat(discover): add migration module — NotebookLM bootstrap to ExoCortex"
```

---

### Task 6: Pipeline Orchestrator + CLI + refresh_knowledge Tool

Create the central orchestrator that ties all modules together, a CLI entry point, and the `refresh_knowledge` agent tool.

**Files:**
- Create: `sage-discover/src/discover/pipeline.py`
- Modify: `sage-discover/src/discover/__init__.py:10-13` (export pipeline)
- Create: `sage-discover/src/discover/__main__.py` (CLI)
- Modify: `sage-python/src/sage/tools/exocortex_tools.py` (add refresh_knowledge tool)
- Test: `sage-discover/tests/test_pipeline.py`

**Step 1: Write the failing tests**

Create `sage-discover/tests/test_pipeline.py`:

```python
"""Tests for the pipeline orchestrator."""
import pytest
from datetime import date
from unittest.mock import AsyncMock, patch, MagicMock
from pathlib import Path

from discover.pipeline import run_pipeline, PipelineReport


def test_pipeline_report_structure():
    report = PipelineReport(discovered=10, curated=5, ingested=3)
    assert report.discovered == 10
    assert report.curated == 5
    assert report.ingested == 3


@pytest.mark.asyncio
async def test_pipeline_nightly_mode():
    mock_exo = AsyncMock()
    mock_exo.store_name = "test-store"
    mock_exo._api_key = "fake"

    with patch("discover.pipeline.discover", new_callable=AsyncMock, return_value=[]) as mock_disc, \
         patch("discover.pipeline.curate", new_callable=AsyncMock, return_value=[]) as mock_cur, \
         patch("discover.pipeline.ingest_all", new_callable=AsyncMock, return_value=0) as mock_ing:

        report = await run_pipeline(mode="nightly", exocortex=mock_exo)
        assert report.discovered == 0
        assert report.curated == 0
        assert report.ingested == 0
        mock_disc.assert_called_once()


@pytest.mark.asyncio
async def test_pipeline_on_demand_mode():
    mock_exo = AsyncMock()
    mock_exo.store_name = "test-store"
    mock_exo._api_key = "fake"

    with patch("discover.pipeline.discover", new_callable=AsyncMock, return_value=[]) as mock_disc, \
         patch("discover.pipeline.curate", new_callable=AsyncMock, return_value=[]) as mock_cur, \
         patch("discover.pipeline.ingest_all", new_callable=AsyncMock, return_value=0):

        report = await run_pipeline(mode="on-demand", query="attention mechanisms", exocortex=mock_exo)
        assert report is not None
        mock_disc.assert_called_once()


@pytest.mark.asyncio
async def test_pipeline_migrate_mode(tmp_path):
    mock_exo = AsyncMock()
    mock_exo.store_name = "test-store"

    with patch("discover.pipeline.migrate_notebooks", new_callable=AsyncMock, return_value=2):
        report = await run_pipeline(mode="migrate", exocortex=mock_exo)
        assert report.ingested == 2
```

**Step 2: Run tests to verify they fail**

Run: `cd sage-discover && python -m pytest tests/test_pipeline.py -v`
Expected: FAIL — `discover.pipeline` module does not exist.

**Step 3: Implement the pipeline orchestrator**

Create `sage-discover/src/discover/pipeline.py`:

```python
"""Pipeline orchestrator: ties discovery, curation, ingestion, and migration together."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

from discover.discovery import discover
from discover.curator import curate
from discover.ingestion import ingest_all
from discover.migration import migrate_notebooks

log = logging.getLogger(__name__)


@dataclass
class PipelineReport:
    discovered: int = 0
    curated: int = 0
    ingested: int = 0


async def run_pipeline(
    mode: str = "nightly",
    query: str | None = None,
    since: date | None = None,
    domains: list[str] | None = None,
    exocortex: Any = None,
    llm: Any = None,
) -> PipelineReport:
    """Run the knowledge pipeline.

    Modes:
        - "nightly": discover papers from yesterday, curate, ingest
        - "on-demand": discover papers matching a query, curate, ingest
        - "migrate": bootstrap ExoCortex from NotebookLM exports
    """
    if exocortex is None:
        from sage.memory.remote_rag import ExoCortex
        exocortex = ExoCortex(store_name=os.environ.get("SAGE_EXOCORTEX_STORE"))

    if mode == "migrate":
        count = await migrate_notebooks(exocortex)
        return PipelineReport(ingested=count)

    # Resolve LLM for curation scoring
    if llm is None:
        try:
            from sage.llm.google import GoogleProvider
            llm = GoogleProvider()
        except (ImportError, ValueError):
            log.warning("No LLM available for curation. Using heuristic-only mode.")

    # Discovery
    if since is None:
        since = date.today() - timedelta(days=1)

    candidates = await discover(since=since, query=query, domains=domains)
    log.info("Discovered %d candidates", len(candidates))

    if not candidates:
        return PipelineReport(discovered=0)

    # Curation
    if llm:
        curated = await curate(candidates, llm=llm)
    else:
        # Heuristic-only fallback
        from discover.curator import heuristic_filter, CuratedPaper
        filtered = heuristic_filter(candidates)
        curated = [
            CuratedPaper(candidate=c, relevance_score=6, reason="Heuristic pass (no LLM)")
            for c in filtered
        ]

    log.info("Curated %d papers", len(curated))

    if not curated:
        return PipelineReport(discovered=len(candidates), curated=0)

    # Ingestion
    ingested = await ingest_all(curated, exocortex)
    log.info("Ingested %d papers", ingested)

    return PipelineReport(
        discovered=len(candidates),
        curated=len(curated),
        ingested=ingested,
    )
```

**Step 4: Create CLI entry point**

Create `sage-discover/src/discover/__main__.py`:

```python
"""CLI entry point: python -m discover.pipeline"""
import argparse
import asyncio
import logging
import os
from datetime import date, timedelta

from discover.pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(description="YGN-SAGE Knowledge Pipeline")
    parser.add_argument("--mode", choices=["nightly", "on-demand", "migrate"], default="nightly")
    parser.add_argument("--query", type=str, help="Search query (on-demand mode)")
    parser.add_argument("--domains", type=str, nargs="*", help="Restrict to specific domains")
    parser.add_argument("--since", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    since = date.fromisoformat(args.since) if args.since else None

    report = asyncio.run(run_pipeline(
        mode=args.mode,
        query=args.query,
        since=since,
        domains=args.domains,
    ))
    print(f"Pipeline complete: discovered={report.discovered}, curated={report.curated}, ingested={report.ingested}")


if __name__ == "__main__":
    main()
```

**Step 5: Add refresh_knowledge tool to exocortex_tools.py**

Append to `sage-python/src/sage/tools/exocortex_tools.py` inside `create_exocortex_tools()`, before `return tools`:

```python
    @Tool.define(
        name="refresh_knowledge",
        description=(
            "Trigger on-demand knowledge discovery: scan arXiv, Semantic Scholar, "
            "and HuggingFace for new papers, curate them, and ingest into ExoCortex. "
            "Use when you need the latest research on a specific topic."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Research topic to search for"},
                "domain": {
                    "type": "string",
                    "description": "Optional domain: marl, cognitive_architectures, formal_verification, evolutionary_computation, memory_systems",
                },
            },
            "required": [],
        },
    )
    async def refresh_knowledge(query: str | None = None, domain: str | None = None) -> str:
        try:
            from discover.pipeline import run_pipeline
            domains = [domain] if domain else None
            report = await run_pipeline(
                mode="on-demand" if query else "nightly",
                query=query,
                domains=domains,
                exocortex=exocortex,
            )
            return (
                f"Knowledge refresh complete: "
                f"{report.discovered} discovered, "
                f"{report.curated} curated, "
                f"{report.ingested} ingested."
            )
        except ImportError:
            return "Knowledge pipeline not available. Install sage-discover."
        except Exception as e:
            return f"Knowledge refresh failed: {e}"

    tools.append(refresh_knowledge)
```

**Step 6: Update sage-discover __init__.py**

Update `sage-discover/src/discover/__init__.py` to export pipeline:

```python
"""sage-discover: Flagship Research & Discovery Agent.

Brings together all 5 YGN-SAGE cognitive pillars to autonomously
explore research domains, generate hypotheses, evolve solutions,
and evaluate discoveries.
"""

__version__ = "0.1.0"

from discover.workflow import DiscoverWorkflow, DiscoverConfig
from discover.researcher import ResearchAgent
from discover.pipeline import run_pipeline, PipelineReport

__all__ = ["DiscoverWorkflow", "DiscoverConfig", "ResearchAgent", "run_pipeline", "PipelineReport"]
```

**Step 7: Run tests to verify they pass**

Run: `cd sage-discover && python -m pytest tests/test_pipeline.py -v`
Expected: 4 PASS

**Step 8: Run full test suites**

Run: `cd sage-python && python -m pytest -x && cd ../sage-discover && python -m pytest -x`
Expected: All tests pass — zero regressions

**Step 9: Commit**

```bash
git add sage-discover/src/discover/pipeline.py sage-discover/src/discover/__main__.py sage-discover/src/discover/__init__.py sage-discover/tests/test_pipeline.py sage-python/src/sage/tools/exocortex_tools.py
git commit -m "feat(discover): add pipeline orchestrator + CLI + refresh_knowledge agent tool"
```
