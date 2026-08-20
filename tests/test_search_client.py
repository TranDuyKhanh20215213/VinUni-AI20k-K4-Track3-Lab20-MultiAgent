from multi_agent_research_lab.services.search_client import SearchClient


def test_search_client_finds_relevant_offline_sources() -> None:
    client = SearchClient()

    results = client.search("role specialization in multi-agent systems", max_results=3)

    assert results
    assert len(results) <= 3
    assert all(r.snippet for r in results)
    assert all(r.metadata.get("corpus_id") for r in results)


def test_search_client_returns_empty_for_nonsense_query() -> None:
    client = SearchClient()

    results = client.search("", max_results=3)

    assert results == []
