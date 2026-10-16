from api.main import app, to_sources


def test_to_sources_maps_citation_fields():
    citation = {"chunk_id": "c1", "doc_name": "BLDS 2015", "article": "Điều 117",
                "clause": "Khoản 1", "point": "Điểm a", "source": "x.docx", "quote": "nguyên văn"}
    assert to_sources([citation]) == [{"doc_title": "BLDS 2015", "article": "Điều 117",
                                       "clause": "Khoản 1 Điểm a", "text": "nguyên văn"}]
    assert to_sources([{**citation, "clause": None, "point": None}])[0]["clause"] == ""


def test_routes_registered():
    paths = {route.path for route in app.routes}
    assert {"/api/health", "/api/index-info", "/api/chat"} <= paths


if __name__ == "__main__":
    test_to_sources_maps_citation_fields()
    test_routes_registered()
    print("ok")
