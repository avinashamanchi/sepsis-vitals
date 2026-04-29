from pathlib import Path


SITE = Path("docs/index.html")


def test_pages_site_contains_research_dashboard_sections():
    html = SITE.read_text()

    required = [
        "Sepsis Vitals",
        "Dataset construction",
        "Model development",
        "Clinical validation",
        "Scale & open-source",
        "Nurse alert console",
        "AUROC >= 0.82",
        "Raspberry Pi 4",
    ]

    for text in required:
        assert text in html


def test_pages_site_is_static_and_github_pages_friendly():
    html = SITE.read_text()

    assert "<script>" in html
    assert "const phases =" in html
    assert "http://" not in html
    assert "https://" not in html


def test_github_pages_workflow_deploys_docs_directory():
    workflow = Path(".github/workflows/pages.yml").read_text()

    assert "actions/deploy-pages" in workflow
    assert "path: docs" in workflow
    assert "pages: write" in workflow
