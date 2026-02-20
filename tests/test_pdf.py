"""Tests for get_pdf_text() in app.py."""

from unittest.mock import MagicMock, patch

from app import get_pdf_text


def _make_mock_upload(name, page_texts):
    """Return a mock Streamlit upload object whose pages yield the given texts."""
    pages = []
    for text in page_texts:
        page = MagicMock()
        page.extract_text.return_value = text
        pages.append(page)

    mock_pdf = MagicMock()
    mock_pdf.name = name
    return mock_pdf, pages


def test_single_pdf_returns_page_tuples():
    mock_pdf, pages = _make_mock_upload("report.pdf", ["Page 1 content", "Page 2 content"])
    with patch("app.PdfReader") as mock_reader_cls, patch("app.st"):
        mock_reader = MagicMock()
        mock_reader.pages = pages
        mock_reader_cls.return_value = mock_reader

        result = get_pdf_text([mock_pdf])

    # Two pages → two tuples
    assert len(result) == 2
    text0, filename0, page_num0 = result[0]
    text1, filename1, page_num1 = result[1]
    assert filename0 == filename1 == "report.pdf"
    assert "Page 1 content" in text0
    assert "Page 2 content" in text1
    assert page_num0 == 1
    assert page_num1 == 2


def test_multiple_pdfs_return_page_tuples():
    mock_a, pages_a = _make_mock_upload("a.pdf", ["Alpha"])
    mock_b, pages_b = _make_mock_upload("b.pdf", ["Beta"])

    def reader_factory(upload):
        r = MagicMock()
        r.pages = pages_a if upload is mock_a else pages_b
        return r

    with patch("app.PdfReader", side_effect=reader_factory), patch("app.st"):
        result = get_pdf_text([mock_a, mock_b])

    assert len(result) == 2
    filenames = [r[1] for r in result]
    assert "a.pdf" in filenames
    assert "b.pdf" in filenames


def test_pdf_with_no_extractable_text_is_excluded():
    mock_pdf, pages = _make_mock_upload("empty.pdf", [None, ""])
    with patch("app.PdfReader") as mock_reader_cls, patch("app.st"):
        mock_reader = MagicMock()
        mock_reader.pages = pages
        mock_reader_cls.return_value = mock_reader

        result = get_pdf_text([mock_pdf])

    assert result == [], "PDFs with no extractable text should be excluded"


def test_broken_pdf_emits_warning_and_is_skipped():
    mock_pdf = MagicMock()
    mock_pdf.name = "broken.pdf"

    with patch("app.PdfReader", side_effect=Exception("corrupt")), patch("app.st") as mock_st:
        result = get_pdf_text([mock_pdf])

    assert result == []
    mock_st.warning.assert_called_once()
    warning_text = mock_st.warning.call_args[0][0]
    assert "broken.pdf" in warning_text


def test_mixed_good_and_broken_pdfs():
    mock_good, pages_good = _make_mock_upload("good.pdf", ["Valid content"])
    mock_bad = MagicMock()
    mock_bad.name = "bad.pdf"

    def reader_factory(upload):
        if upload is mock_bad:
            raise Exception("read error")
        r = MagicMock()
        r.pages = pages_good
        return r

    with patch("app.PdfReader", side_effect=reader_factory), patch("app.st") as mock_st:
        result = get_pdf_text([mock_good, mock_bad])

    assert len(result) == 1
    assert result[0][1] == "good.pdf"
    mock_st.warning.assert_called_once()
