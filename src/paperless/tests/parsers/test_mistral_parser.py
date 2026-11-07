from __future__ import annotations

import base64
import json
from typing import TYPE_CHECKING

import httpx
import pytest
from PIL import Image

from documents.consumer import should_produce_archive
from documents.parsers import ParseError
from paperless.models import RemoteOCRModel
from paperless.parsers.registry import get_parser_registry
from paperless.parsers.remote import RemoteDocumentParser
from paperless.parsers.tesseract import RasterisedDocumentParser

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_django.fixtures import Settings
    from pytest_httpx import HTTPXMock
    from pytest_mock import MockerFixture


@pytest.fixture(autouse=True)
def mistral_settings(settings: Settings, empty_remote_ocr_app_config, mocker):
    settings.REMOTE_OCR_ENGINE = "mistral"
    settings.REMOTE_OCR_API_KEY = "test-mistral-key"
    settings.REMOTE_OCR_ENDPOINT = None
    settings.REMOTE_OCR_MODEL = "mistral-ocr-4-1"
    settings.REMOTE_OCR_ALLOW_INTERNAL_ENDPOINTS = False
    mocker.patch("paperless.network.resolve_hostname_ips", return_value=["8.8.8.8"])
    return settings


@pytest.mark.parametrize("model", RemoteOCRModel.values)
@pytest.mark.parametrize("produce_archive", [True, False])
def test_mistral_pdf_request_and_page_order(
    *,
    model: str,
    produce_archive: bool,
    settings: Settings,
    remote_parser: RemoteDocumentParser,
    simple_digital_pdf_file: Path,
    httpx_mock: HTTPXMock,
) -> None:
    settings.REMOTE_OCR_MODEL = model
    httpx_mock.add_response(
        json={
            "pages": [
                {"index": 1, "markdown": "Second page"},
                {
                    "index": 0,
                    "markdown": "# First page\n\n| Amount |\n| --- |\n| €42 |",
                },
            ],
        },
    )

    assert remote_parser.can_produce_archive is False
    assert (
        should_produce_archive(
            remote_parser,
            "application/pdf",
            simple_digital_pdf_file,
        )
        is False
    )
    remote_parser.parse(
        simple_digital_pdf_file,
        "application/pdf",
        produce_archive=produce_archive,
    )

    assert (
        remote_parser.get_text()
        == "# First page\n\n| Amount |\n| --- |\n| €42 |\n\nSecond page"
    )
    assert remote_parser.get_archive_path() is None
    request = httpx_mock.get_request()
    assert request.method == "POST"
    assert request.url.path == "/v1/ocr"
    assert request.headers["host"] == "api.mistral.ai"
    assert request.headers["Authorization"] == "Bearer test-mistral-key"
    payload = json.loads(request.content)
    assert payload["model"] == model
    assert payload["include_image_base64"] is False
    assert payload["include_blocks"] is False
    assert payload["table_format"] is None
    assert payload["extract_header"] is False
    assert payload["extract_footer"] is False
    assert payload["document"]["type"] == "document_url"
    data_url = payload["document"]["document_url"]
    assert data_url.startswith("data:application/pdf;base64,")
    assert (
        base64.b64decode(data_url.split(",", 1)[1])
        == simple_digital_pdf_file.read_bytes()
    )


def test_mistral_image_and_thumbnail(
    remote_parser: RemoteDocumentParser,
    simple_png_file: Path,
    httpx_mock: HTTPXMock,
) -> None:
    httpx_mock.add_response(json={"pages": [{"index": 0, "markdown": "Image text"}]})
    remote_parser.parse(simple_png_file, "image/png", produce_archive=False)

    document = json.loads(httpx_mock.get_request().content)["document"]
    assert document["type"] == "image_url"
    assert document["image_url"].startswith("data:image/png;base64,")
    assert remote_parser.get_text() == "Image text"
    assert remote_parser.get_archive_path() is None
    with Image.open(remote_parser.get_thumbnail(simple_png_file, "image/png")) as im:
        assert im.format == "WEBP"
        assert im.width <= 500


def test_mistral_blank_page_is_valid(
    remote_parser: RemoteDocumentParser,
    simple_digital_pdf_file: Path,
    httpx_mock: HTTPXMock,
) -> None:
    httpx_mock.add_response(json={"pages": [{"index": 0, "markdown": ""}]})
    remote_parser.parse(simple_digital_pdf_file, "application/pdf")
    assert remote_parser.get_text() == ""


@pytest.mark.parametrize("model", RemoteOCRModel.values)
def test_mistral_removes_image_crops_without_changing_tables_or_captions(
    model: str,
    settings: Settings,
    remote_parser: RemoteDocumentParser,
    simple_digital_pdf_file: Path,
    httpx_mock: HTTPXMock,
) -> None:
    settings.REMOTE_OCR_MODEL = model
    table = "| Item | Value | Unit |\n| --- | --- | --- |\n| Example | ↑ 12.3 | µg/l |"
    httpx_mock.add_response(
        json={
            "pages": [
                {
                    "index": 0,
                    "markdown": (
                        "# Report\n\n![img-0.jpeg](img-0.jpeg)\n\n"
                        f"Printed barcode caption\n\n{table}\n\n"
                        "Keep [img-0.jpeg](img-0.jpeg) and "
                        "![original](original.jpeg).\n\nFooter"
                    ),
                    "images": [{"id": "img-0.jpeg", "image_base64": None}],
                    "tables": [],
                    "header": None,
                    "footer": None,
                    "blocks": [{"content": "Duplicate content from OCR 4.1"}],
                },
                {
                    "index": 1,
                    "markdown": "Before ![img-0.jpeg](img-0.jpeg) after\n\nEnd",
                    "images": [{"id": "img-0.jpeg", "image_base64": None}],
                },
            ],
        },
    )

    remote_parser.parse(simple_digital_pdf_file, "application/pdf")

    assert remote_parser.get_text() == (
        f"# Report\n\nPrinted barcode caption\n\n{table}\n\n"
        "Keep [img-0.jpeg](img-0.jpeg) and ![original](original.jpeg).\n\nFooter"
        "\n\nBefore after\n\nEnd"
    )


def test_mistral_image_only_page_has_empty_text(
    remote_parser: RemoteDocumentParser,
    simple_digital_pdf_file: Path,
    httpx_mock: HTTPXMock,
) -> None:
    httpx_mock.add_response(
        json={
            "pages": [
                {
                    "index": 0,
                    "markdown": "![img-0.jpeg](img-0.jpeg)",
                    "images": [{"id": "img-0.jpeg"}],
                },
            ],
        },
    )
    remote_parser.parse(simple_digital_pdf_file, "application/pdf")
    assert remote_parser.get_text() == ""


@pytest.mark.parametrize(
    "images",
    [None, "invalid", [None], [{}], [{"id": 1}], [{"id": ""}]],
)
def test_mistral_invalid_image_metadata_fails_consumption(
    images,
    remote_parser: RemoteDocumentParser,
    simple_digital_pdf_file: Path,
    httpx_mock: HTTPXMock,
) -> None:
    httpx_mock.add_response(
        json={"pages": [{"index": 0, "markdown": "text", "images": images}]},
    )
    with pytest.raises(ParseError, match="Mistral OCR parsing failed"):
        remote_parser.parse(simple_digital_pdf_file, "application/pdf")


@pytest.mark.parametrize(
    "result",
    [
        None,
        {},
        {"pages": []},
        {"pages": "text"},
        {"pages": [None]},
        {"pages": [{"index": 0, "markdown": None}]},
        {"pages": [{"index": 0}]},
        {"pages": [{"index": True, "markdown": "text"}]},
        {"pages": [{"index": -1, "markdown": "text"}]},
        {"pages": [{"index": 0, "markdown": "a"}, {"index": 0, "markdown": "b"}]},
    ],
)
def test_mistral_malformed_response_fails_consumption(
    result,
    remote_parser: RemoteDocumentParser,
    simple_digital_pdf_file: Path,
    httpx_mock: HTTPXMock,
) -> None:
    httpx_mock.add_response(content=json.dumps(result))
    with pytest.raises(ParseError, match="Mistral OCR parsing failed"):
        remote_parser.parse(simple_digital_pdf_file, "application/pdf")


@pytest.mark.parametrize("status_code", [302, 401, 429, 500])
def test_mistral_http_error_does_not_expose_response_or_follow_redirects(
    status_code: int,
    remote_parser: RemoteDocumentParser,
    simple_digital_pdf_file: Path,
    httpx_mock: HTTPXMock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    httpx_mock.add_response(
        status_code=status_code,
        headers={"Location": "https://other.example/ocr"},
        text="private document text",
    )
    with pytest.raises(ParseError, match=f"HTTP {status_code}") as exc:
        remote_parser.parse(simple_digital_pdf_file, "application/pdf")
    assert exc.value.__suppress_context__
    assert "private document text" not in str(exc.value) + caplog.text
    assert len(httpx_mock.get_requests()) == 1


def test_mistral_timeout_fails_consumption(
    remote_parser: RemoteDocumentParser,
    simple_digital_pdf_file: Path,
    httpx_mock: HTTPXMock,
) -> None:
    httpx_mock.add_exception(httpx.ReadTimeout("timeout"))
    with pytest.raises(ParseError, match="ReadTimeout"):
        remote_parser.parse(simple_digital_pdf_file, "application/pdf")


def test_mistral_custom_endpoint(
    settings: Settings,
    remote_parser: RemoteDocumentParser,
    simple_digital_pdf_file: Path,
    httpx_mock: HTTPXMock,
) -> None:
    settings.REMOTE_OCR_ENDPOINT = "https://ocr.example/custom/ocr"
    httpx_mock.add_response(json={"pages": [{"index": 0, "markdown": "text"}]})
    remote_parser.parse(simple_digital_pdf_file, "application/pdf")
    request = httpx_mock.get_request()
    assert request.headers["host"] == "ocr.example"
    assert request.url.path == "/custom/ocr"


def test_mistral_blocks_internal_endpoint(
    settings: Settings,
    mocker: MockerFixture,
    remote_parser: RemoteDocumentParser,
    simple_digital_pdf_file: Path,
    httpx_mock: HTTPXMock,
) -> None:
    settings.REMOTE_OCR_ENDPOINT = "http://localhost/ocr"
    mocker.patch("paperless.network.resolve_hostname_ips", return_value=["127.0.0.1"])
    with pytest.raises(ParseError, match="ValueError"):
        remote_parser.parse(simple_digital_pdf_file, "application/pdf")
    assert httpx_mock.get_requests() == []


@pytest.mark.parametrize("api_key", [None, ""])
def test_mistral_requires_api_key(api_key, settings: Settings) -> None:
    settings.REMOTE_OCR_API_KEY = api_key
    assert RemoteDocumentParser.score("application/pdf", "doc.pdf") is None


def test_mistral_registry_respects_remote_opt_in() -> None:
    registry = get_parser_registry()
    assert (
        registry.get_parser_for_file("application/pdf", "doc.pdf")
        is RemoteDocumentParser
    )
    assert (
        registry.get_parser_for_file(
            "application/pdf",
            "doc.pdf",
            allow_remote=False,
        )
        is RasterisedDocumentParser
    )
