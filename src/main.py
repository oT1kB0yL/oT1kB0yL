#!/usr/bin/env python3
"""
Run document pipeline on a single input file.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict

from src.utils.central_service import OCR_Inferencer, VLM_Inferencer, LocalEasyOCR
from src.const.keywords_map import KEYWORDS
from src.utils.pdf_operation import page_finder, get_page_extractions
from src.utils.post_processing import normalize_fields

try:
    import doc_pipeline.src.const.api_config as app_config
except Exception:
    app_config = None


def _build_logger(level: str) -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    return logging.getLogger("doc_pipeline")


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def _config_get(attr: str, env_name: str, default: str = "") -> str:
    if app_config is not None and hasattr(app_config, attr):
        v = getattr(app_config, attr)
        if v is not None and str(v).strip() != "":
            return str(v)
    return _env(env_name, default)


def _build_ocr_inferencer(backend: str, logger: logging.Logger) -> Any:
    if backend == "local":
        return LocalEasyOCR(languages=["en", "ch_sim"], gpu=False, download_enabled=True)

    return OCR_Inferencer(
        ocr_endpoint=_config_get("OCR_API_URL", "OCR_API_URL", ""),
        ocr_auth_endpoint=_config_get("OCR_AUTH_URL", "OCR_AUTH_URL", ""),
        client_app_name=_config_get("OCR_CLIENT_APP_NAME", "OCR_CLIENT_APP_NAME", ""),
        server_app_name=_config_get("OCR_API_MODEL", "OCR_API_MODEL", ""),
        api_key=_config_get("OCR_API_KEY", "OCR_API_KEY", ""),
        logger=logger,
    )


def _build_vlm_inferencer(logger: logging.Logger) -> VLM_Inferencer:
    max_files_allowed = int(_env("MAX_FILES_ALLOWED", "6"))
    max_response_tokens = int(_env("MAX_RESPONSE_TOKENS", "2048"))
    return VLM_Inferencer(
        vlm_endpoint=_config_get("VLM_BASE_URL", "VLM_BASE_URL", ""),
        api_key=_config_get("VLM_API_KEY", "VLM_API_KEY", ""),
        model_name=_config_get("VLM_MODEL", "VLM_MODEL", "gpt-4o-mini"),
        max_files_allowed=max_files_allowed,
        max_response_tokens=max_response_tokens,
        logger=logger,
    )


def run_single_file(
    pdf_path: Path,
    doc_type: str,
    ocr_backend: str,
    logger: logging.Logger,
) -> Dict[str, Any]:
    if doc_type not in KEYWORDS:
        raise ValueError(f"Unsupported doc_type: {doc_type}. Supported: {', '.join(sorted(KEYWORDS.keys()))}")

    pdf_path = pdf_path.expanduser().resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(str(pdf_path))

    with open(pdf_path, "rb") as handler:
        file_bytes = handler.read()

    ocr_inferencer = _build_ocr_inferencer(backend=ocr_backend, logger=logger)
    vlm_inferencer = _build_vlm_inferencer(logger=logger)

    imgs_with_lang = page_finder(
        file_bytes,
        doc_type,
        ocr_inferencer,
        logger,
    )

    vlm_json, lang, bboxes = get_page_extractions(
        imgs_with_lang,
        doc_type,
        vlm_inferencer,
        logger,
    )
    print(vlm_json)
    data = normalize_fields(vlm_json, lang, doc_type, bboxes)
    return {
        "pdf_path": str(pdf_path),
        "doc_type": doc_type,
        "origin_lang": lang,
        "data": data,
    }


def _default_sample_pdf() -> Path:
    return Path(__file__).resolve().parent / "data" / "sample_termsheet.pdf"


if __name__ == "__main__":
    PDF_PATH = _default_sample_pdf()
    DOC_TYPE = "financial_statement"
    OCR_BACKEND = "remote"
    if app_config is not None and getattr(app_config, "OCR_MODE", "") == "local":
        OCR_BACKEND = "local"
    LOG_LEVEL = "INFO"
    OUT_PATH = None

    logger = _build_logger(LOG_LEVEL)
    prediction = run_single_file(
        pdf_path=PDF_PATH,
        doc_type=DOC_TYPE,
        ocr_backend=OCR_BACKEND,
        logger=logger,
    )

    out = json.dumps(prediction, ensure_ascii=False, indent=2)
    if OUT_PATH:
        out_path = Path(OUT_PATH)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(out, encoding="utf-8")
    else:
        print(out)
