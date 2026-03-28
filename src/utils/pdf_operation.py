from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import json_repair
import pymupdf

from PIL import Image

from src.const.keywords_map import KEYWORDS, TRANSLATION_KEYS, prompt_keys
from src.const.template import SOWA_EXTRACTION_TEMPLATE
from src.const.page_retrieval_config import PAGE_RETRIEVAL_CONFIG
from src.utils.central_service import OCR_Inferencer, VLM_Inferencer
from src.utils.image_operation import get_pil_from_pdf_page, pil_to_base64
from src.utils.text_operation import detect_language, is_meaningful_machine_read, find_neighboring_ocr_box
from src.utils.token_retrieval import ngram_sets, keyword_match, rank_pages


DEFAULT_PAGE = 3
PAGE_LIMIT = 6


def _page_handler(
    idx: int,
    page: pymupdf.Page,
    doc_type: str,
    ocr_engine: OCR_Inferencer,
    logger: Any = None,
) -> Tuple[Optional[Dict[str, Image.Image]], str, dict, str]:
    """Internal function to deal with the processing logic for each pdf page.

    Args:
        idx (int): Page index.
        page (PyMuPDF.Page): Page object from PyMuPDF
        doc_type (str):
            one of the pre-defined target types in ['finacial_statement', 'payslip', 'noa']
        ocr_engine (OCR_Inferencer):
            ocr inferencer engine to call centralized service.
        logger (Optional[logging.Logger]): centralized logger to record into Grafana,
            skip logging when setting None

    Returns:
        Tuple[Dict[str, Image.Image], str, dict, str]:
            image dict keyed by page_type, language code, OCR bbox results, and raw page text.
    """
    full_text = page.get_text().strip()
    ocr_res: dict = {}
    angle = 0

    if idx == 0 or not full_text or not is_meaningful_machine_read(full_text):
        pil_img = get_pil_from_pdf_page(page, dpi=144)
        if not pil_img:
            return (None, "", {}, "")

        ocr_res = ocr_engine.call_ocr(pil_img)
        if isinstance(ocr_res, list):
            ocr_res = ocr_res[0]
        else:
            print(f"[OCR] error on page {idx+1}: {ocr_res}")

        full_text = " ".join(ocr_res.get("rec_texts", []))
        angle = int(ocr_res.get("angle", 0))

    lang = detect_language(full_text)
    tokens = ngram_sets(full_text, lang)

    img_res: Dict[str, Image.Image] = {}
    bbox_res: dict = {}

    for page_type in KEYWORDS[doc_type]["page_retrieval"].keys():
        keywords_list = KEYWORDS[doc_type]["page_retrieval"][page_type].get(lang, None)
        if isinstance(keywords_list, dict):
            if idx < keywords_list.get("start", 0) or idx >= keywords_list.get("end", 2):
                continue
        elif not keyword_match(tokens, keywords_list):
            continue

        pil_img = get_pil_from_pdf_page(page)
        if angle != 0:
            pil_img = pil_img.rotate(angle, expand=True)

        img_res[page_type] = pil_img
        bbox_res[page_type] = find_neighboring_ocr_box(ocr_res)

    return (img_res, lang, bbox_res, full_text)


def page_finder(
    pdf_content: str,
    doc_type: str,
    ocr_engine: OCR_Inferencer,
    logger: Any = None,
) -> Dict[str, List[Tuple[Image.Image, str, dict]]]:
    """Get the target pages based on the keywords given extracted page contents.

    Args:
        pdf_content (str): PDF content in uploaded files.
        doc_type (str):
            one of the pre-defined target types in
            ['finacial_statement', 'payslip', 'noa']
        ocr_engine (OCR_Inferencer):
            ocr inferencer engine to call centralized service.
        logger (Optional[logging.Logger]): centralized logger to record into Grafana,
            skip logging when setting None

    Returns:
        Dict[str, List[Tuple[Image.Image, str]]]: Targetted images with the detected language in each page type.
    """
    all_page_texts: List[str] = []
    all_page_langs: List[str] = []

    with pymupdf.open(stream=pdf_content) as doc:
        for i, page in enumerate(doc):
            _, lang, _, text = _page_handler(i, page, doc_type, ocr_engine, logger)
            all_page_langs.append(lang)
            all_page_texts.append(text)

    ranked = rank_pages(all_page_texts, all_page_langs, doc_type)
    top_page_indices = set(idx for idx, _ in ranked[:PAGE_RETRIEVAL_CONFIG["page_score_top_k"]])

    target_imgs_with_lang: Dict[str, List[Tuple[Image.Image, str, dict]]] = defaultdict(list)
    langs: List[str] = []
    bboxes: List[dict] = []

    with pymupdf.open(stream=pdf_content) as doc:
        for i, page in enumerate(doc):
            imgs, page_lang, bbox_res, _ = _page_handler(i, page, doc_type, ocr_engine, logger)
            langs.append(page_lang)
            bboxes.append(bbox_res)
            if imgs:
                for page_type, img in imgs.items():
                    if i in top_page_indices or page_type == "main":
                        if len(target_imgs_with_lang[page_type]) < PAGE_LIMIT:
                            target_imgs_with_lang[page_type].append((img, page_lang, bbox_res))

        if not target_imgs_with_lang.get("main"):
            print("[page_finder] could not find main page, using default pages")
            for i, page in enumerate(doc[:DEFAULT_PAGE]):
                target_imgs_with_lang["main"].append((
                    get_pil_from_pdf_page(page), langs[i], bboxes[i],
                ))

    return target_imgs_with_lang


def _dedup_year_data(vlm_json: List[dict], lang: str) -> dict:
    """Deduplicate year data for each doc_type, given one year has only one set of extraction.

    Args:
        vlm_json (List[dict]): raw VLM output
        lang (str): language of the prompt and keys

    Returns:
        dict: json output after deduplicating the same year
    """
    years = {}
    for year_res in vlm_json:
        if not isinstance(year_res, dict):
            continue

        year = TRANSLATION_KEYS["year"].get(lang, "year")
        month = TRANSLATION_KEYS["month"].get(lang, "month")
        key = str(year_res.get(year, ""))
        if month in year_res:
            key = f"{key}-{year_res[month]}"

        if key not in years:
            years[key] = year_res
        else:
            for k, v in year_res.items():
                if v and str(v).isdigit():
                    years[key][k] = max(
                        int(str(years[key].get(k, '0'))) if str(years[key].get(k, '')).isdigit() else 0,
                        int(v)
                    )

    return list(years.values())


def get_page_extractions(
    imgs_with_lang: Dict[str, List[Tuple[Image.Image, str, dict]]],
    doc_type: str,
    vlm_engine: VLM_Inferencer,
    logger: Any = None,
) -> Tuple[dict, str, dict]:
    """Get the extractions based on the predefined doc-type.

    Args:
        imgs_with_lang (Dict[str, List[Tuple[Image.Image, str]]]):
            target PIL images with detected language in each category (e.g., main, dividend).
        doc_type (str):
            the type of submitted document, in [`finacial statement`, `payslip`].
        vlm_engine (VLM_Inferencer):
            the class calling centralized VLM service.
        logger (Optional[logging.Logger]):
            centralized logger to record into Grafana, skip logging when setting None.

    Returns:
        Tuple[dict, str]: extracted json key-value pairs, and detected language.
    """
    page_extractions = defaultdict(list)
    common_lang = {}
    bbox_res = {}
    for page_type, img_lang in imgs_with_lang.items():
        pil_img_list = [item[0] for item in img_lang]
        langs = [item[1] for item in img_lang]
        bbox = [item[2] for item in img_lang]
        most_common_lang = Counter(langs).most_common(1)[0][0]
        common_lang[page_type] = most_common_lang
        template = KEYWORDS[doc_type]["prompt_template"][page_type].get(
            most_common_lang, SOWA_EXTRACTION_TEMPLATE,
        )
        prompt = template.format(
            keywords=prompt_keys(
                lang=common_lang[page_type],
                doc_type=doc_type,
                page_type=page_type,
            ),
        )
        res_code, vlm_res = vlm_engine.call_vlm(
            [pil_to_base64(img) for img in pil_img_list],
            txt_prompt=prompt,
        )
        if res_code != 200:
            return {}, f"VLM Error when calling {doc_type}-{page_type}: {res_code}", {}

        vlm_json = json_repair.loads(vlm_res["generated_text"][0]["text"])
        if not isinstance(vlm_json, list):
            page_extractions[page_type].append(vlm_json)
        else:
            page_extractions[page_type] += _dedup_year_data(vlm_json, common_lang[page_type])

        if page_type not in bbox_res:
            bbox_res[page_type] = bbox

    return page_extractions, common_lang["main"], bbox_res
