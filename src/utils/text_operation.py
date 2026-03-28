import re
import hanzidentifier
from string import Formatter
from collections import defaultdict
from lingua import Language, LanguageDetectorBuilder

languages = [Language.ENGLISH, Language.CHINESE, Language.MALAY, Language.INDONESIAN]
detector = LanguageDetectorBuilder.from_languages(*languages).build()
FLOAT_REGEX = r"^-?\d+(?:,\d+)*(?:\.\d+)?$"


def detect_language(text: str, default_lang: str = "EN") -> str:
    """Detect the language of given text, if not detectable, return default_lang.

    Args:
        text (str): the target text.
        default_lang (str, optional):
            the default language to return if not detectable. Defaults to 'EN'.

    Returns:
        str: detected language in iso_639_1 https://en.wikipedia.org/wiki/ISO_639-1
    """
    res = detector.detect_language_of(text)
    if res:
        iso_639_1_code = res.iso_code_639_1.name
        # Further detect zh-Hans(ZHS) or zh-Hant(ZHT)
        if iso_639_1_code == 'ZH':
            hanzi_type = hanzidentifier.identify(text)
            _HANZI_MAP = {
                0: default_lang, # UNKNOWN
                1: "ZHT", # TRAD
                2: "ZHS", # SIMP
                3: "ZHT", # BOTH
                4: "ZHT" # MIXED
            }
            return _HANZI_MAP.get(hanzi_type, default_lang)

        return iso_639_1_code

    return default_lang


def is_meaningful_machine_read(text: str, min_ratio: float = 0.5, min_char: int = 250) -> bool:
    """Check if the machine readable text is meaningful.
    Due to default OCRs or other reasons, many scanned pdf may
    still contain machine-reaabable text, but output random words.
    This function tries to differentiate such cases from real machine readable docs using heuristic.

    Args:
        text (str): raw text extracted from the pdf
        min_ratio (float, optional):
            minimal ratio of English/Chinese characters over all text. Defaults to 0.5.
        min_char (int, optional):
            minimal length of valid characters. Defaults to 250.

    Returns:
        bool: is machine-readable or not
    """
    valid_char = re.findall(r"[\u4e00-\u9fffA-Za-z]", text)
    return (
        len(valid_char) / max(len(text), 1) >= min_ratio and len(valid_char) > min_char
    )


def find_neighboring_ocr_box(ocr_data: dict) -> dict:
    """Find the largest cell along each row, and the largest cell from the above row.
    This is to compensate the VLM had difficulties on locate the right cell.

    Args:
        ocr_data (dict): raw outout from PaddleOCR.

    Returns:
        dict: each cell value point to its current row and above row value.
    """
    res = {}
    rows, cur_row = [], []
    y_thres, prev_y = 10, -1

    # build table values
    for text, box in zip(ocr_data.get("rec_texts", []), ocr_data.get("rec_boxes", [])):
        if re.match(FLOAT_REGEX, text) and len(text) > 2:
            val = round(float(re.sub(",", "", text)))
            x, y = (box[0] + box[2]) // 2, (box[1] + box[3]) // 2
            if abs(y - prev_y) > y_thres:
                prev_y = y
                rows.append(cur_row)
                cur_row = []

            cur_row.append(val)

    # Assign the max value in the current row and above row
    rows.append(cur_row)
    prev_max = -1
    for row in rows[1:]:
        max_val = max(row, default=0)
        for ele in row:
            res[ele] = {"up": prev_max, "cur": max_val}
        prev_max = max_val

    return res
