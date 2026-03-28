import re
from collections import defaultdict
from .template import (
    COMPANY_NAME_TEMPLATE,
    DIVIDEND_EXTRACTION_TEMPLATE,
    SOWA_EXTRACTION_TEMPLATE,
    TW_EMPLOYMENT_EXTRACTION_TEMPLATE,
    MS_EXTRACTION_TEMPLATE,
)

# Keywords on page retrieval and prompt construction
# - "main" section is must-to-have,
#     it indicates the critical page keywords, if cannot match, return the first 3 pages
# - "dividend" or other names are supplimentary sections, that is optional.
# - For arrays [[keywords]]
# -- Retrieve the page using keyword list
# -- within each sublist, they are 'and' logic, only if all strings exists.
# -- among each sublist, they are 'or' logic, as long as any sublist matches.
# - For dict {"start": 0, "end": 2}
# -- Retrieve the page in a range of page number
KEYWORDS = {
    "financial_statement": {
        "page_retrieval": {
            "main": {
                "ZHS": [["利润", "营业", "单位", "表"]],
                "ZHT": [["利潤", "營業", "單位", "表"]],
                "EN": [
                    ["revenue", "profit", "financial", "statement", "total"],
                    ["revenue", "profit", "financial", "statement", "expense"],
                    ["revenu", "profit", "financial", "statement", "expense"],
                    ["sale", "profit", "financial", "statement", "total"],
                    ["bloomberg", "certificate"],
                ],
                "ID": [
                    ["laba", "rugi", "laporan", "keuangan", "pendapatan"],
                    ["laba", "rugi", "laporan", "keuangan", "pengeluaran"],
                    ["pendapatan", "usaha", "laporan", "keuangan"],
                ],
                "MS": [
                    ["pendapatan", " Untung", "penyata", " kewangan"],
                    ["pendapatan", "pengeluaran", "penyata", "kewangan"],
                    ["jualan", "Untung", "penyata", "kewangan"],
                ],
            },
            "dividend": {
                "ZHS": [["应付", "付利", "利润", "营业", "单位", "表"]],
                "ZHT": [["應付", "付利", "利潤", "營業", "單位", "表"]],
                "EN": [["dividend", "paid", "financial", "statement"]],
                "ID": [["dividen", "dibayar", "laporan", "keuangan"]],
                "MS": [["dividen", "dibayar", "penyata", "kewangan"]],
            },
            "equity": {
                "ZHS": [["资产", "负债", "合计", "所有", "权益", "表"]],
                "ZHT": [["資產", "負債", "合計", "所有", "權益", "表"]],
                "EN": [["equity", "statement", "liabili", "total", "asset"]],
                "ID": [["ekuitas", "laporan", "aset", "liabilitas", "total"]],
                "MS": [["ekuiti", "penyata", "aset", "liability", "total"]],
            },
            "company_name": {
                "ZHS": [["利润", "营业", "单位", "表"]],
                "ZHT": [["利潤", "營業", "單位", "表"]],
                "EN": {"start": 0, "end": 2},
                "ID": {"start": 0, "end": 2},
                "MS": {"start": 0, "end": 2},
            },
        },
        "prompt_keys": {
            "main": ["year", "revenue", "net_profit", "currency_unit"],
            "dividend": ["year", "dividend", "currency_unit"],
            "equity": ["year", "total_equity", "total_liabilities", "currency_unit"],
            "company_name": ["company_name"],
        },
        "prompt_template": {
            "main": {
                "ZHS": SOWA_EXTRACTION_TEMPLATE,
                "ZHT": SOWA_EXTRACTION_TEMPLATE,
                "EN": SOWA_EXTRACTION_TEMPLATE,
                "ID": SOWA_EXTRACTION_TEMPLATE,
                "MS": MS_EXTRACTION_TEMPLATE,
            },
            "dividend": {
                "ZHS": SOWA_EXTRACTION_TEMPLATE,
                "ZHT": SOWA_EXTRACTION_TEMPLATE,
                "EN": DIVIDEND_EXTRACTION_TEMPLATE,
                "ID": DIVIDEND_EXTRACTION_TEMPLATE,
                "MS": DIVIDEND_EXTRACTION_TEMPLATE,
            },
            "equity": {
                "ZHS": SOWA_EXTRACTION_TEMPLATE,
                "ZHT": SOWA_EXTRACTION_TEMPLATE,
                "EN": SOWA_EXTRACTION_TEMPLATE,
                "ID": SOWA_EXTRACTION_TEMPLATE,
                "MS": MS_EXTRACTION_TEMPLATE,
            },
            "company_name": {
                "ZHS": COMPANY_NAME_TEMPLATE,
                "ZHT": COMPANY_NAME_TEMPLATE,
                "EN": COMPANY_NAME_TEMPLATE,
                "ID": COMPANY_NAME_TEMPLATE,
                "MS": COMPANY_NAME_TEMPLATE,
            },
        },
    },
    "payslip": {
        "page_retrieval": {
            "main": {
                "ZHS": [["特此"]],
                "EN": [["salary"]],
                "ZHT": [["證明"], ["納稅", "義務"]],
                "ID": [["gaji"], ["slip", "pendapatan"]],
                "MS": [["gaji"], ["slip", "pendapatan"]],
            },
            "tw_employment": {
                "ZHT": [["證明", "年起"], ["證明", "月起"]], # proof of income
            }
        },
        "prompt_keys": {
            "main": ["year", "month", "date", "net_pay", "commission", "currency_unit"],
            "tw_employment": ["start_year", "end_year", "monthly_salary", "net_pay", "commission", "currency_unit"],
        },
        "prompt_template": {
            "main": {
                "ZHS": SOWA_EXTRACTION_TEMPLATE,
                "EN": SOWA_EXTRACTION_TEMPLATE,
                "ZHT": SOWA_EXTRACTION_TEMPLATE,
                "ID": SOWA_EXTRACTION_TEMPLATE,
                "MS": SOWA_EXTRACTION_TEMPLATE,
            },
            "tw_employment": {
                "ZHT": TW_EMPLOYMENT_EXTRACTION_TEMPLATE,
            }
        },
    },

}

TRANSLATION_KEYS = {
    "year": {"ZHS": "年度", "ZHT": "年度", "ID": "tahun", "MS": "tahun"},
    "month": {"ZHS": "月份", "ZHT": "月份", "MS": "bulan", "ID": "bulan"},
    "date": {"ZHS": "日期", "ZHT": "日期", "ID": "tarikh", "MS": "tarikh"},
    "revenue": {"ZHS": "营收", "ZHT": "營收", "ID": "pendapatan", "MS": "pendapatan"},
    "net_profit": {"ZHS": "净利润", "ZHT": "淨利潤", "ID": "laba_bersih", "MS": "Untung bersih"},
    "net_pay": {"ZHS": "年薪", "ZHT": "年薪", "ID": "gaji_bersih", "MS": "gaji bersih"},
    "commission": {"ZHS": "分红", "ZHT": "分紅", "ID": "komisen", "MS": "komisen"},
    "dividend": {"ZHS": "应付利润", "ZHT": "應付利潤", "ID": "dividen", "MS": "dividen"},
    "currency_unit": {"ZHS": "货币单位", "ZHT": "貨幣單位", "ID": "mata_wang", "MS": "mata_wang"},
    "total_income": {"ZHS": "总收入", "ZHT": "總收入", "ID": "jumlah_pendapatan", "MS": "jumlah_pendapatan"},
    "total_deduction": {"ZHS": "总减免", "ZHT": "總減免", "ID": "jumlah_potongan", "MS": "jumlah_potongan"},
    "employment": {"ZHS": "雇佣", "ZHT": "僱傭", "ID": "pekerjaan", "MS": "pekerjaan"},
    "tax_payable": {"ZHS": "所得税", "ZHT": "所得稅", "ID": "cukai_terutang", "MS": "cukai perlu dibayar"},
    "total_equity": {"ZHS": "期末数_所有者权益合计", "ZHT": "期末數_所有者權益合計", "ID": "jumlah_ekuitas", "MS": "jumlah_ekuiti"},
    "total_liabilities": {"ZHS": "期末数_负债及所有者权益总计", "ZHT": "期末數_負債及所有者權益總計", "ID": "jumlah_liabilitas", "MS": "jumlah_liabiliti"},
    "deductions": {"ZHS": "减免", "ZHT": "減免", "ID": "potongan", "MS": "potongan"},
    "company_name": {"ZHS": "公司名", "ZHT": "公司名", "ID": "nama_syarikat", "MS": "nama syarikat"},
    "earnings": {"ZHS": "收入", "ZHT": "收入", "MS": "pendapatan", "ID": "pendapatan"},
    "monthly_salary": {"ZHS": "月薪", "ZHT": "月薪", "ID": "gaji_bulanan", "MS": "gaji bulanan"},
    "start_year": {"ZHS": "起始年", "ZHT": "起始年", "ID": "tahun_mula", "MS": "tahun mula"},
    "end_year": {"ZHS": "截止年", "ZHT": "截止年", "ID": "tahun_akhir", "MS": "tahun akhir"},
}


REVERSE_LANG_KEY_MAP = defaultdict(dict)
for eng_lang, v in TRANSLATION_KEYS.items():
    for lang_code, target_lang in v.items():
        REVERSE_LANG_KEY_MAP[lang_code][re.sub(r"[0-9]", "", target_lang)] = eng_lang

MONTH_DICT = {
    "1": "Jan",
    "2": "Feb",
    "3": "Mar",
    "4": "Apr",
    "5": "May",
    "6": "Jun",
    "7": "Jul",
    "8": "Aug",
    "9": "Sep",
    "10": "Oct",
    "11": "Nov",
    "12": "Dec",
    "january": "Jan",
    "february": "Feb",
    "march": "Mar",
    "april": "Apr",
    "may": "May",
    "june": "Jun",
    "july": "Jul",
    "august": "Aug",
    "september": "Sep",
    "october": "Oct",
    "november": "Nov",
    "december": "Dec",
}


def prompt_keys(lang: str, doc_type: str, page_type: str) -> str:
    """Generate relevant prompt to VLM with the target field names.

    Args:
        lang (str): the language code of target document.
        doc_type (str): the document type, e.g., payslip, financial statement, etc.
        page_type (str): the target page type, e.g., main, dividend, etc.

    Returns:
        str: the target field names to be embedded in the VLM prompt.
    """
    return ", ".join(
        [TRANSLATION_KEYS[k].get(lang, k) for k in KEYWORDS[doc_type]["prompt_keys"].get(page_type, [])]
    )
