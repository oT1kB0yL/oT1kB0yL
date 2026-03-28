SOWA_EXTRACTION_TEMPLATE = """Extract {keywords} in JSON only. \
Do not change key. Leave value empty if unsure. Return a list for multiple years"""

TW_EMPLOYMENT_EXTRACTION_TEMPLATE = """提取[{keywords}], 並按JSON格式輸出, 值僅為數字。 \
    不做計算, 未出現的信息留作空白。
"""

MS_EXTRACTION_TEMPLATE = """Ekstrak {keywords} dalam format JSON sahaja. \
Jangan tukar kunci. Biarkan nilai kosong jika tidak pasti. Return senarai untuk tahun berbilang."""

DIVIDEND_EXTRACTION_TEMPLATE = """
You are an expert financial data extraction specialist.
Your task is to extract the following keys: [{keywords}] from the given document.

**Important:**
* **Extract dividends paid by the company to shareholders. The amount is commonly stated as "dividends paid".**
* **Do NOT extract:**
    * Dividends *declared* or *approved*, but not yet paid.
    * Dividends *received* by the company.
    * Dividend amount per share.
    * Adjustments for dividend income.
    * Stock dividends.

Output result in JSON only. Do NOT change JSON key. Return a list for data with multiple years.

Your job depends on this task result, make sure to follow all instructions carefully!
"""

COMPANY_NAME_TEMPLATE = """Extract {keywords} in JSON '{keywords}: str'"""
