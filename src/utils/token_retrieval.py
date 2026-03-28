import re
from typing import Dict, List, Set, Tuple, Optional

from wordsegment import load, segment

from src.const.page_retrieval_config import PAGE_RETRIEVAL_CONFIG

load()

cfg = PAGE_RETRIEVAL_CONFIG

_RE_TEXT = re.compile(r"[^\w\u4e00-\u9fa5\s\(\)\-]")
_RE_TOKEN = re.compile(r"[a-z0-9]+|[\u4e00-\u9fa5]")
_RE_STRIP_S = re.compile(r"s$")

_ivector_cache: Dict[str, List[float]] = {}
_vector_model: Optional["SentenceTransformer"] = None


def _get_vector_model():
    global _vector_model
    if _vector_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _vector_model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as e:
            print(f"[token_retrieval] failed to load sentence transformer: {e}")
            _vector_model = None
    return _vector_model


def _tokenize(text: str, lang: str) -> List[str]:
    text = _RE_TEXT.sub("", text.lower())
    tokens = _RE_TOKEN.findall(text)
    if lang == "EN":
        tokens = sum([segment(_RE_STRIP_S.sub("", t)) for t in tokens], [])
    return tokens


def _char_ngrams(text: str, n: int = 3) -> List[str]:
    text = text.lower().strip()
    return [text[i : i + n] for i in range(max(1, len(text) - n + 1))]


def _build_keyword_vector(keywords: List[str], model) -> Optional[List[float]]:
    if model is None:
        return None
    try:
        vec = model.encode([" ".join(keywords)], convert_to_numpy=True)
        return vec[0].tolist()
    except Exception:
        return None


def _cosine_sim(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _vector_similarity_score(page_text: str, kw_vector: List[float], model) -> float:
    if model is None or not kw_vector:
        return 0.0
    try:
        page_vec = model.encode([page_text.lower()], convert_to_numpy=True)
        page_list = page_vec[0].tolist()
        return _cosine_sim(kw_vector, page_list)
    except Exception:
        return 0.0


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_len: int = 0
        self.avgdl: float = 0.0
        self.idf: Dict[str, float] = {}
        self.doc_tokens: List[List[str]] = []

    def _compute_idf(self, docs: List[List[str]]) -> Dict[str, float]:
        idf = {}
        n = len(docs)
        df = {}
        for doc in docs:
            seen = set()
            for token in doc:
                if token not in seen:
                    df[token] = df.get(token, 0) + 1
                    seen.add(token)
        for token, freq in df.items():
            idf[token] = max(0.0, (n - freq + 0.5) / (freq + 0.5))
        return idf

    def add_doc(self, tokens: List[str]) -> None:
        self.doc_tokens.append(tokens)
        self.doc_len += len(tokens)

    def finalize(self) -> None:
        self.avgdl = self.doc_len / max(len(self.doc_tokens), 1)
        self.idf = self._compute_idf(self.doc_tokens)

    def score(self, query_tokens: List[str], doc_idx: int) -> float:
        doc = self.doc_tokens[doc_idx]
        dl = len(doc)
        freq = {}
        for t in doc:
            freq[t] = freq.get(t, 0) + 1
        score = 0.0
        for t in query_tokens:
            if t not in freq:
                continue
            tf = freq[t]
            idf = self.idf.get(t, 0.0)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1))
            score += idf * numerator / denominator
        return score


def ngram_sets(full_text: str, lang: str) -> Set[str]:
    tokens = _tokenize(full_text, lang)
    unigram = set(tokens)
    split = "" if lang.startswith("ZH") else " "
    bigram = {f"{tokens[i]}{split}{tokens[i+1]}" for i in range(len(tokens) - 1)}
    return unigram.union(bigram)


def keyword_match(tokens: Set[str], keywords_list: List[List[str]]) -> bool:
    if not keywords_list:
        return False
    for keywords in keywords_list:
        if all([w in tokens for w in keywords]):
            return True
    return False


def score_page(
    page_text: str,
    page_lang: str,
    doc_type: str,
    page_idx: int,
    bm25_engine: BM25,
) -> float:
    from src.const.keywords_map import KEYWORDS

    page_tokens = _tokenize(page_text, page_lang)
    page_token_set = set(page_tokens)

    model = _get_vector_model()

    keyword_hit = 0.0
    kw_vector_score = 0.0
    page_types = ["main", "dividend", "equity", "company_name"]

    for pt in page_types:
        kw_block = KEYWORDS.get(doc_type, {}).get("page_retrieval", {}).get(pt, {})
        kw_list = kw_block.get(page_lang) or kw_block.get("EN")
        if not kw_list or isinstance(kw_list, dict):
            continue

        flat_kw = []
        for group in kw_list:
            flat_kw.extend(group)

        kw_vec = _build_keyword_vector(flat_kw, model)
        if kw_vec:
            kw_vector_score += _vector_similarity_score(page_text.lower(), kw_vec, model)

        for group in kw_list:
            if all(w in page_token_set for w in group):
                keyword_hit += 1.0
                break

    bm25_q_tokens = _tokenize(page_text, page_lang)
    bm_score = bm25_engine.score(bm25_q_tokens, page_idx)
    norm_bm25 = bm_score / max(bm25_engine.avgdl, 1)

    total = (
        cfg["keyword_weight"] * keyword_hit
        + cfg["bm25_weight"] * norm_bm25
        + cfg["vector_weight"] * kw_vector_score
    )
    return total


def rank_pages(
    page_texts: List[str],
    page_langs: List[str],
    doc_type: str,
) -> List[Tuple[int, float]]:
    bm25 = BM25(k1=cfg["bm25_k1"], b=cfg["bm25_b"])
    for text in page_texts:
        bm25.add_doc(_tokenize(text, "EN"))

    bm25.finalize()

    scores = []
    for i, (text, lang) in enumerate(zip(page_texts, page_langs)):
        s = score_page(text, lang, doc_type, i, bm25)
        scores.append((i, s))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores
