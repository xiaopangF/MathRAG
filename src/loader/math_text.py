import re
import unicodedata


SUPERSCRIPT_MAP = {
    "⁰": "0",
    "¹": "1",
    "²": "2",
    "³": "3",
    "⁴": "4",
    "⁵": "5",
    "⁶": "6",
    "⁷": "7",
    "⁸": "8",
    "⁹": "9",
    "⁺": "+",
    "⁻": "-",
    "⁼": "=",
    "⁽": "(",
    "⁾": ")",
    "ⁿ": "n",
}
SUBSCRIPT_MAP = {
    "₀": "0",
    "₁": "1",
    "₂": "2",
    "₃": "3",
    "₄": "4",
    "₅": "5",
    "₆": "6",
    "₇": "7",
    "₈": "8",
    "₉": "9",
    "₊": "+",
    "₋": "-",
    "₌": "=",
    "₍": "(",
    "₎": ")",
}
SYMBOL_REPLACEMENTS = {
    "−": "-",
    "–": "-",
    "—": "-",
    "×": "*",
    "⋅": "*",
    "·": "*",
    "÷": "/",
    "≤": "<=",
    "≥": ">=",
    "≠": "!=",
    "≈": "~=",
    "⇒": "=>",
    "→": "->",
    "←": "<-",
}
SYMBOL_ALIASES = {
    "∫": "integral 积分",
    "∬": "double integral 二重积分",
    "∭": "triple integral 三重积分",
    "∑": "summation 求和",
    "∏": "product 连乘",
    "√": "sqrt square root 根号 平方根",
    "∞": "infinity 无穷",
    "∂": "partial derivative 偏导数",
    "∇": "nabla gradient 梯度",
    "∆": "delta difference 差分",
    "Δ": "delta difference 差分",
    "∈": "belongs to element of 属于",
    "∉": "not belongs to 不属于",
    "⊂": "subset 子集",
    "∪": "union 并集",
    "∩": "intersection 交集",
}
MATH_SYMBOL_PATTERN = re.compile(r"[=+\-*/^_∫∬∭∑∏√∞≤≥≠≈→←⇒∂∇∆Δ∈∉⊂∪∩]")
MATH_FUNCTION_PATTERN = re.compile(
    r"\b(?:lim|sin|cos|tan|cot|sec|csc|ln|log|exp)\b",
    re.IGNORECASE,
)
DERIVATIVE_PATTERN = re.compile(
    r"(?:d\s*[a-zA-Z]\s*/\s*d\s*[a-zA-Z]|[a-zA-Z]['′″]{1,2}\s*\()"
)


def _replace_script_runs(text: str, mapping: dict[str, str], prefix: str) -> str:
    pattern = re.compile(f"[{re.escape(''.join(mapping))}]+")

    def replace(match: re.Match[str]) -> str:
        value = "".join(mapping[char] for char in match.group(0))
        return f"{prefix}({value})"

    return pattern.sub(replace, text)


def normalize_math_text(text: str) -> str:
    """Normalize equivalent mathematical glyphs without changing semantics."""
    normalized = _replace_script_runs(text or "", SUPERSCRIPT_MAP, "^")
    normalized = _replace_script_runs(normalized, SUBSCRIPT_MAP, "_")
    normalized = unicodedata.normalize("NFKC", normalized)
    for source, target in SYMBOL_REPLACEMENTS.items():
        normalized = normalized.replace(source, target)
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def math_aliases(text: str) -> list[str]:
    aliases = {
        alias
        for symbol, alias in SYMBOL_ALIASES.items()
        if symbol in (text or "")
    }
    if MATH_FUNCTION_PATTERN.search(text or ""):
        aliases.add("mathematical function 数学函数 极限 三角函数 对数")
    if DERIVATIVE_PATTERN.search(text or ""):
        aliases.add("derivative 导数 微分")
    if re.search(r"\^[({]?\s*2\s*[)}]?", normalize_math_text(text or "")):
        aliases.add("squared 平方 二次方")
    return sorted(aliases)


def build_math_search_text(text: str) -> str:
    """Return display text plus normalized forms and retrieval aliases."""
    display_text = (text or "").strip()
    normalized = normalize_math_text(display_text)
    aliases = math_aliases(display_text)
    parts = [display_text]
    if normalized and normalized != display_text:
        parts.append(normalized)
    if aliases:
        parts.append(" ".join(aliases))
    return "\n".join(part for part in parts if part)


def is_formula_candidate(text: str) -> bool:
    compact = re.sub(r"\s+", "", text or "")
    if not compact or len(compact) > 500:
        return False
    symbol_count = len(MATH_SYMBOL_PATTERN.findall(compact))
    signal_ratio = symbol_count / len(compact)
    return (
        symbol_count >= 2
        or signal_ratio >= 0.08
        or bool(MATH_FUNCTION_PATTERN.search(compact))
        or bool(DERIVATIVE_PATTERN.search(compact))
    )
