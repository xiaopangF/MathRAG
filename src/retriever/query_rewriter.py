"""
Rule-based query rewriting for MathRAG retrieval.

The rewriter expands short student questions with textbook-style aliases and
retrieval intent words. It is deliberately deterministic and cheap: no LLM call
is used, so retrieval remains fast and reproducible.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.loader.math_text import build_math_search_text


@dataclass(frozen=True)
class QueryRewriteResult:
    original: str
    search_text: str
    expansions: list[str]


TERM_ALIAS_RULES: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (
        (
            "洛必达",
            "罗必塔",
            "l'hospital",
            "l’hôpital",
            "l'hopital",
            "lhospital",
        ),
        ("洛必达法则", "罗必塔法则", "未定式", "0/0", "∞/∞", "极限", "导数"),
    ),
    (
        ("泰勒", "taylor"),
        ("泰勒公式", "泰勒展开", "Taylor formula", "余项", "麦克劳林公式", "局部近似"),
    ),
    (
        ("麦克劳林", "maclaurin"),
        ("麦克劳林公式", "Maclaurin formula", "泰勒公式", "泰勒展开", "零点展开"),
    ),
    (
        ("牛顿莱布尼茨", "牛顿-莱布尼茨", "微积分基本公式"),
        ("牛顿-莱布尼茨公式", "微积分基本定理", "定积分", "原函数", "变上限积分"),
    ),
    (
        ("分部积分",),
        ("分部积分法", "integration by parts", "乘积求导公式", "不定积分", "定积分"),
    ),
    (
        ("换元积分", "换元法"),
        ("换元积分法", "第一类换元法", "第二类换元法", "变量替换", "不定积分"),
    ),
    (
        ("傅里叶", "fourier"),
        ("傅里叶级数", "三角级数", "欧拉-傅里叶系数", "周期函数", "正弦余弦"),
    ),
    (
        ("幂级数",),
        ("函数项级数", "收敛半径", "收敛区间", "逐项求导", "逐项积分", "泰勒级数"),
    ),
    (
        ("函数项级数",),
        ("一致收敛", "逐项求导", "逐项积分", "幂级数", "傅里叶级数"),
    ),
    (
        ("格林公式", "green formula", "green theorem", "green's theorem"),
        ("Green 公式", "第二类曲线积分", "平面区域", "二重积分", "路径无关"),
    ),
    (
        (
            "高斯公式",
            "gauss formula",
            "gauss theorem",
            "gauss's theorem",
            "divergence theorem",
        ),
        ("Gauss 公式", "散度定理", "曲面积分", "三重积分", "通量"),
    ),
    (
        ("斯托克斯", "stokes formula", "stokes theorem", "stokes' theorem"),
        ("Stokes 公式", "曲线积分", "曲面积分", "旋度", "边界曲线"),
    ),
    (
        ("隐函数",),
        ("隐函数存在定理", "偏导数公式", "全微分", "雅可比", "方程组"),
    ),
    (
        ("极值", "最值"),
        ("驻点", "必要条件", "充分条件", "二阶偏导数", "Hessian", "无条件极值"),
    ),
    (
        ("方向导数",),
        ("梯度", "grad", "方向余弦", "偏导数", "最大方向导数"),
    ),
    (
        ("曲线积分",),
        ("第一类曲线积分", "第二类曲线积分", "路径无关", "弧长", "向量场"),
    ),
    (
        ("曲面积分",),
        ("第一类曲面积分", "第二类曲面积分", "通量", "法向量", "高斯公式"),
    ),
)


INTENT_RULES: tuple[tuple[re.Pattern[str], tuple[str, ...]], ...] = (
    (
        re.compile(
            r"(是什么|什么是|定义|概念|含义|怎么理解|what\s+is|definition|meaning)",
            re.IGNORECASE,
        ),
        ("定义", "概念", "含义", "解释"),
    ),
    (
        re.compile(
            r"(条件|什么时候|何时|适用|前提|要求|conditions?|when|applicable)",
            re.IGNORECASE,
        ),
        ("条件", "适用条件", "前提", "使用条件"),
    ),
    (
        re.compile(
            r"(怎么求|如何求|怎么算|计算|求法|步骤|方法|how\s+to|calculate|steps?|method)",
            re.IGNORECASE,
        ),
        ("计算", "方法", "步骤", "求法"),
    ),
    (
        re.compile(
            r"(几何意义|图形意义|直观意义|geometric\s+meaning)",
            re.IGNORECASE,
        ),
        ("几何意义", "直观解释", "图形意义"),
    ),
    (
        re.compile(
            r"(充分必要|充要|等价|necessary\s+and\s+sufficient|equivalent)",
            re.IGNORECASE,
        ),
        ("充分必要条件", "充要条件", "等价条件"),
    ),
    (
        re.compile(r"(证明|推导|由来|proof|prove|derive)", re.IGNORECASE),
        ("证明", "推导", "定理", "公式"),
    ),
)


def _dedupe(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        normalized = re.sub(r"\s+", " ", value.strip())
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(normalized)
    return result


def _contains_trigger(haystack: str, trigger: str) -> bool:
    if re.search(r"[A-Za-z]", trigger):
        pattern = rf"(?<![A-Za-z0-9]){re.escape(trigger)}(?![A-Za-z0-9])"
        return re.search(pattern, haystack, re.IGNORECASE) is not None
    return trigger in haystack


def rewrite_query(query: str) -> QueryRewriteResult:
    """Expand a user query into retrieval-oriented search text."""
    original = (query or "").strip()
    math_search_text = build_math_search_text(original)
    haystack = f"{original}\n{math_search_text}".lower()

    expansions: list[str] = []
    for triggers, aliases in TERM_ALIAS_RULES:
        if any(_contains_trigger(haystack, trigger) for trigger in triggers):
            expansions.extend(aliases)

    for pattern, aliases in INTENT_RULES:
        if pattern.search(original):
            expansions.extend(aliases)

    expansions = _dedupe(expansions)
    parts = [original]
    if math_search_text and math_search_text != original:
        parts.append(math_search_text)
    if expansions:
        parts.append(build_math_search_text(" ".join(expansions)))

    return QueryRewriteResult(
        original=original,
        search_text="\n".join(part for part in parts if part),
        expansions=expansions,
    )


def rewrite_query_for_retrieval(query: str) -> str:
    """Return only the rewritten search text for retrievers."""
    return rewrite_query(query).search_text
