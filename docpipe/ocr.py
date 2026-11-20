"""OCR backends, selected by `parsing.ocr_engine`. The engine id is config, not code.

An engine turns one BGR image into `[[quad, text, score], ...]` — the observation shape
RapidOCR already returns, so the parse layer keeps no knowledge of which backend ran.

`is_ocr` is what the parse layer actually asks: not "which library" but "does this page
carry OCR-only evidence". It gates the label-gap re-read, the photo-quality flag and
`quality.ocr_used`, so a backend that deliberately reads nothing must not trigger them.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .config import load_config

Observation = Tuple[List[List[float]], str, float]


class OcrEngine:
    name: str = ""
    is_ocr: bool = True

    def recognize(self, image) -> List[Observation]:
        raise NotImplementedError


class RapidOcrEngine(OcrEngine):
    """RapidOCR on ONNX Runtime, CPU: the default and the only engine shipped in the offline bundle."""

    name = "rapidocr"

    def __init__(self) -> None:
        from rapidocr_onnxruntime import RapidOCR

        self._session = RapidOCR()

    def recognize(self, image) -> List[Observation]:
        result, _elapsed = self._session(image)
        return list(result or [])


class NullOcrEngine(OcrEngine):
    """Deliberately reads nothing, so the OCR leg itself becomes measurable.

    Running a batch through this keeps every other stage in place and answers "how much of
    what we deliver comes from OCR at all" — the baseline without which swapping backend
    cannot be priced. It is a diagnostic switch, not a production default.
    """

    name = "null"
    is_ocr = False

    def recognize(self, image) -> List[Observation]:
        return []


BUILDERS: Dict[str, type] = {
    RapidOcrEngine.name: RapidOcrEngine,
    NullOcrEngine.name: NullOcrEngine,
}


_INSTANCES: Dict[str, OcrEngine] = {}


def engine_names() -> List[str]:
    return sorted(BUILDERS)


def get_engine(config: Optional[Dict] = None, name: Optional[str] = None) -> OcrEngine:
    """Build once per id: an ONNX session costs seconds to load and must outlive one page."""
    engine_id = str(name if name is not None
                    else (config or load_config()).get("parsing", {}).get("ocr_engine", "rapidocr")).strip()
    if engine_id not in BUILDERS:
        raise ValueError(
            f"[OCR 引擎不可用] parsing.ocr_engine = {engine_id!r} 没有注册；"
            f"可选：{', '.join(engine_names())}。加新引擎请在 docpipe/ocr.py 注册一个类，"
            "不要让它悄悄退回默认引擎——静默退回会让指标看起来正常、实际读的是别的东西。")
    if engine_id not in _INSTANCES:
        _INSTANCES[engine_id] = BUILDERS[engine_id]()
    return _INSTANCES[engine_id]
