"""Unified multi-provider reasoning/thinking capability registry & parameter mapping.

Synthesizes patterns from LiteLLM, OpenRouter, DashScope, Anthropic, and Ollama:
- Always-on reasoners (e.g. deepseek-reasoner) keep default reasoning without extra tuning flags.
- OpenAI reasoning models (o1/o3/o4/gpt-5) map low/medium/high -> reasoning_effort.
- Anthropic extended thinking models map low/medium/high -> thinking={type, budget_tokens}.
- DashScope Qwen hybrid-thinking models map -> extra_body={enable_thinking, thinking_budget}.
- Ollama reasoning models map -> think=True/False with allowed_openai_params=['think'].
- Standard non-reasoning models return {} so unsupported params never trigger provider 400 errors.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Literal, Optional

ThinkingLevel = Literal["off", "low", "medium", "high"]
VALID_THINKING_LEVELS: tuple[ThinkingLevel, ...] = ("off", "low", "medium", "high")

_ANTHROPIC_BUDGET_MAP: Dict[str, int] = {
    "low": 1024,
    "medium": 4096,
    "high": 10000,
}

_DASHSCOPE_BUDGET_MAP: Dict[str, int] = {
    "low": 1024,
    "medium": 4096,
    "high": 8192,
}


def normalize_thinking_level(level: Optional[str] = None) -> ThinkingLevel:
    cleaned = str(level or "off").strip().lower()
    if cleaned in ("off", "none", "disabled", "false", "0"):
        return "off"
    if cleaned in ("low", "minimal"):
        return "low"
    if cleaned in ("medium", "mid"):
        return "medium"
    if cleaned in ("high", "max", "xhigh"):
        return "high"
    return "off"


def _normalize_model_id(model: str) -> str:
    raw = str(model or "").strip().lower()
    if raw.startswith("mock/"):
        raw = raw[5:]
    return raw


def get_model_capabilities(model: Optional[str]) -> Dict[str, Any]:
    """Return deterministic reasoning capability metadata for a model identifier."""
    norm = _normalize_model_id(model or "")
    display = str(model or "").strip()

    def _pack(
        supports_thinking: bool,
        mechanism: str,
        supported_levels: List[ThinkingLevel],
        default_level: ThinkingLevel,
        returns_reasoning_text: bool,
        note: str,
    ) -> Dict[str, Any]:
        return {
            "model": display,
            "supports_thinking": supports_thinking,
            "mechanism": mechanism,
            "supported_levels": supported_levels,
            "default_level": default_level,
            "returns_reasoning_text": returns_reasoning_text,
            "note": note,
        }

    if not norm:
        return _pack(False, "none", ["off"], "off", False, "未指定模型，思考调节不可用。")

    # 1. Ollama local models
    if norm.startswith(("ollama/", "ollama_chat/")):
        if any(tag in norm for tag in ("deepseek-r1", "qwq", "qwen3", "gpt-oss", "thinking")):
            return _pack(
                True,
                "ollama_think",
                ["off", "low", "medium", "high"],
                "medium",
                True,
                "支持 Ollama think 思考开关，返回 <think> 原始推理过程。",
            )
        return _pack(
            False,
            "none",
            ["off"],
            "off",
            False,
            "当前本地 Ollama 模型未启用思考推理模式，思考强度已置灰。",
        )

    # 2. DeepSeek family
    if "deepseek-reasoner" in norm or (
        "deepseek-r1" in norm and not norm.startswith(("ollama/", "ollama_chat/"))
    ):
        return _pack(
            True,
            "always_on",
            ["medium"],
            "medium",
            True,
            "DeepSeek 原生推理模型，固定开启思考并返回 reasoning_content。",
        )
    if "deepseek-chat" in norm or "deepseek-v3" in norm:
        return _pack(
            False,
            "none",
            ["off"],
            "off",
            False,
            "DeepSeek-Chat 为标准对话模型，不支持思考强度调节。",
        )

    # 3. OpenAI o-series / gpt-5 reasoning models vs gpt-4*
    bare_tail = norm.split("/")[-1]
    if bare_tail.startswith(("o1", "o3", "o4", "gpt-5")):
        return _pack(
            True,
            "reasoning_effort",
            ["low", "medium", "high"],
            "medium",
            False,
            "支持 reasoning_effort（低/中/高）；OpenAI 推理系列通常不返回原始思维链文本。",
        )
    if bare_tail.startswith(("gpt-4", "gpt-3.5")):
        return _pack(
            False,
            "none",
            ["off"],
            "off",
            False,
            "标准对话模型（GPT-4 系列）不支持思考强度参数。",
        )

    # 4. Anthropic Claude 3.7+ / 4*
    if "claude-3-5" in norm:
        return _pack(
            False,
            "none",
            ["off"],
            "off",
            False,
            "Claude 3.5 系列不支持扩展思考模式。",
        )
    if any(tag in norm for tag in ("claude-3-7", "claude-sonnet-4", "claude-opus-4", "claude-4")):
        return _pack(
            True,
            "anthropic_thinking",
            ["off", "low", "medium", "high"],
            "off",
            True,
            "支持 Anthropic Extended Thinking（按 budget_tokens 映射关/低/中/高）。",
        )

    # 5. Google Gemini 2.5+ / 3*
    if "gemini-2.5" in norm or "gemini-3" in norm:
        return _pack(
            True,
            "reasoning_effort",
            ["off", "low", "medium", "high"],
            "medium",
            True,
            "支持 Gemini 思考深度调节（映射到 reasoning_effort / thinking_config）。",
        )

    # 6. Alibaba DashScope Qwen hybrid-thinking family
    if norm.startswith("dashscope/") or "qwen3" in norm or "qwq" in norm:
        if any(tag in norm for tag in ("qwen3", "qwen-plus", "qwen-max", "qwq")):
            return _pack(
                True,
                "dashscope_thinking",
                ["off", "low", "medium", "high"],
                "off",
                True,
                "支持阿里云百炼混合思考（enable_thinking + thinking_budget 映射）。",
            )

    # 7. Other known reasoners (GLM-Z1 / GLM-4.5 / Kimi thinking)
    if any(tag in norm for tag in ("glm-z1", "glm-4.5", "kimi-k2-thinking", "reasoner", "thinking")):
        return _pack(
            True,
            "reasoning_effort",
            ["off", "low", "medium", "high"],
            "medium",
            True,
            "支持 reasoning_effort 推理强度控制。",
        )
    if "moonshot-v1" in norm:
        return _pack(
            False,
            "none",
            ["off"],
            "off",
            False,
            "Moonshot v1 标准模型不支持思考强度调节。",
        )

    # 8. Second-layer fallback via LiteLLM registry if installed
    try:
        import litellm  # type: ignore[import-not-found]

        if hasattr(litellm, "supports_reasoning") and litellm.supports_reasoning(model=norm):
            return _pack(
                True,
                "reasoning_effort",
                ["off", "low", "medium", "high"],
                "medium",
                True,
                "通过 LiteLLM 能力探测支持 reasoning_effort 调节。",
            )
    except Exception:
        pass

    return _pack(
        False,
        "none",
        ["off"],
        "off",
        False,
        "该模型未声明支持思考强度参数，已自动置灰以免请求报错。",
    )


def build_reasoning_kwargs(model: Optional[str], thinking: Optional[str] = "off") -> Dict[str, Any]:
    """Translate unified thinking level ('off'|'low'|'medium'|'high') into provider-specific LiteLLM kwargs."""
    cap = get_model_capabilities(model)
    if not cap.get("supports_thinking"):
        return {}

    mechanism = cap.get("mechanism", "none")
    level = normalize_thinking_level(thinking)

    if mechanism in ("none", "always_on"):
        return {}

    if mechanism == "reasoning_effort":
        if level == "off":
            return {}
        return {"reasoning_effort": level}

    if mechanism == "anthropic_thinking":
        if level == "off":
            return {}
        budget = _ANTHROPIC_BUDGET_MAP.get(level, 4096)
        return {"thinking": {"type": "enabled", "budget_tokens": budget}}

    if mechanism == "dashscope_thinking":
        if level == "off":
            return {"extra_body": {"enable_thinking": False}}
        return {
            "extra_body": {
                "enable_thinking": True,
                "thinking_budget": _DASHSCOPE_BUDGET_MAP.get(level, 4096),
            }
        }

    if mechanism == "ollama_think":
        return {
            "think": level != "off",
            "allowed_openai_params": ["think"],
        }

    return {}


def list_model_capabilities(models: Iterable[str]) -> Dict[str, Dict[str, Any]]:
    """Build capability dictionary keyed by model string for /api/v1/settings/llm."""
    out: Dict[str, Dict[str, Any]] = {}
    for m in models:
        cleaned = str(m or "").strip()
        if cleaned and cleaned not in out:
            out[cleaned] = get_model_capabilities(cleaned)
    return out
