"""Whitelisted medical web retrieval, two-tier trust registry, PII query guard, and adapters.

Strict safety guarantees:
1. Only HTTPS URLs matching built-in `knowledge_base/web_sources.yaml` (Tier 1: `authoritative`)
   or user-added custom sources (Tier 2: `custom_unverified`, SSRF-checked) are allowed.
2. Queries ALWAYS pass PII detection & stripping before leaving the machine; pure PII blocks retrieval.
3. Offline mode (`ORTHO_OFFLINE_MODE=true`) disables all live and mock retrieval deterministically.
4. Anti-bot / CAPTCHA / login walls trigger transparent `blocked_or_unavailable` failure (never bypassed).
5. Retrieved snippets are strictly informational citations and NEVER participate in deterministic
   `rule_scanner` / `calculator` / `verify_draft_against_alerts` clinical safety decisions.
"""

from __future__ import annotations

import html
import ipaddress
import json
import os
import re
import socket
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import yaml
from pydantic import BaseModel, Field

from core_tools.schemas import (
    WebSourceCategory,
    WebSourcesCatalog,
    WebSourceSpec,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEB_SOURCES_PATH = PROJECT_ROOT / "knowledge_base" / "web_sources.yaml"

_USER_AGENT = "OrthoRehab-TCM-Harness/0.4.0 (+https://github.com/ortho-rehab-harness; medical-education-research)"
_HTTP_TIMEOUT_SECONDS = 6.0
_HTML_MAX_BYTES = 64 * 1024

_ANTI_BOT_MARKERS: Tuple[str, ...] = (
    "cf-browser-verification",
    "__cf_chl_",
    "cloudflare",
    "attention required",
    "verify you are human",
    "are you human",
    "hcaptcha",
    "g-recaptcha",
    "recaptcha",
    "turnstile",
    "altcha",
    "请先登录",
    "登录后查看",
    "身份验证",
    "访问受限",
    "captcha",
)

_STATUS_LABEL_ZH: Dict[str, str] = {
    "ok": "已检索",
    "empty": "未命中相关文献",
    "offline_disabled": "离线模式已禁用联网检索",
    "pii_blocked": "查询词含未脱敏隐私，已拦截检索",
    "blocked_or_unavailable": "目标站点反爬/登录墙或暂不可达，已按规则中止",
    "no_sources": "当前类别无可用启用来源",
}


class PlannedWebQuery(BaseModel):
    query: str = Field(min_length=2, max_length=120)
    category: WebSourceCategory = "guideline"
    reason: str = Field(default="", max_length=200)


class WebSearchPlan(BaseModel):
    need_search: bool = False
    rationale: str = Field(default="", max_length=300)
    queries: List[PlannedWebQuery] = Field(default_factory=list, max_length=2)


def is_offline_mode() -> bool:
    """Return True when ORTHO_OFFLINE_MODE is enabled."""
    val = (os.getenv("ORTHO_OFFLINE_MODE") or "").strip().lower()
    return val in ("1", "true", "yes", "on")


def get_custom_web_sources_path() -> Path:
    """Resolve user-customizable Tier-2 web sources JSON file (honoring ORTHO_APP_HOME / frozen exe dir)."""
    app_home = (os.getenv("ORTHO_APP_HOME") or "").strip()
    if app_home:
        return Path(app_home) / "custom_web_sources.json"
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent / "custom_web_sources.json"
    return PROJECT_ROOT / "custom_web_sources.json"


def load_authoritative_sources(
    yaml_path: Optional[Path] = None,
) -> List[WebSourceSpec]:
    """Load Tier-1 authoritative medical sources from knowledge_base/web_sources.yaml."""
    target = Path(yaml_path) if yaml_path else DEFAULT_WEB_SOURCES_PATH
    if not target.is_file():
        return []
    with target.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    catalog = WebSourcesCatalog.model_validate(raw)
    return list(catalog.sources)


def _validate_safe_public_https_url(raw_url: str) -> Tuple[str, str]:
    """Validate HTTPS URL and reject localhost, internal domains, and private/loopback IP literals."""
    cleaned = (raw_url or "").strip()
    parsed = urllib.parse.urlparse(cleaned)
    if parsed.scheme.lower() != "https" or not parsed.hostname:
        raise ValueError("自定义来源仅允许合法的 https:// 网址。")

    host = parsed.hostname.strip().lower().rstrip(".")
    if (
        not host
        or host == "localhost"
        or host.endswith(".localhost")
        or host.endswith(".local")
        or host.endswith(".internal")
        or host.endswith(".lan")
    ):
        raise ValueError(f"出于内网安全隔离策略，禁止添加本地或内网域名：{host}")

    try:
        ip_obj = ipaddress.ip_address(host)
        if (
            ip_obj.is_private
            or ip_obj.is_loopback
            or ip_obj.is_link_local
            or ip_obj.is_reserved
            or ip_obj.is_multicast
            or ip_obj.is_unspecified
        ):
            raise ValueError(f"出于内网安全隔离策略，禁止添加私有或环回 IP 地址：{host}")
    except ValueError as exc:
        if "禁止添加" in str(exc):
            raise
        # Not an IP literal; ensure hostname has at least one dot (FQDN)
        if "." not in host:
            raise ValueError(f"请输入完整的公网域名（HTTPS）：{host}") from exc

    return cleaned, host


def _ensure_resolved_host_is_public(host: str) -> None:
    """Resolve DNS before outbound HTTP request to block DNS-rebinding SSRF to internal IPs."""
    try:
        ip_obj = ipaddress.ip_address(host)
        if (
            ip_obj.is_private
            or ip_obj.is_loopback
            or ip_obj.is_link_local
            or ip_obj.is_reserved
            or ip_obj.is_multicast
        ):
            raise ValueError(f"拦截私有 IP 目标：{host}")
        return
    except ValueError as exc:
        if "拦截私有" in str(exc):
            raise

    try:
        infos = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
    except OSError:
        return

    for info in infos:
        addr = info[4][0]
        try:
            resolved_ip = ipaddress.ip_address(addr.split("%")[0])
            if (
                resolved_ip.is_private
                or resolved_ip.is_loopback
                or resolved_ip.is_link_local
                or resolved_ip.is_reserved
                or resolved_ip.is_multicast
            ):
                raise ValueError(f"目标域名解析为内网/环回地址，已安全拦截：{host}")
        except ValueError as exc:
            if "已安全拦截" in str(exc):
                raise


def load_custom_sources() -> List[WebSourceSpec]:
    """Load Tier-2 user-customizable web sources from custom_web_sources.json."""
    path = get_custom_web_sources_path()
    if not path.is_file():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(raw, list):
        return []
    out: List[WebSourceSpec] = []
    for item in raw:
        try:
            spec = WebSourceSpec.model_validate(item)
            spec.trust_level = "custom_unverified"
            spec.review_status = "draft_for_clinician_review"
            spec.adapter = "html_fetch"
            out.append(spec)
        except Exception:
            continue
    return out


def _save_custom_sources(items: List[WebSourceSpec]) -> None:
    path = get_custom_web_sources_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [item.model_dump(mode="json") for item in items]
    tmp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex[:6]}.tmp")
    try:
        tmp_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def add_custom_source(
    *,
    name: str,
    category: WebSourceCategory,
    base_url: str,
    description: str = "",
) -> WebSourceSpec:
    """Validate and persist a new Tier-2 custom web source (forced unverified & SSRF-checked)."""
    cleaned_url, host = _validate_safe_public_https_url(base_url)
    existing = load_custom_sources()
    source_id = f"CUSTOM_{uuid.uuid4().hex[:8].upper()}"
    spec = WebSourceSpec(
        source_id=source_id,
        name=name.strip(),
        category=category,
        domains=[host],
        adapter="html_fetch",
        base_url=cleaned_url,
        trust_level="custom_unverified",
        review_status="draft_for_clinician_review",
        enabled=True,
        description=(description or "").strip() or "用户自定义检索来源（未经人工审核）",
        probe_note="用户手动添加",
    )
    existing.append(spec)
    _save_custom_sources(existing)
    return spec


def toggle_custom_source(source_id: str, enabled: bool) -> WebSourceSpec:
    """Enable or disable a custom Tier-2 source by source_id."""
    items = load_custom_sources()
    for item in items:
        if item.source_id == source_id:
            item.enabled = bool(enabled)
            _save_custom_sources(items)
            return item
    raise KeyError(f"未找到自定义来源：{source_id}")


def delete_custom_source(source_id: str) -> bool:
    """Delete a custom Tier-2 source by source_id."""
    items = load_custom_sources()
    remaining = [item for item in items if item.source_id != source_id]
    if len(remaining) == len(items):
        return False
    _save_custom_sources(remaining)
    return True


def list_all_web_sources() -> Dict[str, Any]:
    """Return authoritative (Tier 1) and custom (Tier 2) sources alongside offline flag."""
    auth = load_authoritative_sources()
    custom = load_custom_sources()
    return {
        "offline_mode": is_offline_mode(),
        "authoritative_sources": [s.model_dump(mode="json") for s in auth],
        "custom_sources": [s.model_dump(mode="json") for s in custom],
    }


def sanitize_search_query(raw_query: str) -> Tuple[str, Dict[str, Any]]:
    """Strip all PII and bed/patient tokens from a search query before external retrieval."""
    from app_copilot.service import detect_potential_pii_details, sanitize_clinical_pii

    src = (raw_query or "").strip()
    if not src:
        return "", {"pii_detected": False, "pii_blocked": True, "redacted": False}

    before_details = detect_potential_pii_details(src)
    cleaned, counts = sanitize_clinical_pii(src)
    cleaned = re.sub(
        r"\[(?:姓名已脱敏|手机号已脱敏|身份证号已脱敏|身份证已脱敏|编号已脱敏)\]",
        " ",
        cleaned,
    )
    cleaned = re.sub(r"\d+\s*床", " ", cleaned)
    if counts.get("total", 0) > 0 or before_details.get("has_pii"):
        cleaned = re.sub(
            r"(?:患者姓名|姓名|病人|患者|联系人|家属|手机号|联系电话|电话|身份证号|住院号|病历号|门诊号|就诊卡号|医保号|床号)\s*[:：]?",
            " ",
            cleaned,
        )
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = re.sub(r"^[\s,，;；。:：]+|[\s,，;；。:：]+$", "", cleaned).strip()
    if not re.search(r"[A-Za-z0-9\u4e00-\u9fa5]", cleaned):
        cleaned = ""
    cleaned = cleaned[:120].strip()

    after_details = detect_potential_pii_details(cleaned)
    pii_blocked = (not cleaned) or bool(after_details.get("has_pii"))
    return cleaned, {
        "pii_detected": bool(before_details.get("has_pii")),
        "redacted": bool(counts.get("total", 0) > 0) or cleaned != src,
        "pii_blocked": pii_blocked,
    }


def detect_anti_bot_or_login_wall(status_code: int, body_text: str) -> Tuple[bool, str]:
    """Detect HTTP anti-bot/rate-limit/auth failures or HTML login/CAPTCHA walls (no bypass)."""
    if status_code in (401, 403, 412, 429, 503):
        return True, f"目标站点返回 HTTP {status_code}（反爬/限流或需要登录认证）"
    sample = (body_text or "")[:6000].lower()
    for marker in _ANTI_BOT_MARKERS:
        if marker in sample:
            return True, f"检测到登录墙或反爬验证拦截标记（{marker}），按安全规则中止抓取"
    return False, ""


def _http_get_text(url: str, *, allow_domains: List[str]) -> Tuple[int, str]:
    parsed = urllib.parse.urlparse(url)
    host = (parsed.hostname or "").strip().lower()
    if parsed.scheme.lower() != "https" or not any(
        host == d or host.endswith(f".{d}") for d in allow_domains
    ):
        raise ValueError(f"域名 {host} 不在白名单 {allow_domains} 内，已拦截。")
    _ensure_resolved_host_is_public(host)

    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": _USER_AGENT,
            "Accept": "application/json, application/xml, text/html;q=0.9, */*;q=0.8",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT_SECONDS) as resp:
            status = int(getattr(resp, "status", 200) or 200)
            raw_bytes = resp.read(_HTML_MAX_BYTES)
            text = raw_bytes.decode("utf-8", errors="replace")
            return status, text
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read(4096).decode("utf-8", errors="replace")
        except Exception:
            pass
        return int(exc.code or 500), body


def _normalize_item(
    *,
    source: WebSourceSpec,
    title: str,
    url: str,
    snippet: str,
    published_at: str = "",
) -> Dict[str, Any]:
    return {
        "source_id": source.source_id,
        "source_name": source.name,
        "trust_level": source.trust_level,
        "trust_label": "权威来源" if source.trust_level == "authoritative" else "用户自定义 · 未经审核",
        "category": source.category,
        "title": (title or source.name).strip()[:220],
        "url": url.strip(),
        "snippet": re.sub(r"\s+", " ", snippet or "").strip()[:360],
        "published_at": (published_at or "").strip()[:40],
        "mock": False,
    }


def _fetch_pubmed(source: WebSourceSpec, query: str, limit: int = 2) -> List[Dict[str, Any]]:
    encoded = urllib.parse.quote_plus(query)
    esearch_url = (
        f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        f"?db=pubmed&term={encoded}&retmax={limit}&sort=relevance&retmode=json"
    )
    status, body = _http_get_text(esearch_url, allow_domains=source.domains)
    blocked, reason = detect_anti_bot_or_login_wall(status, body)
    if blocked:
        raise PermissionError(reason)
    if status >= 400:
        raise RuntimeError(f"PubMed esearch HTTP {status}")

    data = json.loads(body)
    idlist = (data.get("esearchresult") or {}).get("idlist") or []
    if not idlist:
        return []

    ids_param = ",".join(str(pid) for pid in idlist[:limit])
    summary_url = (
        f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        f"?db=pubmed&id={ids_param}&retmode=json"
    )
    s_status, s_body = _http_get_text(summary_url, allow_domains=source.domains)
    s_blocked, s_reason = detect_anti_bot_or_login_wall(s_status, s_body)
    if s_blocked:
        raise PermissionError(s_reason)
    if s_status >= 400:
        raise RuntimeError(f"PubMed esummary HTTP {s_status}")

    s_data = json.loads(s_body).get("result") or {}
    items: List[Dict[str, Any]] = []
    for pid in idlist[:limit]:
        rec = s_data.get(str(pid)) or {}
        if not isinstance(rec, dict):
            continue
        title = str(rec.get("title") or f"PubMed PMID {pid}")
        pubdate = str(rec.get("pubdate") or "")
        journal = str(rec.get("fulljournalname") or rec.get("source") or "PubMed")
        items.append(
            _normalize_item(
                source=source,
                title=title,
                url=f"https://pubmed.ncbi.nlm.nih.gov/{pid}/",
                snippet=f"期刊：{journal} | 文献编号 PMID: {pid}",
                published_at=pubdate,
            )
        )
    return items


def _fetch_openfda(source: WebSourceSpec, query: str, limit: int = 2) -> List[Dict[str, Any]]:
    token = re.sub(r"[^A-Za-z0-9\-\s]", " ", query).strip() or query
    first_word = token.split()[0] if token.split() else token
    encoded = urllib.parse.quote(first_word)
    url = f"https://api.fda.gov/drug/label.json?search=openfda.generic_name:{encoded}+OR+openfda.brand_name:{encoded}&limit={limit}"
    status, body = _http_get_text(url, allow_domains=source.domains)
    blocked, reason = detect_anti_bot_or_login_wall(status, body)
    if blocked:
        raise PermissionError(reason)
    if status == 404:
        return []
    if status >= 400:
        raise RuntimeError(f"OpenFDA HTTP {status}")

    data = json.loads(body)
    results = data.get("results") or []
    items: List[Dict[str, Any]] = []
    for rec in results[:limit]:
        openfda = rec.get("openfda") or {}
        generic = ", ".join(openfda.get("generic_name") or []) or first_word
        brand = ", ".join(openfda.get("brand_name") or [])
        warn_list = rec.get("boxed_warning") or rec.get("contraindications") or rec.get("drug_interactions") or rec.get("warnings") or []
        snippet = str(warn_list[0]) if warn_list else "FDA 官方药品说明书安全条目"
        set_id = (openfda.get("spl_set_id") or [""])[0]
        target_url = (
            f"https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={set_id}"
            if set_id
            else "https://open.fda.gov/apis/drug/label/"
        )
        items.append(
            _normalize_item(
                source=source,
                title=f"FDA 药品说明书：{generic}" + (f" ({brand})" if brand else ""),
                url=target_url,
                snippet=snippet,
                published_at=str(rec.get("effective_time") or ""),
            )
        )
    return items


def _fetch_dailymed(source: WebSourceSpec, query: str, limit: int = 2) -> List[Dict[str, Any]]:
    token = re.sub(r"[^A-Za-z0-9\-\s]", " ", query).strip()
    drug = token.split()[0] if token.split() else query
    encoded = urllib.parse.quote_plus(drug)
    url = f"https://dailymed.nlm.nih.gov/dailymed/services/v2/spls.json?drug_name={encoded}&pagesize={limit}"
    status, body = _http_get_text(url, allow_domains=source.domains)
    blocked, reason = detect_anti_bot_or_login_wall(status, body)
    if blocked:
        raise PermissionError(reason)
    if status >= 400:
        raise RuntimeError(f"DailyMed HTTP {status}")

    records = (json.loads(body).get("data") or [])[:limit]
    return [
        _normalize_item(
            source=source,
            title=str(rec.get("title") or f"DailyMed SPL {drug}"),
            url=f"https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={rec.get('setid', '')}",
            snippet=f"DailyMed SPL 官方药品专论（Version {rec.get('spl_version', '-')})",
            published_at=str(rec.get("published_date") or ""),
        )
        for rec in records
        if rec.get("setid")
    ]


def _fetch_medlineplus(source: WebSourceSpec, query: str, limit: int = 2) -> List[Dict[str, Any]]:
    encoded = urllib.parse.quote_plus(query)
    url = f"https://wsearch.nlm.nih.gov/ws/query?db=healthTopics&term={encoded}&retmax={limit}"
    status, body = _http_get_text(url, allow_domains=source.domains)
    blocked, reason = detect_anti_bot_or_login_wall(status, body)
    if blocked:
        raise PermissionError(reason)
    if status >= 400:
        raise RuntimeError(f"MedlinePlus HTTP {status}")

    root = ET.fromstring(body)
    items: List[Dict[str, Any]] = []
    for doc in root.findall(".//document")[:limit]:
        doc_url = str(doc.attrib.get("url") or "https://medlineplus.gov/")
        title = ""
        summary = ""
        for content in doc.findall("content"):
            cname = content.attrib.get("name", "")
            raw_text = re.sub(r"<[^>]+>", "", "".join(content.itertext()))
            if cname == "title" and not title:
                title = html.unescape(raw_text)
            elif cname in ("FullSummary", "snippet") and not summary:
                summary = html.unescape(raw_text)
        items.append(
            _normalize_item(
                source=source,
                title=title or "MedlinePlus Health Topic",
                url=doc_url,
                snippet=summary or "NIH MedlinePlus 权威健康专题参考",
            )
        )
    return items


def _fetch_crossref(source: WebSourceSpec, query: str, limit: int = 2) -> List[Dict[str, Any]]:
    encoded = urllib.parse.quote_plus(query)
    url = f"https://api.crossref.org/works?query={encoded}&rows={limit}&select=DOI,title,published-print,container-title"
    status, body = _http_get_text(url, allow_domains=source.domains)
    blocked, reason = detect_anti_bot_or_login_wall(status, body)
    if blocked:
        raise PermissionError(reason)
    if status >= 400:
        raise RuntimeError(f"Crossref HTTP {status}")

    works = ((json.loads(body).get("message") or {}).get("items") or [])[:limit]
    items: List[Dict[str, Any]] = []
    for w in works:
        doi = str(w.get("DOI") or "")
        titles = w.get("title") or []
        journals = w.get("container-title") or []
        parts = ((w.get("published-print") or {}).get("date-parts") or [[]])[0]
        pub = "-".join(str(p) for p in parts[:2]) if parts else ""
        items.append(
            _normalize_item(
                source=source,
                title=str(titles[0] if titles else doi or "Crossref Record"),
                url=f"https://doi.org/{doi}" if doi else "https://api.crossref.org/",
                snippet=f"期刊：{journals[0] if journals else 'Peer-reviewed'} | DOI: {doi}",
                published_at=pub,
            )
        )
    return items


def _fetch_clinicaltrials(source: WebSourceSpec, query: str, limit: int = 2) -> List[Dict[str, Any]]:
    encoded = urllib.parse.quote_plus(query)
    url = f"https://clinicaltrials.gov/api/v2/studies?query.term={encoded}&pageSize={limit}"
    status, body = _http_get_text(url, allow_domains=source.domains)
    blocked, reason = detect_anti_bot_or_login_wall(status, body)
    if blocked:
        raise PermissionError(reason)
    if status >= 400:
        raise RuntimeError(f"ClinicalTrials HTTP {status}")

    studies = (json.loads(body).get("studies") or [])[:limit]
    items: List[Dict[str, Any]] = []
    for st in studies:
        proto = st.get("protocolSection") or {}
        ident = proto.get("identificationModule") or {}
        status_mod = proto.get("statusModule") or {}
        nct = str(ident.get("nctId") or "")
        title = str(ident.get("briefTitle") or ident.get("officialTitle") or nct)
        overall = str(status_mod.get("overallStatus") or "")
        if nct:
            items.append(
                _normalize_item(
                    source=source,
                    title=title,
                    url=f"https://clinicaltrials.gov/study/{nct}",
                    snippet=f"临床研究 {nct}（状态：{overall or '登记在案'}）",
                    published_at=str((status_mod.get("lastUpdatePostDateStruct") or {}).get("date") or ""),
                )
            )
    return items


def _fetch_html(source: WebSourceSpec, query: str) -> List[Dict[str, Any]]:
    status, body = _http_get_text(source.base_url, allow_domains=source.domains)
    blocked, reason = detect_anti_bot_or_login_wall(status, body)
    if blocked:
        raise PermissionError(reason)
    if status >= 400:
        raise RuntimeError(f"{source.name} HTTP {status}")

    title_m = re.search(r"<title[^>]*>([\s\S]*?)</title>", body, flags=re.IGNORECASE)
    page_title = html.unescape(re.sub(r"\s+", " ", title_m.group(1))).strip() if title_m else source.name
    cleaned = re.sub(r"<(script|style|noscript)[^>]*>[\s\S]*?</\1>", " ", body, flags=re.IGNORECASE)
    cleaned = html.unescape(re.sub(r"<[^>]+>", " ", cleaned))
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return [
        _normalize_item(
            source=source,
            title=f"{page_title}（自定义检索词：{query}）",
            url=source.base_url,
            snippet=cleaned[:320] or f"来自自定义来源 {source.base_url}",
        )
    ]


def _execute_live_source(source: WebSourceSpec, query: str) -> List[Dict[str, Any]]:
    if source.adapter == "pubmed_esearch":
        return _fetch_pubmed(source, query)
    if source.adapter == "openfda_label":
        return _fetch_openfda(source, query)
    if source.adapter == "dailymed_spl":
        return _fetch_dailymed(source, query)
    if source.adapter == "medlineplus_ws":
        return _fetch_medlineplus(source, query)
    if source.adapter == "crossref_works":
        return _fetch_crossref(source, query)
    if source.adapter == "clinicaltrials_v2":
        return _fetch_clinicaltrials(source, query)
    return _fetch_html(source, query)


def _mock_citations_for_query(
    query: str,
    sources: List[WebSourceSpec],
) -> List[Dict[str, Any]]:
    """Generate offline deterministic mock citations grounded in enabled whitelist sources."""
    if not sources:
        return []
    q_lower = query.lower()
    out: List[Dict[str, Any]] = []
    for src in sources[:2]:
        if src.category == "drug":
            title = f"{src.name} 药品相互作用与禁忌参考：{query}"
            snippet = (
                "【Mock 仿真文献摘录】参考 FDA/DailyMed 药品说明书框架：活血化瘀类中药或 NSAIDs 与抗凝药"
                "（如华法林/利伐沙班）联用可显著升高消化道与系统出血风险，具体禁忌仍以本地确定性规则引擎为准。"
            )
        else:
            title = f"{src.name} 临床康复指南摘要：{query}"
            snippet = (
                f"【Mock 仿真文献摘录】针对“{query}”的循证康复综述提示：优先排查颈脊髓受压、马尾征象与急性血管危险征象；"
                "若存在渐进性神经功能缺损，应立即脊柱外科/骨科评估，禁止盲目高速旋转手法。"
            )
            if "womac" in q_lower or "osteoarthritis" in q_lower:
                snippet = (
                    "【Mock 仿真文献摘录】骨关节炎国际指南建议优先采用患者教育、可控肌力训练与减重管理，定期复评 WOMAC 疼痛与功能维度。"
                )
        out.append(
            {
                "source_id": src.source_id,
                "source_name": src.name,
                "trust_level": src.trust_level,
                "trust_label": (
                    "权威来源" if src.trust_level == "authoritative" else "用户自定义 · 未经审核"
                ),
                "category": src.category,
                "title": title,
                "url": src.base_url,
                "snippet": snippet,
                "published_at": "2025-01 (Mock)",
                "mock": True,
            }
        )
    return out


def plan_web_queries(
    user_input: str,
    extracted_note: Optional[Dict[str, Any]],
    unified_alerts: Optional[List[Dict[str, Any]]],
    *,
    client: Any = None,
    model: Optional[str] = None,
    mock_mode: bool = False,
) -> WebSearchPlan:
    """Autonomously decide whether external medical search adds value and construct clean queries."""
    cleaned_input = (user_input or "").strip()
    if not cleaned_input:
        return WebSearchPlan(need_search=False, rationale="空输入无需检索。")

    if mock_mode or client is None:
        text = cleaned_input
        queries: List[PlannedWebQuery] = []
        if any(k in text for k in ("踩棉花", "脊髓", "颈椎", "颈痛", "束带感", "霍夫曼")):
            queries.append(
                PlannedWebQuery(
                    query="cervical spondylotic myelopathy rehabilitation contraindications guideline",
                    category="guideline",
                    reason="检索脊髓型颈椎病及康复手法禁忌最新循证指南",
                )
            )
        if any(k in text for k in ("华法林", "利伐沙班", "阿司匹林", "抗凝", "三七", "活血", "布洛芬", "双氯芬酸")):
            queries.append(
                PlannedWebQuery(
                    query="warfarin NSAIDs herbal bleeding interaction",
                    category="drug",
                    reason="检索抗凝药与止痛/活血药物相互作用官方安全说明",
                )
            )
        if not queries and any(k in text for k in ("腰椎", "马尾", "膝关节", "骨关节炎", "骨折", "术后", "疼痛", "康复")):
            queries.append(
                PlannedWebQuery(
                    query="orthopedic rehabilitation red flags conservative management guideline",
                    category="guideline",
                    reason="补充骨伤康复评估与循证指南参考",
                )
            )
        if not queries:
            return WebSearchPlan(
                need_search=False,
                rationale="常规寒暄或非医学指令，模型判断本轮无需触发外部医学检索。",
                queries=[],
            )
        return WebSearchPlan(
            need_search=True,
            rationale="模型判定需要补充最新权威指南与药品说明书作为临床背景参考。",
            queries=queries[:2],
        )

    alert_names = [str(a.get("name") or "") for a in (unified_alerts or []) if a.get("name")]
    meds = list((extracted_note or {}).get("medications") or [])
    sanitized_preview, _ = sanitize_search_query(cleaned_input)
    prompt = (
        "你正在为中医骨伤/康复临床医师生成辅助参考。请判断本轮是否需要检索权威医学指南(guideline)或药品说明书(drug)。\n"
        "严禁在检索词中出现患者姓名、床号、电话、身份证号、医院名等隐私。\n"
        f"脱敏症状摘要：{sanitized_preview}\n"
        f"已识别药物：{meds[:5]}\n"
        f"已触发本地安全告警：{alert_names[:4]}\n"
        "请输出严格符合 JSON Schema 的检索计划：仅在确实需要外部文献或药品说明书佐证时设 need_search=true，给出最多 2 条英文/专业术语 query。"
    )
    try:
        res = client.extract_structured_json(
            [{"role": "user", "content": prompt}],
            schema_model=WebSearchPlan,
            model=model,
            mock_mode=False,
        )
        if res.get("status") == "ok" and isinstance(res.get("data"), dict):
            return WebSearchPlan.model_validate(res["data"])
    except Exception:
        pass
    return WebSearchPlan(
        need_search=False,
        rationale="自主检索规划未返回有效计划，本轮跳过外部检索。",
        queries=[],
    )


def run_autonomous_web_search(
    *,
    user_input: str,
    extracted_note: Optional[Dict[str, Any]] = None,
    unified_alerts: Optional[List[Dict[str, Any]]] = None,
    client: Any = None,
    model: Optional[str] = None,
    mock_mode: bool = False,
) -> List[Dict[str, Any]]:
    """Run D9 whitelisted autonomous web search and return structured trace-ready fetch records."""
    if is_offline_mode():
        return [
            {
                "query_id": f"wf_{uuid.uuid4().hex[:8]}",
                "raw_query": "",
                "sanitized_query": "",
                "category": "guideline",
                "rationale": "全局离线模式已开启（ORTHO_OFFLINE_MODE=true）",
                "pii_redacted": False,
                "status": "offline_disabled",
                "status_label": _STATUS_LABEL_ZH["offline_disabled"],
                "source_ids": [],
                "items": [],
                "errors": [],
                "latency_ms": 0.0,
            }
        ]

    plan = plan_web_queries(
        user_input,
        extracted_note,
        unified_alerts,
        client=client,
        model=model,
        mock_mode=mock_mode,
    )
    if not plan.need_search or not plan.queries:
        return []

    all_sources = [s for s in [*load_authoritative_sources(), *load_custom_sources()] if s.enabled]
    records: List[Dict[str, Any]] = []

    for planned in plan.queries[:2]:
        t0 = time.perf_counter()
        query_id = f"wf_{uuid.uuid4().hex[:8]}"
        sanitized, pii_meta = sanitize_search_query(planned.query)
        if pii_meta.get("pii_blocked"):
            records.append(
                {
                    "query_id": query_id,
                    "raw_query": "[已脱敏隐藏]",
                    "sanitized_query": "",
                    "category": planned.category,
                    "rationale": planned.reason or plan.rationale,
                    "pii_redacted": True,
                    "status": "pii_blocked",
                    "status_label": _STATUS_LABEL_ZH["pii_blocked"],
                    "source_ids": [],
                    "items": [],
                    "errors": ["查询词包含未脱敏隐私或清空后为空，禁止对外发送检索。"],
                    "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                }
            )
            continue

        cat_sources = [s for s in all_sources if s.category == planned.category]
        if not cat_sources:
            cat_sources = list(all_sources)

        if not cat_sources:
            records.append(
                {
                    "query_id": query_id,
                    "raw_query": sanitized,
                    "sanitized_query": sanitized,
                    "category": planned.category,
                    "rationale": planned.reason or plan.rationale,
                    "pii_redacted": bool(pii_meta.get("redacted")),
                    "status": "no_sources",
                    "status_label": _STATUS_LABEL_ZH["no_sources"],
                    "source_ids": [],
                    "items": [],
                    "errors": ["当前白名单下没有启用的检索来源。"],
                    "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                }
            )
            continue

        target_sources = cat_sources[:2]
        if mock_mode:
            items = _mock_citations_for_query(sanitized, target_sources)
            records.append(
                {
                    "query_id": query_id,
                    "raw_query": sanitized,
                    "sanitized_query": sanitized,
                    "category": planned.category,
                    "rationale": planned.reason or plan.rationale,
                    "pii_redacted": bool(pii_meta.get("redacted")),
                    "status": "ok" if items else "empty",
                    "status_label": _STATUS_LABEL_ZH["ok" if items else "empty"],
                    "source_ids": [s.source_id for s in target_sources],
                    "items": items,
                    "errors": [],
                    "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                }
            )
            continue

        collected: List[Dict[str, Any]] = []
        errors: List[str] = []
        saw_blocked = False
        for src in target_sources:
            try:
                collected.extend(_execute_live_source(src, sanitized))
            except PermissionError as exc:
                saw_blocked = True
                errors.append(f"{src.name}: {exc}")
            except Exception as exc:
                errors.append(f"{src.name}: {exc}")

        if collected:
            status: Literal["ok", "empty", "blocked_or_unavailable"] = "ok"
        elif saw_blocked or errors:
            status = "blocked_or_unavailable"
        else:
            status = "empty"

        records.append(
            {
                "query_id": query_id,
                "raw_query": sanitized,
                "sanitized_query": sanitized,
                "category": planned.category,
                "rationale": planned.reason or plan.rationale,
                "pii_redacted": bool(pii_meta.get("redacted")),
                "status": status,
                "status_label": _STATUS_LABEL_ZH[status],
                "source_ids": [s.source_id for s in target_sources],
                "items": collected,
                "errors": errors,
                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            }
        )

    return records


def format_web_fetches_for_prompt(web_fetches: Optional[List[Dict[str, Any]]]) -> str:
    """Format retrieved external citations cleanly as non-binding reference context for LLM generation."""
    blocks: List[str] = []
    for rec in web_fetches or []:
        if rec.get("status") != "ok":
            continue
        for item in rec.get("items") or []:
            trust = (
                "权威来源"
                if item.get("trust_level") == "authoritative"
                else "用户自定义 · 未经审核"
            )
            blocks.append(
                f"- [{trust} | {item.get('source_name', '')}] {item.get('title', '')} "
                f"({item.get('url', '')})\n  摘要：{item.get('snippet', '')}"
            )
    if not blocks:
        return ""
    return (
        "【外部联网检索参考文献（仅供参考引用，严禁以此覆盖或推翻本地确定性红旗/用药冲突护栏）】\n"
        + "\n".join(blocks)
    )
