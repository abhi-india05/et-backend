import csv
import glob
import os
import re
from typing import Any, Dict, Optional, List
from backend.utils.logger import get_logger
from backend.utils.helpers import generate_id, now_iso

logger = get_logger("scraping_tool")

INDUSTRY_PERSONAS = {
    "saas": ["VP of Engineering", "CTO", "Head of Product"],
    "healthcare": ["Chief Medical Officer", "VP Operations", "Director of IT"],
    "finance": ["CFO", "VP Finance", "Director of Technology"],
    "logistics": ["VP Supply Chain", "COO", "Director of Operations"],
    "retail": ["CMO", "VP eCommerce", "Director of Digital"],
    "manufacturing": ["VP Operations", "CTO", "Director of IT"],
    "default": ["CEO", "VP Sales", "Head of Operations"],
}

COMPANY_SIGNALS = {
    "hiring": "Company is actively hiring in key departments",
    "funding": "Recently secured funding round",
    "expansion": "Expanding into new markets",
    "product_launch": "Recently launched new product/service",
    "competitor_switch": "Evaluating alternatives to current vendor",
    "pain_point": "Public discussion of operational challenges",
}


def _env_int(name: str, default: int, *, minimum: int = 1, maximum: int = 1000) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        parsed = int(raw)
    except ValueError:
        return default
    return max(minimum, min(maximum, parsed))


def _normalize_company_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (value or "").lower()).strip()


def _company_candidates_from_row(row: Dict[str, Any]) -> List[str]:
    candidates = [
        row.get("company"),
        row.get("company_name"),
        row.get("experiences0company"),
        row.get("experiences1company"),
        row.get("occupation"),
        row.get("headline"),
    ]
    values = []
    for item in candidates:
        text = (item or "").strip()
        if text:
            values.append(text)
    return values


def _matches_company(requested_company: str, row: Dict[str, Any]) -> bool:
    return _company_match_score(requested_company, row) >= 0.5


def _company_match_score(requested_company: str, row: Dict[str, Any]) -> float:
    requested = _normalize_company_name(requested_company)
    if not requested:
        return 0.0

    request_tokens = {token for token in requested.split() if len(token) >= 2}
    best = 0.0

    for candidate in _company_candidates_from_row(row):
        norm_candidate = _normalize_company_name(candidate)
        if not norm_candidate:
            continue
        if requested == norm_candidate:
            return 1.0
        if requested in norm_candidate or norm_candidate in requested:
            best = max(best, 0.92)

        candidate_tokens = {token for token in norm_candidate.split() if len(token) >= 2}
        overlap = len(request_tokens.intersection(candidate_tokens))
        if overlap:
            token_ratio = overlap / max(len(request_tokens), 1)
            best = max(best, 0.45 + (0.45 * token_ratio))

    return round(min(best, 1.0), 4)


def _profile_data_density_score(row: Dict[str, Any]) -> float:
    profile_fields = [
        "full_name",
        "experiences0title",
        "experiences0company",
        "headline",
        "summary",
        "public_identifier",
        "experiences0description",
    ]
    populated = sum(bool(str(row.get(field) or "").strip()) for field in profile_fields)
    return round(populated / max(len(profile_fields), 1), 4)


def _row_rank_score(requested_company: str, row: Dict[str, Any]) -> float:
    company_score = _company_match_score(requested_company, row)
    if company_score <= 0.0:
        return 0.0

    density = _profile_data_density_score(row)
    title = (
        str(row.get("experiences0title") or "").strip().lower()
        or str(row.get("occupation") or "").strip().lower()
    )
    seniority_bonus = 0.06 if any(term in title for term in ["head", "vp", "chief", "director", "founder"]) else 0.0

    score = (company_score * 0.74) + (density * 0.2) + seniority_bonus
    return round(min(score, 1.0), 4)


def _build_dataset_paths() -> List[str]:
    paths: List[str] = []
    env_path = os.environ.get("LINKEDIN_CSV_PATH")
    if env_path:
        paths.append(os.path.abspath(env_path))

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    root_data_dir = os.path.abspath(os.path.join(os.getcwd(), "backend", "data"))

    glob_patterns = [
        os.path.join(base_dir, "*linkedin*.csv"),
        os.path.join(base_dir, "*10k*li*.csv"),
        os.path.join(base_dir, "*.csv"),
        os.path.join(root_data_dir, "*linkedin*.csv"),
        os.path.join(root_data_dir, "*10k*li*.csv"),
        os.path.join(root_data_dir, "*.csv"),
    ]
    for pattern in glob_patterns:
        paths.extend(glob.glob(pattern))

    deduped: List[str] = []
    seen = set()
    for path in paths:
        abs_path = os.path.abspath(path)
        if abs_path in seen:
            continue
        seen.add(abs_path)
        deduped.append(abs_path)
    return deduped


def _safe_snippet(value: str, max_len: int = 180) -> str:
    compact = re.sub(r"\s+", " ", (value or "").strip())
    if len(compact) <= max_len:
        return compact
    return compact[: max_len - 3].rstrip() + "..."


def _row_to_lead(row: Dict[str, Any], fallback_company: str, dataset_path: str) -> Dict[str, Any]:
    lead_id = generate_id("lead")
    full_name = (row.get("full_name") or "").strip()
    first_name = (row.get("first_name") or "").strip()
    last_name = (row.get("last_name") or "").strip()
    name = full_name or " ".join(part for part in [first_name, last_name] if part).strip() or "Unknown Lead"

    role = (
        (row.get("experiences0title") or "").strip()
        or (row.get("occupation") or "").strip()
        or (row.get("experiences1title") or "").strip()
        or ""
    )
    company = (
        (row.get("experiences0company") or "").strip()
        or (row.get("experiences1company") or "").strip()
        or (row.get("company") or "").strip()
        or fallback_company
    )
    headline = (row.get("headline") or "").strip()
    about = (row.get("summary") or "").strip()
    activity = (
        (row.get("experiences0description") or "").strip()
        or (row.get("experiences1description") or "").strip()
    )

    public_identifier = (row.get("public_identifier") or "").strip()
    linkedin = (
        (row.get("linkedin") or "").strip()
        or (row.get("linkedin_url") or "").strip()
        or (f"https://www.linkedin.com/in/{public_identifier}" if public_identifier else "")
    )
    email = ((row.get("email") or "").strip() or (row.get("work_email") or "").strip() or "")

    signals: List[str] = []
    if headline:
        signals.append(f"headline: {_safe_snippet(headline, 120)}")
    if activity:
        signals.append(f"activity: {_safe_snippet(activity, 120)}")
    if role:
        signals.append(f"role: {role}")

    completeness = sum(bool(value) for value in [role, company, headline, about, activity, linkedin])
    completeness_ratio = completeness / 6.0
    rank_score = float(row.get("_row_rank_score") or 0.0)
    score = round(min(0.99, 0.2 + (0.35 * completeness_ratio) + (0.45 * rank_score)), 2)

    return {
        "lead_id": lead_id,
        "id": lead_id,
        "name": name,
        "title": role or "",
        "role": role or "",
        "company": company,
        "email": email,
        "linkedin": linkedin,
        "linkedin_url": linkedin,
        "headline": headline,
        "about": about,
        "activity": activity,
        "source_profile": public_identifier,
        "source_dataset": os.path.basename(dataset_path),
        "raw_data": {
            k: v
            for k, v in row.items()
            if not str(k).startswith("_") and v is not None and str(v).strip() != ""
        },
        "score": score,
        "company_match_score": rank_score,
        "signals": signals,
        "enriched_at": now_iso(),
    }


def enrich_company(company: str, industry: str = "default") -> Dict[str, Any]:
    logger.info("Enriching company data", company=company, industry=industry)
    csv_paths = _build_dataset_paths()
    max_candidate_pool = _env_int("LINKEDIN_MAX_CANDIDATE_POOL", 120, minimum=20, maximum=600)
    max_returned_leads = _env_int(
        "LINKEDIN_MAX_RETURNED_LEADS",
        30,
        minimum=5,
        maximum=max_candidate_pool,
    )

    slug = re.sub(r'[^a-z0-9]', '-', company.lower())
    domain = f"{slug}.com"

    leads: List[Dict[str, Any]] = []
    matched_path = ""

    for path in csv_paths:
        if not os.path.exists(path):
            continue
        try:
            with open(path, newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                fieldnames = set(reader.fieldnames or [])
                if not fieldnames.intersection({"public_identifier", "full_name", "headline", "experiences0company"}):
                    continue

                matched_rows: List[Dict[str, Any]] = []
                scanned_rows = 0
                for row in reader:
                    scanned_rows += 1
                    row_score = _row_rank_score(company, row)
                    if row_score <= 0.0:
                        continue
                    ranked_row = dict(row)
                    ranked_row["_row_rank_score"] = row_score
                    matched_rows.append(ranked_row)

                if matched_rows:
                    matched_rows.sort(key=lambda item: float(item.get("_row_rank_score") or 0.0), reverse=True)
                    candidate_rows = matched_rows[:max_candidate_pool]
                    matched_path = path
                    leads = [_row_to_lead(row, company, path) for row in candidate_rows[:max_returned_leads]]
                    logger.info(
                        "Loaded leads from LinkedIn dataset",
                        path=path,
                        matches=len(leads),
                        candidate_pool=len(matched_rows),
                        scanned_rows=scanned_rows,
                    )
                    break
        except Exception as exc:
            logger.warning("linkedin_dataset_read_failed", path=path, error=str(exc))

    intent_signals: List[str] = []
    for lead in leads:
        for signal in lead.get("signals", []):
            if signal and signal not in intent_signals:
                intent_signals.append(signal)
            if len(intent_signals) >= 6:
                break
        if len(intent_signals) >= 6:
            break

    industry_key = industry.lower().split()[0] if industry else "default"
    employees_hint = "unknown"
    if industry_key in INDUSTRY_PERSONAS:
        employees_hint = "dataset_not_provided"

    return {
        "company": company,
        "domain": domain,
        "industry": industry,
        "estimated_employees": employees_hint,
        "tech_stack_signals": [],
        "leads": leads,
        "intent_signals": intent_signals,
        "data_source": os.path.basename(matched_path) if matched_path else "",
        "enriched_at": now_iso(),
    }


def search_company_news(company: str) -> Dict[str, Any]:
    logger.info("Searching company news", company=company)

    mock_news = [
        {
            "title": f"{company} announces Q4 growth strategy",
            "source": "TechCrunch",
            "date": now_iso()[:10],
            "summary": f"{company} is expanding operations and investing in new technology",
            "sentiment": "positive",
        },
        {
            "title": f"{company} faces competitive pressure in market",
            "source": "Bloomberg",
            "date": now_iso()[:10],
            "summary": "Market analysis shows increased competition in the segment",
            "sentiment": "neutral",
        },
    ]

    competitor_signals = []
    competitor_keywords = ["evaluating alternatives", "competitive", "switch", "migration"]
    for news in mock_news:
        for kw in competitor_keywords:
            if kw in news["summary"].lower():
                competitor_signals.append({
                    "signal": f"Competitor signal: {news['title']}",
                    "source": news["source"],
                    "severity": "medium",
                })

    return {
        "company": company,
        "news": mock_news,
        "competitor_signals": competitor_signals,
        "searched_at": now_iso(),
    }


def detect_intent_signals(company: str, account_data: Optional[Dict] = None) -> Dict[str, Any]:
    signals = []
    risk_factors = []

    if account_data:
        days_inactive = account_data.get("days_inactive", 0)
        health_score = account_data.get("health_score", 100)
        open_tickets = account_data.get("open_tickets", 0)
        logins = account_data.get("logins_last_30_days", 10)

        if days_inactive >= 10:
            risk_factors.append(f"No contact for {days_inactive} days")
        if health_score < 40:
            risk_factors.append(f"Low health score: {health_score}")
        if open_tickets > 5:
            risk_factors.append(f"High support load: {open_tickets} open tickets")
        if logins < 5:
            risk_factors.append(f"Low engagement: {logins} logins in 30 days")

        if health_score > 30 and days_inactive < 30:
            signals.append("Account shows signs of recovery potential")
        if logins > 0:
            signals.append("Buyer still logging in - relationship not lost")

    news_data = search_company_news(company)
    signals.extend([n["title"] for n in news_data["news"][:1]])
    risk_factors.extend([s["signal"] for s in news_data["competitor_signals"]])

    return {
        "company": company,
        "positive_signals": signals,
        "risk_factors": risk_factors,
        "competitor_activity": len(news_data["competitor_signals"]) > 0,
        "overall_sentiment": "at_risk" if len(risk_factors) > 2 else "neutral",
        "analyzed_at": now_iso(),
    }
