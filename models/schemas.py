from __future__ import annotations

import re
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from backend.auth.passwords import validate_password_strength
from backend.utils.helpers import sanitize_text, utcnow

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class AgentStatus(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    PENDING = "pending"
    RETRYING = "retrying"
    ESCALATED = "escalated"


class AgentResponse(StrictBaseModel):
    status: AgentStatus
    data: Dict[str, Any] = Field(default_factory=dict)
    reasoning: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    agent_name: str = ""
    timestamp: datetime = Field(default_factory=utcnow)
    retry_count: int = 0
    error: Optional[str] = None
    tools_used: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ExecutionPlan(StrictBaseModel):
    task_type: str
    allowed_tools: List[str]
    steps: List[str]
    fallback_strategy: str
    product_context: Dict[str, str] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utcnow)


class WorkflowValidation(StrictBaseModel):
    valid: bool = True
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)


class Lead(StrictBaseModel):
    id: Optional[str] = None
    name: str
    title: str
    role: Optional[str] = None
    company: str
    email: Optional[str] = None
    linkedin: Optional[str] = None
    linkedin_url: Optional[str] = None
    headline: Optional[str] = None
    about: Optional[str] = None
    activity: Optional[str] = None
    source_profile: Optional[str] = None
    raw_data: Optional[Dict[str, Any]] = None
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    signals: List[str] = Field(default_factory=list)
    pain_points: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _sync_linkedin_fields(self) -> "Lead":
        if not self.linkedin_url and self.linkedin:
            self.linkedin_url = self.linkedin
        if not self.linkedin and self.linkedin_url:
            self.linkedin = self.linkedin_url
        return self


class EmailExplanation(StrictBaseModel):
    used_fields: List[str] = Field(default_factory=list)
    insight: str = ""
    reasoning: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class EmailStep(StrictBaseModel):
    step: int = Field(ge=1, le=10)
    send_day: int = Field(ge=1, le=30)
    subject: str
    body: str
    email: str = ""
    cta: str
    angle: str
    explanation: Optional[EmailExplanation] = None

    @model_validator(mode="after")
    def _sync_email_and_body(self) -> "EmailStep":
        if not self.email and self.body:
            self.email = self.body
        return self


class EmailSequenceResult(StrictBaseModel):
    lead_id: Optional[str] = None
    lead_name: str
    lead_email: str = ""
    sequence_id: str
    emails: List[EmailStep]
    sequence_strategy: str
    predicted_open_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    predicted_reply_rate: float = Field(default=0.0, ge=0.0, le=1.0)


class DealRisk(StrictBaseModel):
    deal_id: str
    company: str
    risk_level: str
    risk_score: float = Field(default=0.0, ge=0.0, le=1.0)
    risk_signals: List[str] = Field(default_factory=list)
    competitor_threat: bool = False
    competitor_name: Optional[str] = None
    deal_velocity: str = "stalled"
    days_inactive: int = Field(default=0, ge=0)
    recovery_strategy: str
    recommended_actions: List[str] = Field(default_factory=list)
    escalate_to_manager: bool = False
    predicted_close_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    reasoning: str = ""


class ChurnRisk(StrictBaseModel):
    account_id: str
    company: str
    churn_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    risk_factors: List[str] = Field(default_factory=list)
    retention_strategy: str
    urgency: str


class OutreachRequest(StrictBaseModel):
    session_id: Optional[str] = None
    company: str
    industry: str
    size: str
    website: Optional[str] = None
    notes: Optional[str] = None
    product_name: Optional[str] = None
    product_description: Optional[str] = None
    auto_send: bool = False

    @field_validator("company", "industry", "size")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = sanitize_text(value, max_len=200)
        if not cleaned:
            raise ValueError("Field is required")
        return cleaned

    @field_validator("website")
    @classmethod
    def _website(cls, value: Optional[str]) -> Optional[str]:
        cleaned = sanitize_text(value, max_len=500)
        if not cleaned:
            return None
        if not cleaned.startswith(("http://", "https://")):
            raise ValueError("Website must start with http:// or https://")
        return cleaned

    @field_validator("notes", "product_name")
    @classmethod
    def _short_text(cls, value: Optional[str]) -> Optional[str]:
        return sanitize_text(value, max_len=500)

    @field_validator("product_description")
    @classmethod
    def _long_text(cls, value: Optional[str]) -> Optional[str]:
        return sanitize_text(value, max_len=5000)

    @field_validator("session_id")
    @classmethod
    def _session_id(cls, value: Optional[str]) -> Optional[str]:
        cleaned = sanitize_text(value, max_len=120)
        return cleaned or None


class AuthLoginRequest(StrictBaseModel):
    username: str
    password: str

    @field_validator("username")
    @classmethod
    def _username(cls, value: str) -> str:
        cleaned = sanitize_text(value, max_len=64)
        if not cleaned:
            raise ValueError("Username is required")
        return cleaned

    @field_validator("password")
    @classmethod
    def _password(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Password is required")
        return value


class AuthRegisterRequest(StrictBaseModel):
    username: str
    password: str

    @field_validator("username")
    @classmethod
    def _username(cls, value: str) -> str:
        cleaned = sanitize_text(value, max_len=64)
        if not cleaned:
            raise ValueError("Username is required")
        return cleaned

    @field_validator("password")
    @classmethod
    def _password(cls, value: str) -> str:
        validate_password_strength(value)
        return value


class AuthRefreshRequest(StrictBaseModel):
    refresh_token: Optional[str] = None


class AuthTokenResponse(StrictBaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    refresh_expires_in: int
    role: str


class AuthMeResponse(StrictBaseModel):
    user_id: str
    username: str
    role: str
    is_admin: bool = False


class RiskDetectionRequest(StrictBaseModel):
    deal_ids: Optional[List[str]] = None
    check_all: bool = True
    inactivity_threshold_days: int = Field(default=10, ge=1, le=180)


class ChurnPredictionRequest(StrictBaseModel):
    account_ids: Optional[List[str]] = None
    top_n: int = Field(default=3, ge=1, le=20)


class SendEmailRequest(StrictBaseModel):
    to_email: str
    to_name: Optional[str] = ""
    subject: str
    body_text: str
    body_html: Optional[str] = None

    @field_validator("to_email")
    @classmethod
    def _email(cls, value: str) -> str:
        if not EMAIL_RE.match(value.strip()):
            raise ValueError("Invalid email address")
        return value.strip().lower()

    @field_validator("subject")
    @classmethod
    def _subject(cls, value: str) -> str:
        cleaned = sanitize_text(value, max_len=255)
        if not cleaned:
            raise ValueError("Subject is required")
        return cleaned

    @field_validator("body_text", "body_html")
    @classmethod
    def _body(cls, value: Optional[str]) -> Optional[str]:
        return sanitize_text(value, max_len=10000, allow_empty=True)


class SingleLeadSendRequest(StrictBaseModel):
    lead_id: Optional[str] = None
    lead_name: Optional[str] = ""
    sequence_id: Optional[str] = None
    email: str
    content: str
    subject: Optional[str] = None
    from_email: Optional[str] = None
    from_name: Optional[str] = None

    @field_validator("email")
    @classmethod
    def _recipient_email(cls, value: str) -> str:
        candidate = (value or "").strip().lower()
        if not EMAIL_RE.match(candidate):
            raise ValueError("Invalid email address")
        return candidate

    @field_validator("content")
    @classmethod
    def _content(cls, value: str) -> str:
        cleaned = sanitize_text(value, max_len=20000)
        if not cleaned:
            raise ValueError("content is required")
        return cleaned

    @field_validator("subject")
    @classmethod
    def _subject(cls, value: Optional[str]) -> Optional[str]:
        return sanitize_text(value, max_len=255)

    @field_validator("from_name")
    @classmethod
    def _from_name(cls, value: Optional[str]) -> Optional[str]:
        return sanitize_text(value, max_len=120)

    @field_validator("from_email")
    @classmethod
    def _from_email(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        cleaned = value.strip().lower()
        if not cleaned:
            return None
        if not EMAIL_RE.match(cleaned):
            raise ValueError("Invalid from_email address")
        return cleaned


class RefineEmailRequest(StrictBaseModel):
    lead_id: str
    original_email: str
    prompt: str
    lead_context: Dict[str, Any] = Field(default_factory=dict)
    insights: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("lead_id")
    @classmethod
    def _lead_id(cls, value: str) -> str:
        cleaned = sanitize_text(value, max_len=200)
        if not cleaned:
            raise ValueError("lead_id is required")
        return cleaned

    @field_validator("original_email")
    @classmethod
    def _original_email(cls, value: str) -> str:
        cleaned = sanitize_text(value, max_len=20000)
        if not cleaned:
            raise ValueError("original_email is required")
        return cleaned

    @field_validator("prompt")
    @classmethod
    def _prompt(cls, value: str) -> str:
        cleaned = sanitize_text(value, max_len=2000)
        if not cleaned:
            raise ValueError("prompt is required")
        return cleaned


class ReviewedEmail(StrictBaseModel):
    subject: str
    body: str
    from_email: Optional[str] = None
    from_name: Optional[str] = None


class ReviewedSequence(StrictBaseModel):
    lead_id: Optional[str] = None
    lead_email: str = ""
    email: Optional[str] = None
    lead_name: Optional[str] = ""
    sequence_id: Optional[str] = None
    content: Optional[str] = None
    emails: List[ReviewedEmail] = Field(default_factory=list)

    @model_validator(mode="after")
    def _sync_email_fields(self) -> "ReviewedSequence":
        if not self.lead_email and self.email:
            self.lead_email = self.email
        if not self.email and self.lead_email:
            self.email = self.lead_email
        return self


class SendSequencesRequest(StrictBaseModel):
    sequences: List[ReviewedSequence] = Field(default_factory=list)


class SessionResponse(StrictBaseModel):
    session_id: str
    owner_user_id: str
    task_type: str
    status: str
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    request_id: Optional[str] = None


class ProspectingLeadOutput(Lead):
    why_prioritized: str = ""


class ProspectingOutput(StrictBaseModel):
    leads: List[ProspectingLeadOutput] = Field(default_factory=list)
    company_summary: str = ""
    recommended_approach: str = ""
    icp_fit_score: float = Field(default=0.0, ge=0.0, le=1.0)


class DigitalTwinObjection(StrictBaseModel):
    objection: str
    severity: str
    counter_strategy: str


class DigitalTwinProfileOutput(StrictBaseModel):
    buyer_name: str
    buyer_title: str
    buying_style: str
    primary_motivations: List[str] = Field(default_factory=list)
    top_objections: List[DigitalTwinObjection] = Field(default_factory=list)
    decision_criteria: List[str] = Field(default_factory=list)
    likely_questions: List[str] = Field(default_factory=list)
    emotional_triggers: List[str] = Field(default_factory=list)
    risk_perception: str = "medium"
    estimated_decision_timeline: str = ""
    recommended_tone: str = "consultative"
    opening_hook: str = ""
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)


class ExplainabilityDecision(StrictBaseModel):
    step: int
    agent: str
    decision: str
    why: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    impact: str


class ExplainabilityOutput(StrictBaseModel):
    executive_summary: str
    decision_chain: List[ExplainabilityDecision] = Field(default_factory=list)
    key_insights: List[str] = Field(default_factory=list)
    data_sources_used: List[str] = Field(default_factory=list)
    overall_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    limitations: List[str] = Field(default_factory=list)
    human_review_recommended: bool = False
    human_review_reasons: List[str] = Field(default_factory=list)
    impact_metrics: Dict[str, Any] = Field(default_factory=dict)


class OutreachEntryStatus(str, Enum):
    DRAFT = "draft"
    SENT = "sent"
    OPENED = "opened"
    REPLIED = "replied"
    MEETING_SCHEDULED = "meeting_scheduled"
    CLOSED_WON = "closed_won"
    CLOSED_LOST = "closed_lost"


class OutreachEntry(StrictBaseModel):
    id: str
    user_id: str
    company_name: str
    company_domain: Optional[str] = None
    outreach_type: str = "email"
    message: Optional[str] = None
    status: OutreachEntryStatus
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


class OutreachEntryStatusUpdate(StrictBaseModel):
    status: OutreachEntryStatus


class Customer(StrictBaseModel):
    id: str
    user_id: str
    company_name: str
    company_domain: Optional[str] = None
    contact_name: Optional[str] = None
    contact_email: Optional[str] = None
    notes: Optional[str] = None
    source_entry_id: Optional[str] = None
    source_outreach_status: Optional[OutreachEntryStatus] = None
    marked_as_customer_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)

    @field_validator("company_name")
    @classmethod
    def _company_name(cls, value: str) -> str:
        cleaned = sanitize_text(value, max_len=255)
        if not cleaned:
            raise ValueError("company_name is required")
        return cleaned

    @field_validator("company_domain")
    @classmethod
    def _company_domain(cls, value: Optional[str]) -> Optional[str]:
        return sanitize_text(value, max_len=500)

    @field_validator("contact_name")
    @classmethod
    def _contact_name(cls, value: Optional[str]) -> Optional[str]:
        return sanitize_text(value, max_len=120)

    @field_validator("contact_email")
    @classmethod
    def _contact_email(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        cleaned = value.strip().lower()
        if not cleaned:
            return None
        if not EMAIL_RE.match(cleaned):
            raise ValueError("Invalid contact_email address")
        return cleaned

    @field_validator("notes")
    @classmethod
    def _notes(cls, value: Optional[str]) -> Optional[str]:
        return sanitize_text(value, max_len=5000)

    @field_validator("source_entry_id")
    @classmethod
    def _source_entry_id(cls, value: Optional[str]) -> Optional[str]:
        return sanitize_text(value, max_len=120)


class CustomerCreateRequest(StrictBaseModel):
    company_name: str
    company_domain: Optional[str] = None
    contact_name: Optional[str] = None
    contact_email: Optional[str] = None
    notes: Optional[str] = None
    source_entry_id: Optional[str] = None
    source_outreach_status: Optional[OutreachEntryStatus] = None

    @field_validator("company_name")
    @classmethod
    def _company_name(cls, value: str) -> str:
        cleaned = sanitize_text(value, max_len=255)
        if not cleaned:
            raise ValueError("company_name is required")
        return cleaned

    @field_validator("company_domain")
    @classmethod
    def _company_domain(cls, value: Optional[str]) -> Optional[str]:
        return sanitize_text(value, max_len=500)

    @field_validator("contact_name")
    @classmethod
    def _contact_name(cls, value: Optional[str]) -> Optional[str]:
        return sanitize_text(value, max_len=120)

    @field_validator("contact_email")
    @classmethod
    def _contact_email(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        cleaned = value.strip().lower()
        if not cleaned:
            return None
        if not EMAIL_RE.match(cleaned):
            raise ValueError("Invalid contact_email address")
        return cleaned

    @field_validator("notes")
    @classmethod
    def _notes(cls, value: Optional[str]) -> Optional[str]:
        return sanitize_text(value, max_len=5000)

    @field_validator("source_entry_id")
    @classmethod
    def _source_entry_id(cls, value: Optional[str]) -> Optional[str]:
        return sanitize_text(value, max_len=120)


class CustomerCreateFromEntryRequest(StrictBaseModel):
    contact_name: Optional[str] = None
    contact_email: Optional[str] = None
    notes: Optional[str] = None

    @field_validator("contact_name")
    @classmethod
    def _contact_name(cls, value: Optional[str]) -> Optional[str]:
        return sanitize_text(value, max_len=120)

    @field_validator("contact_email")
    @classmethod
    def _contact_email(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        cleaned = value.strip().lower()
        if not cleaned:
            return None
        if not EMAIL_RE.match(cleaned):
            raise ValueError("Invalid contact_email address")
        return cleaned

    @field_validator("notes")
    @classmethod
    def _notes(cls, value: Optional[str]) -> Optional[str]:
        return sanitize_text(value, max_len=5000)
