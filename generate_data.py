#!/usr/bin/env python3
"""
Generate synthetic raw data files for the PII curation pipeline.

Outputs
-------
data/raw/slack_messages.json    200 messages, 30 % with PII
data/raw/project_records.csv   300 rows,     40 % with PII in notes
data/raw/code_comments.txt     150 comment blocks, 25 % with PII
"""

import csv
import json
import random
from datetime import datetime, timedelta
from pathlib import Path

from faker import Faker

fake = Faker()
Faker.seed(42)
random.seed(42)

BASE_DIR = Path(__file__).parent
RAW_DIR = BASE_DIR / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
#  Slack messages
# ============================================================

CHANNELS = [
    "general", "engineering", "design", "product", "ops",
    "data-team", "backend", "frontend", "devops", "hr-announcements",
]

_NON_PII_SLACK = [
    "Has anyone reviewed the latest deployment plan for {project}?",
    "The {project} sprint is wrapping up — time to close tickets.",
    "Heads-up: maintenance on the {system} cluster is scheduled for tonight.",
    "Can someone push the {project} docs to Confluence?",
    "Stand-up in 10 minutes — join the {channel} call.",
    "Just merged the PR for the {feature} module. Needs QA sign-off.",
    "Dashboard metrics look solid after the {project} rollout.",
    "Who owns the {service} service? We need an architecture review.",
    "Bumping thread — any update on the {project} launch timeline?",
    "The CI/CD pipeline is green. Ready to tag release v{version}.",
    "Let's sync about the {project} retro after the demo.",
    "Reminder: {project} QA sign-off is due by end of day.",
    "Infrastructure team: the {service} pod is throwing 502s in staging.",
    "Docs for {feature} updated — link posted in the channel description.",
    "Design review scheduled for {project} mockups at 3 PM in Zoom.",
    "Deploying {project} to us-east-1. ETA 20 minutes.",
    "The {system} migration is complete. All health checks green.",
    "Opening a ticket for the {feature} regression found in QA.",
    "Sprint velocity for {project} is up 12 % this cycle.",
    "New runbook for {service} failover added to the wiki.",
]

_PII_SLACK = [
    "Hi team, I'm {name}. Just joined — excited to work with everyone!",
    "Reach out to {name} at {email} for vendor onboarding documents.",
    "Call me at {phone} if you need an urgent update on the deploy.",
    "For billing questions contact {name}, {org} account manager.",
    "New hire spotlight: {name} starts Monday on the {channel} team.",
    "FYI — {name}'s personal number is {phone} for the on-call rotation.",
    "The NDA was sent to {name} at {email}. Waiting on a signature.",
    "Sensitive — do NOT forward: SSN {ssn} for contractor {name}.",
    "Expense report from {name}: card ending {card_last4}, total $412.00.",
    "Emergency contact for {name}: {phone}.",
    "Wire transfer authorised by {name} ({email}) — ref #{ref}.",
    "{name} from {org} confirmed the deal. Please loop in {email}.",
    "My dev subscription card: {cc}. Please charge and delete this message.",
    "Contractor W-9 for {name}, SSN: {ssn}, address: {address}.",
    "Please update the vendor record — {name} moved to {address}.",
    "On-call this week: {name} ({phone}). Backup: {name2} ({email}).",
    "Resending login for {name}: temp password {phone} — change immediately.",
    "The {org} rep is {name}, reachable at {email} or {phone}.",
    "Data access request approved for {name} ({email}) — valid 30 days.",
    "HR note: {name}'s SSN on file is {ssn}. Please verify and shred this.",
]


def _pii_vars():
    name = fake.name()
    name2 = fake.name()
    return dict(
        name=name,
        name2=name2,
        email=fake.email(),
        phone=fake.phone_number(),
        org=fake.company(),
        channel=random.choice(CHANNELS),
        ssn=f"{random.randint(100,999)}-{random.randint(10,99)}-{random.randint(1000,9999)}",
        cc=(
            f"{random.randint(4000,4999)}-{random.randint(1000,9999)}"
            f"-{random.randint(1000,9999)}-{random.randint(1000,9999)}"
        ),
        card_last4=str(random.randint(1000, 9999)),
        ref=fake.bothify(text="??-######").upper(),
        address=fake.address().replace("\n", ", "),
    )


def _non_pii_vars():
    return dict(
        project=fake.bs().split()[0].capitalize() + "Flow",
        system=random.choice(["Kafka", "Redis", "Postgres", "Elasticsearch", "MongoDB"]),
        channel=random.choice(CHANNELS),
        feature=fake.bs().split()[-1].capitalize(),
        service=fake.bs().split()[0].capitalize() + "Service",
        version=f"{random.randint(1,3)}.{random.randint(0,9)}.{random.randint(0,9)}",
    )


def generate_slack_messages(n: int = 200, pii_fraction: float = 0.30) -> list[dict]:
    pii_count = round(n * pii_fraction)          # exactly 60
    pii_indices = set(random.sample(range(n), pii_count))
    base_ts = datetime(2024, 1, 15, 9, 0, 0)
    messages = []

    for i in range(n):
        ts = base_ts + timedelta(minutes=i * random.randint(4, 18))
        if i in pii_indices:
            template = random.choice(_PII_SLACK)
            text = template.format(**_pii_vars())
        else:
            template = random.choice(_NON_PII_SLACK)
            text = template.format(**_non_pii_vars())

        messages.append(
            {
                "user_id": f"U{random.randint(100_000, 999_999)}",
                "username": fake.user_name(),
                "timestamp": ts.isoformat() + "Z",
                "channel": random.choice(CHANNELS),
                "text": text,
            }
        )
    return messages


# ============================================================
#  Project records
# ============================================================

_STATUSES = ["Open", "In Progress", "Review", "Done", "Blocked", "Cancelled"]
_PROJECTS = [
    "DataSync", "CloudMigrate", "APIGateway", "UserPortal", "ReportingEngine",
    "MLPipeline", "AuthService", "BillingSystem", "NotificationHub", "SearchIndex",
    "AnalyticsDash", "MobileSDK", "InfraAutomation", "ComplianceAudit", "DevPortal",
]
_DATE_FMTS = ["%Y-%m-%d", "%m/%d/%Y", "%d-%b-%Y", "%Y/%m/%d", "%B %d, %Y"]

_PII_NOTES = [
    "Contact {name} at {email} for detailed requirements.",
    "Approved by {name} — direct line: {phone}.",
    "Vendor representative: {name}, {email}.",
    "Escalate to {name} if blocked; direct line {phone}.",
    "Reviewed by {name} from {org}.",
    "CC {email} on all future status updates.",
    "Emergency contact for this project: {name} at {phone}.",
    "{name} will handle all stakeholder communication.",
    "Data shared with {name} at {org} ({email}).",
    "Written sign-off from {name} required before release.",
    "Stakeholder: {name} ({org}) — notified via {email}.",
    "Budget holder: {name}, confirm at {phone} before spend.",
    "Point of contact changed to {name} ({email}) as of last week.",
    "Legal review requested by {name} — response sent to {email}.",
    "On-site contact at {org}: {name}, {phone}.",
]

_NON_PII_NOTES = [
    "Waiting on infrastructure provisioning from the DevOps team.",
    "Blocked by upstream API contract changes — see open ticket.",
    "Needs design sign-off before development work begins.",
    "All unit tests passing; currently awaiting code review.",
    "Deployment scheduled for the next sprint cycle.",
    "Dependencies updated; migration validated in staging.",
    "Feature gated behind {flag} — enable when sign-off received.",
    "Full technical spec available on the Confluence page.",
    "Requires legal review of the updated data retention policy.",
    "Performance benchmarks met in load testing; ready for prod.",
    "Automated regression tests added for all identified edge cases.",
    "Rollback plan documented in the runbook.",
    "Architecture decision record (ADR) filed; awaiting approval.",
    "",   # intentionally empty note
    "",
    "",
]


def _pii_note() -> str:
    t = random.choice(_PII_NOTES)
    return t.format(
        name=fake.name(),
        email=fake.email(),
        phone=fake.phone_number(),
        org=fake.company(),
    )


def _non_pii_note() -> str:
    t = random.choice(_NON_PII_NOTES)
    return t.format(flag=fake.bothify(text="FEAT_????_####").upper())


def _rand_date() -> str:
    d = fake.date_between(start_date="-1y", end_date="+1y")
    return d.strftime(random.choice(_DATE_FMTS))


def generate_project_records(
    n: int = 300,
    pii_fraction: float = 0.40,
    dupe_count: int = 6,
    missing_id_count: int = 9,
) -> list[dict]:
    pii_count = round(n * pii_fraction)          # exactly 120
    pii_indices = set(random.sample(range(n), pii_count))

    # Build a list of record IDs, then inject anomalies
    record_ids: list[str] = [str(i) for i in range(1001, 1001 + n)]

    for _ in range(dupe_count):
        idx = random.randint(0, n - 2)
        record_ids[idx + 1] = record_ids[idx]     # consecutive duplicate

    for idx in random.sample(range(n), missing_id_count):
        record_ids[idx] = ""                       # missing ID

    records = []
    for i in range(n):
        # ~7 % of rows have missing assignee info
        if random.random() < 0.07:
            assignee_name = ""
            assignee_email = ""
        else:
            assignee_name = fake.name()
            assignee_email = fake.email() if random.random() > 0.05 else ""

        records.append(
            {
                "record_id": record_ids[i],
                "assignee_name": assignee_name,
                "assignee_email": assignee_email,
                "project_name": random.choice(_PROJECTS),
                "status": random.choice(_STATUSES),
                "due_date": _rand_date(),
                "notes": _pii_note() if i in pii_indices else _non_pii_note(),
            }
        )
    return records


# ============================================================
#  Code comments
# ============================================================

_NON_PII_COMMENTS = [
    "# {func}: validates the input schema before writing to the data lake.",
    "# TODO: refactor this loop — O(n^2) complexity will hurt at scale.",
    "# NOTE: this retry logic mirrors the pattern in {module}.py.",
    "# FIXME: edge case when {var} is None — needs a null guard.",
    "# Batch size capped at {n} rows to stay under the API rate limit.",
    "# Cache TTL set to {ttl}s; tune in config if eviction is too aggressive.",
    "# {algo} chosen for stability over raw throughput in this context.",
    "# Step {step}: normalise column names before merge.",
    "# Deprecated — use {replacement}() for all new call sites.",
    "# This filter drops rows where {col} is missing or malformed.",
    "# Partition key must be set before the transaction begins.",
    "# Max retries: {n}. Exponential back-off starting at 200 ms.",
    "# See ADR-{step}04 for the rationale behind this design choice.",
    "# Output written to {module}_results.parquet via PyArrow.",
    "# Idempotent: safe to re-run; duplicate rows are de-duped on write.",
    "# WARNING: do not change the sort order — downstream jobs depend on it.",
    "# Schema version {step}.0 — bump on any breaking column change.",
    "# Async task; check the {module}_queue for completion status.",
    "# Memory budget: {n} MB. Increase only after profiling.",
    "# Unit-tested in tests/test_{module}.py.",
]

_PII_COMMENTS = [
    "# TODO({name}): clean up this workaround once the infra ticket is resolved.",
    "# Written by {name} ({email}) — contact before modifying this module.",
    "# DEBUG — temp local-dev creds: user={name_lower} pwd={phone} CHANGE ME",
    "# FIXME({name}): breaks when VPN is disconnected — see Slack thread.",
    "# Originally authored by {name}. Last refactored {date}.",
    "# Contact {name} at {email} before altering the schema.",
    "# On-call escalation path: {name}, {phone}.",
    "# Test account created by {name} during UAT — do not delete or expire.",
    "# TODO: remove the hardcoded API key below — added by {name} for a demo.",
    "# Reviewed and approved by {name} ({email}) on {date}.",
    "# Paired with {name} ({email}) to debug the race condition.",
    "# Initial version sent to {name} at {email} for external audit.",
    "# Staging DB password set by {name} — rotate before prod release.",
    "# Report any anomalies directly to {name} ({phone}).",
    "# Authored jointly by {name} and {name2}; primary contact: {email}.",
]


def _pii_comment_vars() -> dict:
    name = fake.name()
    name2 = fake.name()
    return dict(
        name=name,
        name2=name2,
        name_lower=name.split()[0].lower(),
        email=fake.email(),
        phone=fake.phone_number(),
        date=fake.date_this_decade().strftime("%Y-%m-%d"),
    )


def _non_pii_comment_vars() -> dict:
    return dict(
        func=fake.bs().split()[-1].lower() + "_handler",
        module=fake.bs().split()[0].lower(),
        var=random.choice(["payload", "record", "batch", "config", "session", "cursor"]),
        n=random.choice([100, 500, 1_000, 5_000, 10_000]),
        ttl=random.choice([60, 300, 900, 3_600]),
        algo=random.choice(["quicksort", "merge-sort", "heapsort", "radix", "topological"]),
        step=random.randint(1, 8),
        replacement=fake.bs().split()[-1].lower() + "_v2",
        col=random.choice(["record_id", "timestamp", "email", "status", "user_id"]),
    )


def generate_code_comments(n: int = 150, pii_fraction: float = 0.25) -> str:
    pii_count = round(n * pii_fraction)          # exactly 37–38
    pii_indices = set(random.sample(range(n), pii_count))
    lines = []

    for i in range(n):
        if i in pii_indices:
            t = random.choice(_PII_COMMENTS)
            lines.append(t.format(**_pii_comment_vars()))
        else:
            t = random.choice(_NON_PII_COMMENTS)
            lines.append(t.format(**_non_pii_comment_vars()))
    return "\n".join(lines)


# ============================================================
#  Entry point
# ============================================================

def main() -> None:
    # slack_messages.json
    messages = generate_slack_messages()
    with open(RAW_DIR / "slack_messages.json", "w", encoding="utf-8") as f:
        json.dump(messages, f, indent=2, ensure_ascii=False)
    print(f"[OK] slack_messages.json  — {len(messages)} messages "
          f"({sum(1 for m in messages if any(kw in m['text'] for kw in ['@', 'SSN', 'call me', 'card']))}"
          f"+ detectable PII)")

    # project_records.csv
    records = generate_project_records()
    fieldnames = [
        "record_id", "assignee_name", "assignee_email",
        "project_name", "status", "due_date", "notes",
    ]
    with open(RAW_DIR / "project_records.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"[OK] project_records.csv  — {len(records)} rows")

    # code_comments.txt
    comments = generate_code_comments()
    with open(RAW_DIR / "code_comments.txt", "w", encoding="utf-8") as f:
        f.write(comments)
    block_count = comments.count("\n") + 1
    print(f"[OK] code_comments.txt    — {block_count} comment blocks")


if __name__ == "__main__":
    main()
