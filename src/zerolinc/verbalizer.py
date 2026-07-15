"""NIST SP 800-61r3 category definitions and zero-shot prompt configurations.

The 12 categories mirror the taxonomy used by the reference LLM studies
(SecLINC / FrameworkPE): English names and descriptions follow their
zero-shot prompt; Portuguese names and descriptions follow the taxonomy
table published with the SBSeg study. Keyword lists are the exact "search
terms" the reference prompts embed per category.

A *prompt configuration* is the zero-shot analogue of a prompt-engineering
technique: it fixes (a) the hypothesis template handed to the NLI model and
(b) how each category is verbalized into a candidate label.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Category:
    code: str
    name_en: str
    name_pt: str
    desc_en: str
    desc_pt: str
    keywords: tuple[str, ...]


CATEGORIES: tuple[Category, ...] = (
    Category(
        "CAT1", "account compromise", "comprometimento de conta",
        "unauthorized access to user or administrator accounts",
        "acesso não autorizado a contas de usuários ou administradores",
        ("phishing", "brute force", "unauthorized access", "compromised password",
         "credential theft", "account compromise", "token", "oauth", "ssh", "suspicious login"),
    ),
    Category(
        "CAT2", "malware", "malware",
        "infection by malicious code",
        "infecção por código malicioso que compromete dispositivos ou dados",
        ("malware", "ransomware", "trojan", "virus", "spyware", "rootkit",
         "infection", "malicious code"),
    ),
    Category(
        "CAT3", "denial of service attack", "ataque de negação de serviço",
        "making systems or networks unavailable",
        "tornar sistemas ou redes indisponíveis",
        ("ddos", "dos", "denial of service", "flood", "syn flood", "udp flood",
         "botnet", "api outage", "site down"),
    ),
    Category(
        "CAT4", "data leak", "exfiltração ou vazamento de dados",
        "unauthorized disclosure of sensitive data",
        "acesso, cópia ou divulgação não autorizada de dados sensíveis",
        ("data leak", "exposed data", "leaked credentials", "sensitive information",
         "data exfiltration", "unauthorized disclosure"),
    ),
    Category(
        "CAT5", "vulnerability exploitation", "exploração de vulnerabilidade",
        "use of known or unknown technical flaws to compromise assets",
        "uso de falhas conhecidas ou desconhecidas para comprometer ativos",
        ("exploit", "vulnerability", "cve", "remote execution", "sql injection",
         "injection", "rce", "security flaw"),
    ),
    Category(
        "CAT6", "insider abuse", "abuso interno",
        "malicious or negligent actions by internal users",
        "ações intencionais ou negligentes de usuários internos",
        ("insider", "internal abuse", "employee", "internal leak", "sabotage",
         "intentional action", "staff"),
    ),
    Category(
        "CAT7", "social engineering", "engenharia social",
        "deceiving people to obtain access or information",
        "engano de pessoas para obter acesso ou informações",
        ("social engineering", "phishing", "vishing", "fraud", "deception",
         "spoofing", "manipulation", "scam", "ceo fraud"),
    ),
    Category(
        "CAT8", "physical incident", "incidente físico ou de infraestrutura",
        "physical breach impacting computational assets",
        "violação física que impacta ativos computacionais",
        ("physical access", "equipment theft", "burglary", "unauthorized entry",
         "broken door", "physical breach"),
    ),
    Category(
        "CAT9", "unauthorized modification", "alteração não autorizada",
        "unauthorized changes to systems, data, or configurations",
        "modificação não autorizada em sistemas, dados ou configurações",
        ("modification", "defacement", "unauthorized change", "erased",
         "altered record", "tampering"),
    ),
    Category(
        "CAT10", "misuse of resources", "uso indevido de recursos",
        "unauthorized use of systems for other purposes",
        "uso não autorizado de sistemas para outros fins",
        ("misuse", "resource abuse", "crypto mining", "compromised server",
         "malware hosting", "unauthorized use"),
    ),
    Category(
        "CAT11", "third-party incident", "problema de fornecedor ou terceiro",
        "incident originating from a third-party security failure",
        "incidente originado por falha de segurança de terceiros",
        ("third party", "supplier", "partner", "vendor", "supply chain",
         "external breach", "saas issue"),
    ),
    Category(
        "CAT12", "intrusion attempt", "tentativa de intrusão",
        "hostile attempts to break in, not yet confirmed as successful",
        "tentativas hostis de invasão ainda não confirmadas como bem-sucedidas",
        ("intrusion attempt", "scan", "reconnaissance", "probing", "port scan",
         "blocked exploit", "failed attempt"),
    ),
)

CODES: tuple[str, ...] = tuple(c.code for c in CATEGORIES)

# Event-style hypotheses: each category verbalized as a declarative statement of
# the incident event, the genuine NLI formulation (the model judges whether the
# report entails the event, not whether it is "about a topic").
EVENTS_EN: dict[str, str] = {
    "CAT1": "An attacker gained unauthorized access to a user or administrator account.",
    "CAT2": "A machine was infected by malicious code such as ransomware or a trojan.",
    "CAT3": "A denial-of-service attack made systems or networks unavailable.",
    "CAT4": "Sensitive data was disclosed, copied, or exfiltrated without authorization.",
    "CAT5": "A technical vulnerability or insecurely exposed service was used to compromise assets.",
    "CAT6": "An internal user intentionally or negligently abused their access.",
    "CAT7": "People were deceived into giving away access or information.",
    "CAT8": "A physical breach or physical access impacted computational assets.",
    "CAT9": "Systems, records, or configurations were modified without authorization.",
    "CAT10": "Systems were used without authorization for other purposes, such as "
             "cryptocurrency mining or sending spam.",
    "CAT11": "The incident originated from a security failure at a third party or supplier.",
    "CAT12": "A hostile intrusion attempt such as scanning or probing was observed, "
             "not confirmed as successful.",
}

# Verbatim example lists from the reference zero-shot LLM prompt, per category.
EXAMPLES_EN: dict[str, str] = {
    "CAT1": "credential phishing, SSH brute force, OAuth token theft",
    "CAT2": "ransomware, Trojan horse, macro virus",
    "CAT3": "volumetric DoS or DDoS (UDP flood, SYN flood, HTTP flood), attacks on APIs "
            "or websites, Mirai botnet",
    "CAT4": "database theft, leaked credentials",
    "CAT5": "exploitation of CVE, RCE, SQL injection, or insecure service exposure "
            "(NTP monlist, DNS ANY, open Memcached)",
    "CAT6": "copying confidential data, sabotage, misuse of access",
    "CAT7": "phishing, vishing, CEO fraud, pretexting",
    "CAT8": "equipment theft, data center break-in",
    "CAT9": "website defacement, alteration of records or logs",
    "CAT10": "cryptocurrency mining, spam campaigns, malware hosting",
    "CAT11": "SaaS breach, supply-chain compromise",
    "CAT12": "network scans, brute force attempts, blocked exploit attempts",
}


@dataclass(frozen=True)
class PromptConfig:
    """One zero-shot technique: hypothesis template + label verbalization."""

    name: str
    template: str
    labels: dict[str, str]  # candidate label text -> category code


def _cfg(name: str, template: str, verbalize) -> PromptConfig:
    return PromptConfig(name, template, {verbalize(c): c.code for c in CATEGORIES})


PROMPT_CONFIGS: dict[str, PromptConfig] = {
    cfg.name: cfg
    for cfg in (
        _cfg("en-name", "This text is about {}.",
             lambda c: c.name_en),
        _cfg("en-desc", "This text is about {}.",
             lambda c: f"{c.name_en}: {c.desc_en}"),
        _cfg("en-desc-domain", "This security incident report describes {}.",
             lambda c: f"{c.name_en}, that is, {c.desc_en}"),
        _cfg("en-desc-kw", "This security incident report describes {}.",
             lambda c: f"{c.name_en}: {c.desc_en} (related terms: {', '.join(c.keywords)})"),
        _cfg("en-desc-ex", "This security incident report describes {}.",
             lambda c: f"{c.name_en}: {c.desc_en}, for example {EXAMPLES_EN[c.code]}"),
        _cfg("en-event", "{}",
             lambda c: EVENTS_EN[c.code]),
        _cfg("pt-name", "Este texto é sobre {}.",
             lambda c: c.name_pt),
        _cfg("pt-desc", "Este relato de incidente de segurança descreve {}.",
             lambda c: f"{c.name_pt}, ou seja, {c.desc_pt}"),
    )
}
