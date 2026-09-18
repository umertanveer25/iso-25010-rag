"""
ISO/IEC 25010 Knowledge Base for Software Requirements Classification (FNFC 14-Class Scheme).
Maps each class (Functional + 13 Non-Functional types) to official ISO/IEC 25010 definitions,
sub-characteristics, and domain indicator keywords.
"""

ISO_25010_KNOWLEDGE_BASE = {
    "F": {
        "name": "Functional Requirements",
        "iso_category": "Functional Suitability",
        "definition": "Functions that meet stated and implied needs when used under specified conditions. Covers functional completeness, correctness, and appropriateness.",
        "sub_characteristics": ["Functional Completeness", "Functional Correctness", "Functional Appropriateness"],
        "keywords": [
            "shall", "allow", "enable", "calculate", "provide", "display", "process", 
            "store", "retrieve", "send", "generate", "create", "delete", "update", "input", "output", "system must"
        ]
    },
    "A": {
        "name": "Availability",
        "iso_category": "Reliability - Availability",
        "definition": "Degree to which a system, product or component is operational and accessible when required for use. Operational uptime, accessibility, MTBF.",
        "sub_characteristics": ["Operational Availability", "Uptime", "Accessibility", "System Readiness"],
        "keywords": [
            "availability", "uptime", "available 24/7", "operational hours", "accessible", 
            "continuous operation", "99.9%", "downtime limit", "readiness", "always online"
        ]
    },
    "AU": {
        "name": "Autonomy",
        "iso_category": "Functional Appropriateness - Autonomous Behavior",
        "definition": "Degree to which a system performs operations independently without human intervention or external manual control.",
        "sub_characteristics": ["Self-Regulation", "Automated Execution", "Independent Operation", "Unattended Behavior"],
        "keywords": [
            "automatically", "autonomous", "unattended", "without human intervention", 
            "self-acting", "self-triggering", "independent action", "auto-detect", "auto-recovery"
        ]
    },
    "FT": {
        "name": "Fault Tolerance",
        "iso_category": "Reliability - Fault Tolerance",
        "definition": "Degree to which a system operates as intended despite the presence of hardware or software faults or environmental interruptions.",
        "sub_characteristics": ["Failure Handling", "Graceful Degradation", "Fault Recovery", "Error Resilience"],
        "keywords": [
            "fault tolerance", "failover", "graceful degradation", "hardware failure", 
            "power loss recovery", "exception handling", "redundancy", "resilient to errors", "backup node"
        ]
    },
    "LF": {
        "name": "Look and Feel",
        "iso_category": "Usability - User Interface Aesthetics",
        "definition": "Aesthetic design, visual appearance, style, layout, color scheme, and visual brand identity of the user interface.",
        "sub_characteristics": ["User Interface Aesthetics", "Visual Layout", "Styling", "Branding"],
        "keywords": [
            "look and feel", "visual theme", "color scheme", "font style", "logo placement", 
            "gui layout", "aesthetic", "branding", "ui appearance", "screen design"
        ]
    },
    "LL": {
        "name": "Legal & Licensing",
        "iso_category": "Compliance - Legal & Regulatory",
        "definition": "Adherence to laws, regulations, copyright, privacy acts (GDPR, HIPAA), license terms, and industry standards.",
        "sub_characteristics": ["Regulatory Compliance", "Intellectual Property", "Privacy Laws", "Licensing Terms"],
        "keywords": [
            "legal", "compliance", "copyright", "license", "gdpr", "hipaa", "regulation", 
            "privacy act", "terms of service", "statutory", "intellectual property"
        ]
    },
    "M": {
        "name": "Maintainability",
        "iso_category": "Maintainability",
        "definition": "Degree of effectiveness and efficiency with which a product or system can be modified, corrected, improved or adapted to changes.",
        "sub_characteristics": ["Modifiability", "Modularity", "Reusability", "Analysability", "Testability"],
        "keywords": [
            "maintainability", "modular code", "ease of maintenance", "documentation", 
            "extensible", "refactoring", "code clean", "configurable", "testability"
        ]
    },
    "O": {
        "name": "Inter-Operability (Operational)",
        "iso_category": "Compatibility - Interoperability",
        "definition": "Degree to which two or more systems, products or components can exchange information and use the information that has been exchanged.",
        "sub_characteristics": ["Data Exchange", "API Compatibility", "External Integration", "Protocol Support"],
        "keywords": [
            "interoperability", "inter-operable", "api integration", "external system", 
            "data exchange", "third-party service", "protocol compatibility", "export format", "import format"
        ]
    },
    "P": {
        "name": "Portability",
        "iso_category": "Portability",
        "definition": "Degree of effectiveness and efficiency with which a system, product or component can be transferred from one hardware, software or operational environment to another.",
        "sub_characteristics": ["Adaptability", "Installability", "Replaceability", "Cross-Platform Support"],
        "keywords": [
            "portability", "cross-platform", "windows/mac/linux", "mobile and desktop", 
            "multi-browser", "os compatible", "installation package", "docker container", "portable"
        ]
    },
    "PE": {
        "name": "Performance",
        "iso_category": "Performance Efficiency",
        "definition": "Performance relative to the amount of resources used under stated conditions. Encompasses response time, throughput, latency, and resource utilization.",
        "sub_characteristics": ["Time Behaviour", "Resource Utilization", "Capacity", "Throughput"],
        "keywords": [
            "performance", "response time", "latency", "seconds", "milliseconds", "throughput", 
            "transactions per second", "cpu usage", "memory consumption", "fast response"
        ]
    },
    "R": {
        "name": "Reliability",
        "iso_category": "Reliability - Maturity",
        "definition": "Degree to which a system performs specified functions under specified conditions for a specified period without system crashes or data corruption.",
        "sub_characteristics": ["Maturity", "Consistency", "Mean Time Between Failures (MTBF)", "Data Integrity"],
        "keywords": [
            "reliability", "reliable", "crash-free", "data integrity", "mtbf", 
            "consistent operation", "accurate results", "no data loss", "stable"
        ]
    },
    "SC": {
        "name": "Scalability",
        "iso_category": "Performance Efficiency - Capacity / Scalability",
        "definition": "Degree to which a system can expand its workload capacity, handle increased traffic or data volume by scaling resources.",
        "sub_characteristics": ["Capacity Scaling", "Horizontal Scaling", "Vertical Scaling", "Load Growth"],
        "keywords": [
            "scalability", "scalable", "concurrent users", "workload capacity", "scale up", 
            "scale out", "high volume data", "increased load", "10000 users"
        ]
    },
    "SE": {
        "name": "Security",
        "iso_category": "Security",
        "definition": "Degree to which a product or system protects information and data so that persons or other products or systems have the degree of data access appropriate to their types and levels of authorization.",
        "sub_characteristics": ["Confidentiality", "Integrity", "Non-repudiation", "Accountability", "Authenticity"],
        "keywords": [
            "security", "authentication", "authorization", "encryption", "password", 
            "access control", "ssl/tls", "role-based", "confidentiality", "cyberattack", "secure"
        ]
    },
    "US": {
        "name": "Usability",
        "iso_category": "Usability",
        "definition": "Degree to which a product or system can be used by specified users to achieve specified goals with effectiveness, efficiency and satisfaction.",
        "sub_characteristics": ["Appropriateness Recognizability", "Learnability", "Operability", "User Error Protection"],
        "keywords": [
            "usability", "user-friendly", "easy to use", "intuitive", "learning curve", 
            "help documentation", "accessibility for disabled", "user satisfaction", "ui feedback"
        ]
    }
}
