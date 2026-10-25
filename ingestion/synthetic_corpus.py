"""High-fidelity synthetic technical documentation generator for 5,000+ enterprise documents."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple
from core.models import Document, DocumentMetadata, EvaluationSample

# Core engineering topics & templates
DOMAINS = {
    "distributed_systems": {
        "topics": [
            "Raft Consensus Protocol Heartbeat and Election Timeout Tuning",
            "Multi-Raft Log Replication and Group Partitioning Strategy",
            "Consistent Hashing and Virtual Node Distribution in Sharded Clusters",
            "Vector Clock Implementation for Concurrent Conflict Resolution",
            "Gossip Protocol Dissemination and Anti-Entropy Mechanics",
            "Distributed Lock Manager Using Lease Grants and Fencing Tokens",
            "Two-Phase Commit vs Saga Pattern for Long-Running Distributed Transactions",
            "Paxos State Machine Replication and Learner Catch-Up Optimization",
            "Split-Brain Recovery and Quorum Loss Resolution Runbook",
            "Geo-Replicated Active-Active Conflict-Free Replicated Data Types (CRDTs)",
        ],
        "tags": ["distributed_systems", "consensus", "replication", "high-availability", "sharding"],
    },
    "database_internals": {
        "topics": [
            "Log-Structured Merge-Tree (LSM) Compaction Algorithms: Leveled vs Size-Tiered",
            "Write-Ahead Log (WAL) Disk Synchronization and Group Commit Latency",
            "Multi-Version Concurrency Control (MVCC) Garbage Collection and Tuple Bloat",
            "B+ Tree Leaf Page Splits and Concurrency Control via Latch Crabbing",
            "Cost-Based Query Optimizer Statistics Generation and Plan Invalidation",
            "Buffer Pool Replacement Policies: 2Q vs LRU-K Cache Management",
            "PostgreSQL Vacuum Optimization for High-Write Transaction Processing",
            "Distributed Query Pushdown and Partition Pruning in Columnar Engines",
            "Database Connection Pooling Starvation and Max Lifetime Configuration",
            "Change Data Capture (CDC) Debezium Streaming Pipeline Architecture",
        ],
        "tags": ["database", "storage_engine", "indexing", "mvcc", "query_optimization"],
    },
    "microservices": {
        "topics": [
            "API Gateway Rate Limiting via Distributed Redis Sliding Window",
            "Envoy Service Mesh mTLS Certificate Rotation and Handshake Latency",
            "Circuit Breaker State Transitions and Fallback Strategy Implementation",
            "Distributed Tracing Context Propagation with W3C TraceContext and OpenTelemetry",
            "Event-Driven Outbox Pattern and Transactional Messaging De-duplication",
            "Backend-for-Frontend (BFF) Pattern GraphQL Federation Gateway",
            "gRPC HTTP/2 Stream Multiplexing and Flow Control Starvation",
            "Idempotency Key Verification and Distributed Token Store Configuration",
            "Microservice Zero-Downtime Blue-Green and Canary Deployment Orchestration",
            "Asynchronous Event Sourcing and Snapshot Serialization Benchmarks",
        ],
        "tags": ["microservices", "service_mesh", "resilience", "grpc", "tracing"],
    },
    "cloud_infrastructure": {
        "topics": [
            "Kubernetes Etcd Quorum Disaster Recovery and Snapshot Restoration Runbook",
            "Cilium eBPF CNI Direct Routing vs Geneve Encapsulation Throughput",
            "Horizontal Pod Autoscaler (HPA) Custom Metric Throttling with KEDA",
            "Container Network Interface (CNI) IPAM Exhaustion and Subnet Migration",
            "Kubernetes Node OOMKilled Priority Eviction and QoS Class Guarantees",
            "Calico NetworkPolicy Ingress Isolation and Micro-segmentation Audit",
            "CoreDNS Query Saturation and NodeLocal DNSCache Deployment",
            "Custom Resource Definition (CRD) Webhook Validation and Latency Budgets",
            "Kubelet Garbage Collection of Dead Containers and Unused Images",
            "StatefulSet Persistent Volume Claim (PVC) Expansion Without Downtime",
        ],
        "tags": ["kubernetes", "infrastructure", "networking", "ebpf", "containers"],
    },
    "security": {
        "topics": [
            "Zero Trust Architecture SPIFFE/SPIRE Workload Identity Issuance",
            "OAuth2 and OIDC Token Exchange for Cross-Boundary Microservice Calls",
            "HashiCorp Vault Dynamic Secret Lease Renewal and PKI Engine Tuning",
            "Role-Based Access Control (RBAC) Least Privilege Audit and Policy Linting",
            "Mutual TLS (mTLS) Cipher Suite Selection and Session Resumption Security",
            "KMS Envelope Encryption Key Rotation Protocol and Cache Coherency",
            "OWASP Top 10 API Security Gateways Defense and JWT Signing Validation",
            "Secrets Management in Kubernetes: SealedSecrets vs External Secrets Operator",
            "Supply Chain Security Software Bill of Materials (SBOM) Verification via Cosign",
            "Incident Response: Compromised API Token Revocation and Forensics Protocol",
        ],
        "tags": ["security", "zero_trust", "oauth2", "encryption", "vault"],
    },
    "observability": {
        "topics": [
            "Prometheus TSDB Head Block Memory Compaction and WAL Retention Strategy",
            "Alertmanager High-Availability Mesh Gossip Clustering and Deduping",
            "Grafana Loki Chunk Flush Frequency and Compressed Label Indexing",
            "OpenTelemetry Collector Memory Limiter and Batch Processor Optimization",
            "Distributed Continuous Profiling with Parca and CPU Flamegraph Analysis",
            "SRE Service Level Objective (SLO) Multi-Window Multi-Burn-Rate Alerts",
            "Log Ingestion Rate Limiting and Dynamic Cardinality Reduction Filters",
            "Synthetic Canaries and Real User Monitoring (RUM) Latency Percentiles",
            "Chaos Engineering: Automated Chaos Mesh Network Latency Injection Test",
            "Postmortem Analysis: Inter-Region Transit Gateway Blackhole Outage",
        ],
        "tags": ["observability", "prometheus", "sre", "metrics", "alerts"],
    },
    "streaming": {
        "topics": [
            "Apache Kafka Consumer Group Partition Rebalance Protocol: Cooperative Sticky",
            "Kafka Broker Page Cache Saturation and Zero-Copy Sendfile Diagnostics",
            "Exactly-Once Semantics (EOS) Two-Phase Commit in Kafka Streams",
            "Confluent Schema Registry Backward Compatibility and Subject Evolution",
            "Apache Flink RocksDB State Backend Memory Management and Incremental Checkpoints",
            "Kafka Connect Distributed Worker Rebalance and Connector Task Failover",
            "Message Deduplication via Bloom Filter and RocksDB Persistent Cache",
            "High-Throughput Protobuf Serialization vs JSON Benchmarks in Kafka",
            "Kafka MirrorMaker 2 Active-Passive Disaster Recovery Synchronization",
            "Tiered Storage in Apache Kafka: S3 Cold Tier Offloading Policies",
        ],
        "tags": ["streaming", "kafka", "flink", "event_driven", "messaging"],
    },
    "networking": {
        "topics": [
            "TCP BBR Congestion Control vs CUBIC in High-Bandwidth Cloud Networks",
            "QUIC Protocol 0-RTT Connection Resumption Security and Replay Defense",
            "Linux Kernel eBPF XDP Packet Filtering for Line-Rate DDoS Mitigation",
            "BGP Anycast Routing and ECMP Path Balancing across Border Leaf Routers",
            "TLS 1.3 Early Data (0-RTT) Replay Attack Prevention and Session Tickets",
            "DNS Over HTTPS (DoH) Edge Resolver Latency and Cache Invalidation",
            "Linux Socket Buffer Tuning (rmem/wmem) for 100GbE High-Throughput NICs",
            "VPC Peering vs AWS Transit Gateway Cost and Bandwidth Scaling Analysis",
            "IPSec Tunnel Keepalive and Dead Peer Detection (DPD) Recovery Tuning",
            "HTTP/3 Multiplexing Head-of-Line Blocking Elimination Architecture",
        ],
        "tags": ["networking", "protocols", "tcp", "quic", "ebpf"],
    },
}

DOC_TYPES = [
    "architecture_spec",
    "rfc",
    "runbook",
    "incident_postmortem",
    "api_doc",
    "troubleshooting_guide",
]


def generate_document_content(
    doc_id: str,
    title: str,
    domain: str,
    doc_type: str,
    tags: List[str],
    version: str = "1.2.0",
    index_seed: int = 0,
) -> Tuple[Document, List[Dict[str, str]]]:
    """
    Generate deep, realistic markdown technical document content with headers,
    tables, parameter specs, code blocks, and operational guidance.
    Returns (Document, ground_truth_qa_pairs).
    """
    rng = random.Random(index_seed)

    # Key technical parameters for this specific doc
    threshold_ms = rng.choice([25, 50, 100, 250, 500, 1000])
    buffer_mb = rng.choice([64, 128, 256, 512, 1024, 2048])
    retry_count = rng.choice([3, 5, 7, 10])
    port_number = rng.choice([8080, 8443, 9092, 2379, 5432, 6379, 8500])
    algorithm = rng.choice(["Raft-v2", "BBRv3", "Leveled-Compaction", "StickyAssignor", "SPIRE-X509"])

    content = f"""# {title}

---
doc_id: {doc_id}
title: "{title}"
domain: "{domain}"
doc_type: "{doc_type}"
tags: "{', '.join(tags)}"
version: "{version}"
status: "PRODUCTION_APPROVED"
---

## Executive Summary
This document specifies the enterprise production architecture and operational procedures for **{title}**. 
Within the **{domain}** infrastructure domain, proper adherence to these specifications is mandatory to maintain 99.999% system availability, deterministic latencies, and strict zero-trust compliance.

## System Architecture & Specifications
The subsystem operates under high concurrency constraints and interfaces directly with cluster-wide orchestration layers.

### Configuration Parameters
The following table establishes the immutable production thresholds:

| Parameter Key | Recommended Value | Enforcement Level | Failure Impact |
| :--- | :--- | :--- | :--- |
| `max_latency_threshold_ms` | `{threshold_ms}ms` | Strict SLA | Circuit breaker tripped |
| `internal_buffer_capacity_mb` | `{buffer_mb}MB` | Hard Limit | Backpressure signal propagated |
| `max_retry_attempts` | `{retry_count}` | Kernel Policy | DLQ (Dead Letter Queue) rerouting |
| `default_listener_port` | `{port_number}` | Security Group | Connection dropped |
| `active_protocol_engine` | `{algorithm}` | Engine Core | Fallback to safe mode |

### Core Mechanical Flow
When a state change or I/O request reaches the `{algorithm}` controller on port `{port_number}`:
1. The ingress request is validated against the active token signature and passed to the primary thread ring buffer of `{buffer_mb}MB`.
2. If processing time exceeds `{threshold_ms}ms`, the system automatically increments the telemetry counter `sys_{domain}_latency_breach_total`.
3. If errors persist beyond `{retry_count}` consecutive attempts, the automated failover worker triggers partition isolation.

```yaml
# Production Config: {doc_id}.yaml
subsystem:
  domain: {domain}
  protocol_engine: "{algorithm}"
  listen_port: {port_number}
  buffer_limit_mb: {buffer_mb}
  timeout_ms: {threshold_ms}
  retry_policy:
    max_retries: {retry_count}
    backoff_factor: 1.5
    jitter_enabled: true
```

## Failure Modes and Troubleshooting Runbook

### Diagnostic Triage
If an on-call engineer detects alerts related to **{title}**:
- **Symptom 1: Buffer Saturation**: Check if memory consumption exceeds `{buffer_mb}MB`. If saturated, issue `kill -USR2` to dump heap profile.
- **Symptom 2: Connection Timeouts on Port {port_number}**: Verify firewall rules and execute `curl -Iv http://127.0.0.1:{port_number}/healthz`.
- **Symptom 3: Latency Spike above {threshold_ms}ms**: Inspect downstream locks and ensure the `{algorithm}` replication lag is under 10ms.

### Recovery Command Line
Execute the following verification sequence:
```bash
# Verify process health and active listener
netstat -tulpn | grep {port_number}

# Check cluster replication state
curl -s http://localhost:{port_number}/api/v1/cluster/status | jq .
```

## Security & Compliance
All communication must enforce TLS 1.3 with mutual authentication. Access logs are archived to immutable cold storage with a retention period of 365 days.
"""

    metadata = DocumentMetadata(
        doc_id=doc_id,
        title=title,
        doc_type=doc_type,
        domain=domain,
        tags=tags,
        version=version,
        source_path=f"data/raw/{doc_id}.md",
    )

    doc = Document(id=doc_id, content=content, metadata=metadata)

    # Ground truth questions tailored to test both exact parameter lookup and semantic concepts
    type_label = doc_type.replace('_', ' ').title()
    clean_topic = title.split('(')[0].strip()

    qa_pairs = [
        {
            "query": f"What is the recommended buffer capacity in the {type_label} for {clean_topic}?",
            "ground_truth_answer": f"The recommended buffer capacity is {buffer_mb}MB (internal_buffer_capacity_mb).",
            "ground_truth_doc_ids": [doc_id],
            "domain": domain,
        },
        {
            "query": f"What listener port and timeout threshold are configured in the {type_label} for {clean_topic}?",
            "ground_truth_answer": f"The listener port is {port_number} and the latency threshold is {threshold_ms}ms.",
            "ground_truth_doc_ids": [doc_id],
            "domain": domain,
        },
        {
            "query": f"How to resolve memory saturation exceeding {buffer_mb}MB in {doc_id}?",
            "ground_truth_answer": f"Issue kill -USR2 to dump heap profile when buffer saturation exceeds {buffer_mb}MB.",
            "ground_truth_doc_ids": [doc_id],
            "domain": domain,
        },
        {
            "query": f"Which counter tracks latency breaches for {doc_id} when processing exceeds the threshold in {domain}?",
            "ground_truth_answer": f"The telemetry counter sys_{domain}_latency_breach_total tracks breaches exceeding {threshold_ms}ms.",
            "ground_truth_doc_ids": [doc_id],
            "domain": domain,
        },
    ]

    return doc, qa_pairs


def generate_enterprise_corpus(
    output_dir: str | Path,
    target_count: int = 5000,
    save_to_disk: bool = True,
    seed: int = 42,
) -> Tuple[List[Document], List[EvaluationSample]]:
    """
    Generate 5,000+ technical documents across domains and produce
    a paired golden evaluation benchmark dataset for RAGAS evaluation.
    """
    rng = random.Random(seed)
    output_path = Path(output_dir)
    if save_to_disk:
        output_path.mkdir(parents=True, exist_ok=True)

    documents: List[Document] = []
    gold_evaluation_samples: List[EvaluationSample] = []

    domain_keys = list(DOMAINS.keys())

    # Build unique document variations
    doc_index = 0
    for domain_name in domain_keys:
        domain_info = DOMAINS[domain_name]
        topics = domain_info["topics"]
        base_tags = domain_info["tags"]

        for topic in topics:
            for doc_type in DOC_TYPES:
                # E.g. 8 domains * 10 topics * 6 doc_types = 480 core primary architecture docs
                doc_id = f"doc_{domain_name[:4]}_{doc_index:05d}"
                title = f"{topic} ({doc_type.replace('_', ' ').title()})"
                doc, qa_pairs = generate_document_content(
                    doc_id=doc_id,
                    title=title,
                    domain=domain_name,
                    doc_type=doc_type,
                    tags=base_tags + [doc_type],
                    version=f"{rng.randint(1, 4)}.{rng.randint(0, 9)}.{rng.randint(0, 9)}",
                    index_seed=doc_index + seed,
                )
                documents.append(doc)

                # Collect gold evaluation benchmarks from first 100 docs
                if len(gold_evaluation_samples) < 100:
                    for qa in qa_pairs:
                        gold_evaluation_samples.append(
                            EvaluationSample(
                                query=qa["query"],
                                ground_truth_answer=qa["ground_truth_answer"],
                                ground_truth_doc_ids=qa["ground_truth_doc_ids"],
                                domain=qa["domain"],
                            )
                        )

                doc_index += 1

    # Now scale up to target_count (e.g. 5,000+ docs) through systematic subsystem variants
    # representing component RFCs, regional clusters, and microservice service modules
    components = [
        "IngressProxy", "StateStore", "QueryRouter", "SyncCoordinator",
        "CacheLayer", "WorkerPool", "MetricsScraper", "AuditLogger",
        "EventDispatcher", "SnapshotManager", "TokenValidator", "ConnectionManager"
    ]
    environments = ["us-east-1", "us-west-2", "eu-central-1", "ap-southeast-1"]

    while len(documents) < target_count:
        domain_name = rng.choice(domain_keys)
        domain_info = DOMAINS[domain_name]
        base_topic = rng.choice(domain_info["topics"])
        comp = rng.choice(components)
        env = rng.choice(environments)
        doc_type = rng.choice(DOC_TYPES)

        doc_id = f"doc_{domain_name[:4]}_{doc_index:05d}"
        title = f"{base_topic} - {comp} Module [{env}] ({doc_type.replace('_', ' ').title()})"

        doc, qa_pairs = generate_document_content(
            doc_id=doc_id,
            title=title,
            domain=domain_name,
            doc_type=doc_type,
            tags=domain_info["tags"] + [comp.lower(), env],
            version=f"{rng.randint(1, 3)}.{rng.randint(0, 12)}.{rng.randint(0, 20)}",
            index_seed=doc_index + seed,
        )
        documents.append(doc)
        doc_index += 1

    # Save to disk if requested
    if save_to_disk:
        # Save a master JSON manifest and markdown files
        manifest_path = output_path / "corpus_manifest.json"
        manifest_data = [
            {
                "id": d.id,
                "title": d.metadata.title,
                "domain": d.metadata.domain,
                "doc_type": d.metadata.doc_type,
                "tags": d.metadata.tags,
            }
            for d in documents
        ]
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)

        # Save individual markdown files
        for doc in documents:
            file_path = output_path / f"{doc.id}.md"
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(doc.content)

        # Save gold benchmark samples
        benchmark_path = output_path.parent / "evaluations" / "gold_benchmark_dataset.json"
        benchmark_path.parent.mkdir(parents=True, exist_ok=True)
        with open(benchmark_path, "w", encoding="utf-8") as f:
            json.dump([s.model_dump() for s in gold_evaluation_samples], f, indent=2)

    return documents, gold_evaluation_samples
