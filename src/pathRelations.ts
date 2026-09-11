const DAY_SECONDS = 24 * 60 * 60;

export interface StoredPathRelation {
  source_key_id: string;
  bridge_memory_id: string;
  target_key_id: string;
  namespace: string;
  weight: number;
  active: boolean;
  evidence_count: number;
  last_evidence_id: string | null;
  updated_at: number;
}

export interface PathRelationRoute {
  target_key_id: string;
  bridge_memory_id: string;
  weight: number;
  evidence_count: number;
}

export interface PathRelationConfig {
  wallHalfLifeSeconds: number;
  opportunityHalfLife: number;
  promoteWeight: number;
  demoteWeight: number;
  minEvidenceEvents: number;
  maxWeight: number;
}

export const DEFAULT_PATH_RELATION_CONFIG: Readonly<PathRelationConfig> = Object.freeze({
  wallHalfLifeSeconds: 3 * DAY_SECONDS,
  opportunityHalfLife: 8,
  promoteWeight: 1.5,
  demoteWeight: 1,
  minEvidenceEvents: 2,
  maxWeight: 8,
});

function relationId(namespace: string, source: string, bridge: string, target: string): string {
  return `${namespace}\u0000${source}\u0000${bridge}\u0000${target}`;
}

export class PathRelationGraph {
  private readonly records = new Map<string, StoredPathRelation>();

  constructor(
    private readonly now: () => number = () => Date.now() / 1_000,
    private readonly config: Readonly<PathRelationConfig> = DEFAULT_PATH_RELATION_CONFIG,
  ) {}

  load(
    records: readonly StoredPathRelation[] | undefined,
    isValid: (record: StoredPathRelation) => boolean,
  ): boolean {
    this.records.clear();
    let repaired = false;
    for (const candidate of records ?? []) {
      if (!this.isWellFormed(candidate) || !isValid(candidate)) {
        repaired = true;
        continue;
      }
      const normalized = this.reconcile({
        ...candidate,
        weight: Math.min(candidate.weight, this.config.maxWeight),
      });
      const id = relationId(
        normalized.namespace,
        normalized.source_key_id,
        normalized.bridge_memory_id,
        normalized.target_key_id,
      );
      const existing = this.records.get(id);
      if (existing) {
        this.records.set(id, this.mergeRecords(existing, normalized));
        repaired = true;
      } else {
        this.records.set(id, normalized);
      }
      if (normalized.weight !== candidate.weight || normalized.active !== candidate.active) repaired = true;
    }
    return repaired;
  }

  serialize(): StoredPathRelation[] {
    return [...this.records.values()]
      .map((record) => ({ ...record }))
      .sort((left, right) =>
        left.namespace.localeCompare(right.namespace) ||
        left.source_key_id.localeCompare(right.source_key_id) ||
        left.bridge_memory_id.localeCompare(right.bridge_memory_id) ||
        left.target_key_id.localeCompare(right.target_key_id)
      );
  }

  observe(input: {
    namespace: string;
    sourceKeyId: string;
    bridgeMemoryId: string;
    targetKeyId: string | null;
    evidenceId: string;
  }): void {
    const namespace = input.namespace.trim();
    const source = input.sourceKeyId.trim();
    const bridge = input.bridgeMemoryId.trim();
    const target = input.targetKeyId?.trim() || null;
    const evidence = input.evidenceId.trim();
    if (!namespace || !source || !bridge || !evidence || target === source) return;

    const now = this.now();
    const opportunityFactor = 0.5 ** (1 / this.config.opportunityHalfLife);
    for (const record of this.records.values()) {
      if (
        record.namespace !== namespace ||
        record.source_key_id !== source ||
        record.bridge_memory_id !== bridge ||
        record.target_key_id === target
      ) continue;
      this.materializeWallDecay(record, now);
      record.weight *= opportunityFactor;
      record.updated_at = now;
      this.reconcile(record);
    }
    if (!target) return;

    const id = relationId(namespace, source, bridge, target);
    const existing = this.records.get(id);
    if (existing?.last_evidence_id === evidence) return;
    const record = existing ?? {
      source_key_id: source,
      bridge_memory_id: bridge,
      target_key_id: target,
      namespace,
      weight: 0,
      active: false,
      evidence_count: 0,
      last_evidence_id: null,
      updated_at: now,
    };
    if (existing) this.materializeWallDecay(record, now);
    record.weight = Math.min(this.config.maxWeight, record.weight + 1);
    record.evidence_count += 1;
    record.last_evidence_id = evidence;
    record.updated_at = now;
    this.records.set(id, this.reconcile(record));
  }

  routes(namespace: string, sourceKeyId: string, bridgeMemoryId: string): PathRelationRoute[] {
    const now = this.now();
    return [...this.records.values()]
      .filter((record) =>
        record.active &&
        record.namespace === namespace &&
        record.source_key_id === sourceKeyId &&
        record.bridge_memory_id === bridgeMemoryId
      )
      .map((record) => ({
        target_key_id: record.target_key_id,
        bridge_memory_id: record.bridge_memory_id,
        weight: this.effectiveWeight(record, now),
        evidence_count: record.evidence_count,
      }))
      .filter((route) => route.weight > this.config.demoteWeight)
      .sort((left, right) => right.weight - left.weight || left.target_key_id.localeCompare(right.target_key_id));
  }

  routesForSource(namespace: string, sourceKeyId: string): PathRelationRoute[] {
    const now = this.now();
    return [...this.records.values()]
      .filter((record) =>
        record.active &&
        record.namespace === namespace &&
        record.source_key_id === sourceKeyId
      )
      .map((record) => ({
        target_key_id: record.target_key_id,
        bridge_memory_id: record.bridge_memory_id,
        weight: this.effectiveWeight(record, now),
        evidence_count: record.evidence_count,
      }))
      .filter((route) => route.weight > this.config.demoteWeight)
      .sort((left, right) => right.weight - left.weight || left.target_key_id.localeCompare(right.target_key_id));
  }

  rewriteKey(fromId: string, intoId: string): void {
    if (!fromId || !intoId || fromId === intoId) return;
    const rewritten = new Map<string, StoredPathRelation>();
    for (const record of this.records.values()) {
      const source = record.source_key_id === fromId ? intoId : record.source_key_id;
      const target = record.target_key_id === fromId ? intoId : record.target_key_id;
      if (source === target) continue;
      const next = { ...record, source_key_id: source, target_key_id: target };
      const id = relationId(next.namespace, source, next.bridge_memory_id, target);
      const existing = rewritten.get(id);
      rewritten.set(id, existing ? this.mergeRecords(existing, next) : next);
    }
    this.records.clear();
    for (const [id, record] of rewritten) this.records.set(id, record);
  }

  prune(isValid: (record: StoredPathRelation) => boolean): boolean {
    let changed = false;
    for (const [id, record] of this.records) {
      if (isValid(record)) continue;
      this.records.delete(id);
      changed = true;
    }
    return changed;
  }

  private isWellFormed(record: StoredPathRelation): boolean {
    return Boolean(
      record &&
      typeof record.source_key_id === "string" && record.source_key_id &&
      typeof record.bridge_memory_id === "string" && record.bridge_memory_id &&
      typeof record.target_key_id === "string" && record.target_key_id &&
      record.source_key_id !== record.target_key_id &&
      typeof record.namespace === "string" && record.namespace.trim() === record.namespace && record.namespace &&
      Number.isFinite(record.weight) && record.weight > 0 &&
      typeof record.active === "boolean" &&
      Number.isInteger(record.evidence_count) && record.evidence_count > 0 &&
      (record.last_evidence_id === null || typeof record.last_evidence_id === "string") &&
      Number.isFinite(record.updated_at) && record.updated_at >= 0
    );
  }

  private effectiveWeight(record: StoredPathRelation, now: number): number {
    const elapsed = Math.max(0, now - record.updated_at);
    return record.weight * 0.5 ** (elapsed / this.config.wallHalfLifeSeconds);
  }

  private materializeWallDecay(record: StoredPathRelation, now: number): void {
    record.weight = this.effectiveWeight(record, now);
    record.updated_at = now;
    this.reconcile(record);
  }

  private reconcile(record: StoredPathRelation): StoredPathRelation {
    if (record.active) {
      if (record.weight <= this.config.demoteWeight) record.active = false;
    } else if (
      record.weight >= this.config.promoteWeight &&
      record.evidence_count >= this.config.minEvidenceEvents
    ) {
      record.active = true;
    }
    return record;
  }

  private mergeRecords(left: StoredPathRelation, right: StoredPathRelation): StoredPathRelation {
    const newest = left.updated_at >= right.updated_at ? left : right;
    return this.reconcile({
      ...newest,
      weight: Math.max(left.weight, right.weight),
      evidence_count: Math.max(left.evidence_count, right.evidence_count),
      active: left.active || right.active,
      updated_at: Math.max(left.updated_at, right.updated_at),
    });
  }
}
