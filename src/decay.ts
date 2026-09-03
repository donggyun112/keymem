import { cfgName, cfgRaw } from "./env.js";

export type DecayProfile = "transient" | "standard" | "stable" | "permanent";
export type ConfirmationEvidence = "user" | "authoritative_source" | "observation";
export type FreshnessStatus = "fresh" | "aging" | "stale";

export interface DecayConfig {
  halfLivesSeconds: Record<Exclude<DecayProfile, "permanent">, number>;
}

export interface ValidityView {
  freshness: number;
  status: FreshnessStatus;
  age_days: number;
  last_confirmed_at: number;
  confirmation_count: number;
  decay_profile: DecayProfile;
  verification_recommended: boolean;
  verification_required: boolean;
}

const DAY = 24 * 3600;
export const DEFAULT_DECAY_CONFIG: DecayConfig = {
  halfLivesSeconds: { transient: 7 * DAY, standard: 90 * DAY, stable: 365 * DAY },
};

export function loadDecayConfig(
  read: (suffix: string) => string | undefined = cfgRaw,
  warn: (message: string) => void = console.error
): DecayConfig {
  const load = (suffix: string, fallback: number): number => {
    const raw = read(suffix);
    if (raw === undefined) return fallback;
    const days = Number(raw);
    if (Number.isFinite(days) && days > 0 && Number.isFinite(days * DAY)) return days * DAY;
    warn(`[decay] invalid ${cfgName(suffix)}=${JSON.stringify(raw)}; using ${fallback / DAY} days`);
    return fallback;
  };
  return {
    halfLivesSeconds: {
      transient: load("DECAY_TRANSIENT_DAYS", DEFAULT_DECAY_CONFIG.halfLivesSeconds.transient),
      standard: load("DECAY_STANDARD_DAYS", DEFAULT_DECAY_CONFIG.halfLivesSeconds.standard),
      stable: load("DECAY_STABLE_DAYS", DEFAULT_DECAY_CONFIG.halfLivesSeconds.stable),
    },
  };
}

export const DECAY_CONFIG = loadDecayConfig();

export function parseDecayProfile(value: unknown): DecayProfile {
  if (value === undefined || value === null) return "standard";
  if (value === "transient" || value === "standard" || value === "stable" || value === "permanent") return value;
  throw new Error(`Unknown decay profile: ${String(value)}`);
}

export function computeFreshness(
  lastConfirmedAt: number,
  profile: DecayProfile,
  now: number,
  config: DecayConfig = DECAY_CONFIG
): number {
  if (profile === "permanent") return 1;
  const age = Math.max(0, now - lastConfirmedAt);
  return 2 ** -(age / config.halfLivesSeconds[profile]);
}

export function freshnessRankFactor(freshness: number): number {
  return 0.2 + 0.8 * Math.max(0, Math.min(1, freshness));
}

export function buildValidityView(
  memory: { last_confirmed_at: number; confirmation_count: number; decay_profile: DecayProfile },
  now: number,
  config: DecayConfig = DECAY_CONFIG
): ValidityView {
  const age = Math.max(0, now - memory.last_confirmed_at);
  const raw = computeFreshness(memory.last_confirmed_at, memory.decay_profile, now, config);
  const status: FreshnessStatus = raw >= 0.5 ? "fresh" : raw >= 0.125 ? "aging" : "stale";
  return {
    freshness: Math.round(raw * 1000) / 1000,
    status,
    age_days: Math.round((age / DAY) * 1000) / 1000,
    last_confirmed_at: memory.last_confirmed_at,
    confirmation_count: memory.confirmation_count,
    decay_profile: memory.decay_profile,
    verification_recommended: status !== "fresh",
    verification_required: status === "stale",
  };
}
