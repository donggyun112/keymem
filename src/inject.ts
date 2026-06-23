// Selection policy for inject-mode recall. Given associatively-expanded candidates (already in
// relevance order), choose which top-K to inject:
//   preferDepth   — surface confirmed (deep) memories first, so the injected set is more reliable
//                   (fewer low-precision neighbours in the noise slots).
//   exploreShallow — reserve one slot for the SHALLOWEST relevant candidate, so weak/recent
//                   memories occasionally resurface (and can be reinforced) instead of being
//                   permanently buried by deep ones. An ε-exploration on memory.
// Both are opt-in. NOTE: their *value* is longitudinal (it only shows once memories sit at
// different depths over real use) and is not captured by a one-shot benchmark — this module just
// makes the policy correct and testable.
export interface InjectCandidate {
  id: string;
  depth: number;
}

export function selectInject(
  candidates: InjectCandidate[],
  topK: number,
  opts: { preferDepth?: boolean; exploreShallow?: boolean } = {}
): string[] {
  if (topK <= 0 || candidates.length === 0) return [];
  let ranked = candidates;
  if (opts.preferDepth) {
    // depth desc, original relevance order as the tiebreak (stable).
    ranked = candidates
      .map((c, i) => ({ c, i }))
      .sort((x, y) => y.c.depth - x.c.depth || x.i - y.i)
      .map((x) => x.c);
  }
  const pick = ranked.slice(0, topK).map((c) => c.id);
  if (opts.exploreShallow && candidates.length > topK) {
    const shallowest = candidates.reduce((a, b) => (b.depth < a.depth ? b : a));
    if (!pick.includes(shallowest.id)) pick[pick.length - 1] = shallowest.id; // give a weak memory a slot
  }
  return pick;
}
