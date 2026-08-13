// Plugin manifests carry their own version, and both hosts key upgrade detection off it —
// a stale one means installed users never see the new release. Single source: package.json.
import { readFileSync, writeFileSync } from "node:fs";

const { version } = JSON.parse(readFileSync("package.json", "utf8"));
for (const path of ["plugin.json", ".claude-plugin/plugin.json"]) {
  const manifest = JSON.parse(readFileSync(path, "utf8"));
  if (manifest.version === version) continue;
  manifest.version = version;
  writeFileSync(path, `${JSON.stringify(manifest, null, 2)}\n`);
  console.log(`[sync-plugin-version] ${path} → ${version}`);
}
