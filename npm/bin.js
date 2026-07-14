#!/usr/bin/env node
/**
 * Entry point: locates and executes the cuba-memorys binary.
 *
 * Resolves in order:
 *   1. npm/.bin/cuba-memorys[.exe]  — what postinstall downloaded, for THIS version
 *   2. cuba-memorys in PATH         — system install / pip install, IF it matches
 *
 * Step 2 used to be unconditional, and that was a trap. When postinstall does not
 * run — `npm install --ignore-scripts`, standard practice in hardened CI, or a
 * download that failed — this file would spawn whatever `cuba-memorys` happened to
 * be on the PATH and say nothing. Installing 0.11.0 and silently running an 0.6.0
 * left over from an old pip install is not a fallback; it is a wrong answer
 * delivered confidently. Here it is also dangerous: the server applies migrations
 * on startup, so a stale binary does not merely misbehave — it reshapes a database
 * it was never meant to touch.
 *
 * So the PATH binary must now prove it is the version this package expects. One too
 * old to answer `--version` boots the whole server instead (exactly what 0.6.0
 * does), hits the timeout, and is refused.
 */
const { spawn, execFileSync } = require("child_process");
const path = require("path");
const fs = require("fs");

const EXPECTED = require("../package.json").version;
const isWindows = process.platform === "win32";
const binName = isWindows ? "cuba-memorys.exe" : "cuba-memorys";
const localBin = path.join(__dirname, ".bin", binName);

/** Version of a binary, or null if it cannot say — missing, too old, or hung. */
function probeVersion(bin) {
  try {
    const out = execFileSync(bin, ["--version"], {
      timeout: 5000,
      encoding: "utf8",
      stdio: ["ignore", "pipe", "ignore"],
    });
    const m = out.match(/(\d+\.\d+\.\d+)/);
    return m ? m[1] : null;
  } catch {
    return null; // ENOENT, non-zero exit, or the 5s timeout on a pre-0.11 binary
  }
}

function resolveBinary() {
  if (fs.existsSync(localBin)) return localBin;

  const found = probeVersion(binName);
  if (found === EXPECTED) return binName;

  const reason =
    found === null
      ? "no cuba-memorys on PATH (or it is too old to report its version)"
      : `PATH has cuba-memorys ${found}, but this package is ${EXPECTED}`;

  // `npm rebuild` ALSO obeys ignore-scripts, so telling someone to run it when
  // ignore-scripts is exactly what broke them sends them in a circle. It did, and the
  // command printed here was verified against a machine with `ignore-scripts=true`
  // set globally — which is a reasonable hardening, not a misconfiguration, and is
  // why this path is common rather than exotic.
  process.stderr.write(
    `cuba-memorys: cannot find the ${EXPECTED} binary — ${reason}.\n\n` +
      `Postinstall downloads it, so this means postinstall did not run. Almost always\n` +
      `that is npm's ignore-scripts (check: npm config get ignore-scripts).\n\n` +
      `Fix with one of:\n` +
      `  npm rebuild cuba-memorys --ignore-scripts=false --foreground-scripts\n` +
      `  pip install cuba-memorys\n` +
      `  https://github.com/LeandroPG19/cuba-memorys/releases/tag/v${EXPECTED}\n\n` +
      `(Plain \`npm rebuild\` will NOT work: it obeys ignore-scripts too.)\n\n` +
      `Refusing to run a different version: this server migrates the database it\n` +
      `connects to, so the wrong binary does not just misbehave — it rewrites schema.\n`
  );
  process.exit(1);
}

const proc = spawn(resolveBinary(), process.argv.slice(2), {
  stdio: "inherit",
  env: process.env,
});

proc.on("exit", (code) => process.exit(code ?? 1));
proc.on("error", (err) => {
  process.stderr.write(`cuba-memorys: failed to start binary — ${err.message}\n`);
  process.exit(1);
});
