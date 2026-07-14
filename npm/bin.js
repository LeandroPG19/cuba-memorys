#!/usr/bin/env node
/**
 * Entry point: locates and executes the cuba-memorys binary.
 *
 * Resolves in order:
 *   1. npm/.bin/cuba-memorys[.exe]  — what postinstall downloaded, for THIS version
 *   2. download it now              — postinstall did not run; fetch on first use
 *   3. cuba-memorys in PATH         — system install / pip install, IF it matches
 *
 * Step 2 is why `npm install -g cuba-memorys` survives npm 12. Install-time lifecycle
 * scripts are moving to off-by-default (the postinstall that downloads the binary is
 * exactly the kind npm is disabling), and when it does not run the package ships with
 * no binary. The field report that prompted this was a Windows box where the install
 * "succeeded" and the command did not exist. So if the binary is missing, we fetch it
 * here, once, on first use — the same download the postinstall would have done.
 *
 * Step 3 must PROVE its version. Falling back to whatever `cuba-memorys` is on the PATH
 * once silently ran a leftover 0.6.0 in place of the installed version — and this server
 * migrates the database it connects to on startup, so the wrong binary does not merely
 * misbehave, it reshapes a schema it was never meant to touch.
 */
const { spawn, execFileSync } = require("child_process");
const path = require("path");
const fs = require("fs");
const { ensureBinary } = require("./postinstall.js");

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

async function resolveBinary() {
  if (fs.existsSync(localBin)) return localBin;

  // Postinstall did not run (npm 12 default, hardened CI, ignore-scripts). Fetch the
  // binary now — the whole point of doing this here is that it no longer depends on a
  // lifecycle script the ecosystem is turning off.
  try {
    return await ensureBinary();
  } catch (err) {
    process.stderr.write(
      `cuba-memorys: could not download the ${EXPECTED} binary — ${err.message}\n`
    );
    // The download is the primary path now; the PATH probe below is only a courtesy
    // for people who installed the binary some other way (pip, manual, package manager).
  }

  const found = probeVersion(binName);
  if (found === EXPECTED) return binName;

  const reason =
    found === null
      ? "no cuba-memorys on PATH either (or it is too old to report its version)"
      : `PATH has cuba-memorys ${found}, but this package is ${EXPECTED}`;

  process.stderr.write(
    `cuba-memorys: cannot obtain the ${EXPECTED} binary — ${reason}.\n\n` +
      `The download on first run failed (offline, or GitHub Releases unreachable).\n` +
      `Get it another way:\n` +
      `  pip install cuba-memorys\n` +
      `  https://github.com/LeandroPG19/cuba-memorys/releases/tag/v${EXPECTED}\n\n` +
      `Refusing to run a different version: this server migrates the database it\n` +
      `connects to, so the wrong binary does not just misbehave — it rewrites schema.\n`
  );
  process.exit(1);
}

resolveBinary().then((bin) => {
  const proc = spawn(bin, process.argv.slice(2), {
    stdio: "inherit",
    env: process.env,
  });
  proc.on("exit", (code) => process.exit(code ?? 1));
  proc.on("error", (err) => {
    process.stderr.write(`cuba-memorys: failed to start binary — ${err.message}\n`);
    process.exit(1);
  });
});
