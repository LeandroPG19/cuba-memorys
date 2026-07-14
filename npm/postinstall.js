#!/usr/bin/env node
/**
 * Fetches the cuba-memorys binary from GitHub Releases for the current
 * platform/arch and places it in npm/.bin/cuba-memorys[.exe].
 *
 * # This runs at two moments, not one
 *
 * It is npm's `postinstall`, and it is ALSO called by `bin.js` on first use. That
 * second path is not a nicety — it is the only one that will survive. npm is moving
 * install-time lifecycle scripts to off-by-default (npm 12, mid-2026; the transition
 * flag landed in 11.15), because a postinstall is the single largest arbitrary-code
 * surface in the ecosystem. When postinstall does not run, `npm install -g cuba-memorys`
 * used to leave a package with no binary and say nothing, and the field report that
 * started this was exactly that: a Windows box where the install "succeeded" and the
 * command did not exist.
 *
 * So the download logic lives here as a reusable function, and `bin.js` calls it the
 * first time the binary is missing. Whichever moment gets there first wins; the other
 * finds the binary already present and returns.
 */
const https = require("https");
const fs = require("fs");
const path = require("path");

const VERSION = require("../package.json").version;
const REPO = "LeandroPG19/cuba-memorys";
const BIN_DIR = path.join(__dirname, ".bin");

const PLATFORM_MAP = {
  "linux-x64": "cuba-memorys-linux-x64",
  "linux-arm64": "cuba-memorys-linux-arm64",
  "darwin-x64": "cuba-memorys-macos-x64",
  "darwin-arm64": "cuba-memorys-macos-arm64",
  "win32-x64": "cuba-memorys-windows-x64.exe",
};

function artifactName() {
  const key = `${process.platform}-${process.arch}`;
  const name = PLATFORM_MAP[key];
  if (!name) {
    throw new Error(
      `Unsupported platform: ${key}. Supported: ${Object.keys(PLATFORM_MAP).join(", ")}`
    );
  }
  return name;
}

/** Absolute path where the binary for this platform belongs. */
function binPath() {
  const binName = process.platform === "win32" ? "cuba-memorys.exe" : "cuba-memorys";
  return path.join(BIN_DIR, binName);
}

function download(url, dest) {
  return new Promise((resolve, reject) => {
    const file = fs.createWriteStream(dest);
    const fail = (err) => {
      file.close();
      fs.unlink(dest, () => {});
      reject(err);
    };
    const request = (u) => {
      https
        .get(u, (res) => {
          if (res.statusCode === 301 || res.statusCode === 302) {
            request(res.headers.location);
            return;
          }
          if (res.statusCode !== 200) {
            fail(new Error(`HTTP ${res.statusCode} for ${u}`));
            return;
          }
          // A dropped connection fires the stream's 'finish' event all the same,
          // leaving a file that is shorter than promised but looks complete. The
          // binary would then be spawned and fail cryptically. So we hold the server
          // to its own Content-Length and reject a short read rather than publish it.
          const expected = Number(res.headers["content-length"]) || 0;
          let got = 0;
          res.on("data", (chunk) => {
            got += chunk.length;
          });
          res.pipe(file);
          file.on("finish", () => {
            file.close(() => {
              if (expected && got !== expected) {
                fs.unlink(dest, () => {});
                reject(new Error(`truncated download: got ${got} of ${expected} bytes`));
              } else {
                resolve();
              }
            });
          });
        })
        .on("error", fail);
    };
    request(url);
  });
}

/**
 * Ensure the binary is present, downloading it if not. Idempotent and safe to call
 * from either postinstall or first-use.
 *
 * Returns the binary path on success. Throws on failure — the caller decides whether
 * that is fatal (bin.js: yes, nothing to run) or a warning (postinstall: no, the user
 * can still install by other means).
 *
 * @param {(msg: string) => void} [log] where progress goes. Defaults to stderr, which
 *   is the only safe stream: this can run inside an MCP handshake on stdout, and one
 *   stray line of text there corrupts the JSON-RPC framing.
 */
async function ensureBinary(log = (m) => process.stderr.write(m + "\n")) {
  const dest = binPath();

  // A truncated binary that `existsSync` reports as present is worse than none: it
  // would be spawned and fail cryptically. `download` writes to a temp path and only
  // this rename publishes it, so a present file is always a complete one.
  if (fs.existsSync(dest) && fs.statSync(dest).size > 0) {
    return dest;
  }

  if (!fs.existsSync(BIN_DIR)) {
    fs.mkdirSync(BIN_DIR, { recursive: true });
  }

  const name = artifactName();
  const url = `https://github.com/${REPO}/releases/download/v${VERSION}/${name}`;
  const tmp = dest + ".part";

  log(`cuba-memorys: downloading ${VERSION} binary from ${url}`);
  await download(url, tmp);
  fs.renameSync(tmp, dest);
  if (process.platform !== "win32") {
    fs.chmodSync(dest, 0o755);
  }
  log(`cuba-memorys: binary installed to ${dest}`);
  return dest;
}

module.exports = { ensureBinary, binPath, artifactName, download };

// Invoked directly as the npm postinstall step.
if (require.main === module) {
  if (process.env.CUBA_MEMORYS_SKIP_INSTALL) {
    console.log("cuba-memorys: skipping binary install (CUBA_MEMORYS_SKIP_INSTALL set)");
    process.exit(0);
  }
  ensureBinary((m) => console.log(m)).catch((err) => {
    // Non-fatal here: bin.js will retry the download on first use. Exit 0 so a failed
    // download does not fail the whole `npm install`.
    console.warn(`cuba-memorys: postinstall could not fetch the binary — ${err.message}`);
    console.warn("  It will be retried automatically the first time you run cuba-memorys.");
    process.exit(0);
  });
}
