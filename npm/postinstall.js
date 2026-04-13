#!/usr/bin/env node
/**
 * Postinstall: downloads the cuba-memorys binary from GitHub Releases
 * for the current platform/arch and places it in npm/.bin/cuba-memorys[.exe]
 */
const https = require("https");
const fs = require("fs");
const path = require("path");
const { execSync } = require("child_process");

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

function getArtifactName() {
  const platform = process.platform;
  const arch = process.arch;
  const key = `${platform}-${arch}`;
  const name = PLATFORM_MAP[key];
  if (!name) {
    throw new Error(
      `Unsupported platform: ${key}. Supported: ${Object.keys(PLATFORM_MAP).join(", ")}`
    );
  }
  return name;
}

function download(url, dest) {
  return new Promise((resolve, reject) => {
    const file = fs.createWriteStream(dest);
    const request = (u) => {
      https.get(u, (res) => {
        if (res.statusCode === 301 || res.statusCode === 302) {
          request(res.headers.location);
          return;
        }
        if (res.statusCode !== 200) {
          reject(new Error(`HTTP ${res.statusCode} for ${u}`));
          return;
        }
        res.pipe(file);
        file.on("finish", () => file.close(resolve));
      }).on("error", reject);
    };
    request(url);
  });
}

async function main() {
  // Skip in CI environments that don't need the binary
  if (process.env.CUBA_MEMORYS_SKIP_INSTALL) {
    console.log("cuba-memorys: skipping binary install (CUBA_MEMORYS_SKIP_INSTALL set)");
    return;
  }

  const artifactName = getArtifactName();
  const isWindows = process.platform === "win32";
  const binName = isWindows ? "cuba-memorys.exe" : "cuba-memorys";
  const binDest = path.join(BIN_DIR, binName);

  // Already installed
  if (fs.existsSync(binDest)) {
    console.log(`cuba-memorys: binary already installed at ${binDest}`);
    return;
  }

  if (!fs.existsSync(BIN_DIR)) {
    fs.mkdirSync(BIN_DIR, { recursive: true });
  }

  const url = `https://github.com/${REPO}/releases/download/v${VERSION}/${artifactName}`;
  console.log(`cuba-memorys: downloading binary from ${url}`);

  try {
    await download(url, binDest);
    if (!isWindows) {
      fs.chmodSync(binDest, 0o755);
    }
    console.log(`cuba-memorys: binary installed to ${binDest}`);
  } catch (err) {
    // Non-fatal: user can set DATABASE_URL and run directly
    console.warn(`cuba-memorys: failed to download binary — ${err.message}`);
    console.warn("  You can download it manually from:");
    console.warn(`  https://github.com/${REPO}/releases/tag/v${VERSION}`);
  }
}

main().catch((err) => {
  console.warn(`cuba-memorys postinstall warning: ${err.message}`);
  // Exit 0 so npm install doesn't fail
  process.exit(0);
});
