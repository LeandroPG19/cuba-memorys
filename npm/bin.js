#!/usr/bin/env node
/**
 * Entry point: locates and executes the cuba-memorys binary.
 * Resolves in order:
 *   1. npm/.bin/cuba-memorys[.exe]  (postinstall downloaded)
 *   2. cuba-memorys in PATH         (system install / pip install)
 */
const { spawn } = require("child_process");
const path = require("path");
const fs = require("fs");

const isWindows = process.platform === "win32";
const binName = isWindows ? "cuba-memorys.exe" : "cuba-memorys";
const localBin = path.join(__dirname, ".bin", binName);

const binary = fs.existsSync(localBin) ? localBin : "cuba-memorys";

const proc = spawn(binary, process.argv.slice(2), {
  stdio: "inherit",
  env: process.env,
});

proc.on("exit", (code) => process.exit(code ?? 1));
proc.on("error", (err) => {
  process.stderr.write(
    `cuba-memorys: failed to start binary — ${err.message}\n` +
    `  Install it via: pip install cuba-memorys\n` +
    `  Or download from: https://github.com/LeandroPG19/cuba-memorys/releases\n`
  );
  process.exit(1);
});
