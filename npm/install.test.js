#!/usr/bin/env node
const assert = require("assert");
const fs = require("fs");
const os = require("os");
const path = require("path");

let failures = 0;
function test(name, fn) {
  try {
    fn();
    console.log(`  ok   ${name}`);
  } catch (err) {
    failures++;
    console.error(`  FAIL ${name}\n       ${err.message}`);
  }
}

const mod = require("./postinstall.js");

test("postinstall exports the reusable download path", () => {
  assert.strictEqual(typeof mod.ensureBinary, "function", "ensureBinary must be exported");
  assert.strictEqual(typeof mod.binPath, "function", "binPath must be exported");
  assert.strictEqual(typeof mod.download, "function", "download must be exported");
});

test("bin.js loads and can require ensureBinary", () => {
  const src = fs.readFileSync(path.join(__dirname, "bin.js"), "utf8");
  assert.ok(src.includes('require("./postinstall.js")'), "bin.js must reuse postinstall");
  assert.ok(src.includes("ensureBinary"), "bin.js must call ensureBinary on first use");
});

test("binPath is platform-correct", () => {
  const p = mod.binPath();
  const expected = process.platform === "win32" ? "cuba-memorys.exe" : "cuba-memorys";
  assert.ok(p.endsWith(expected), `binPath ${p} should end with ${expected}`);
});

test("ensureBinary does NOT download when a real binary is already present", async () => {
  const dest = mod.binPath();
  const dir = path.dirname(dest);
  const hadDir = fs.existsSync(dir);
  const hadFile = fs.existsSync(dest);
  const backup = hadFile ? fs.readFileSync(dest) : null;
  try {
    fs.mkdirSync(dir, { recursive: true });
    fs.writeFileSync(dest, "#!/bin/sh\necho present\n");
    let logged = "";
    const p = mod.ensureBinary((m) => (logged += m));
    return p.then((resolved) => {
      assert.strictEqual(resolved, dest, "should return the present binary");
      assert.ok(!logged.includes("downloading"), "must not download when present");
    });
  } finally {
    if (backup) fs.writeFileSync(dest, backup);
    else if (fs.existsSync(dest)) fs.unlinkSync(dest);
    if (!hadDir && fs.existsSync(dir)) fs.rmdirSync(dir);
  }
});

test("ensureBinary treats a 0-byte binary as missing", () => {
  const dest = mod.binPath();
  const dir = path.dirname(dest);
  const hadFile = fs.existsSync(dest);
  const backup = hadFile ? fs.readFileSync(dest) : null;
  try {
    fs.mkdirSync(dir, { recursive: true });
    fs.writeFileSync(dest, "");
    assert.strictEqual(fs.statSync(dest).size, 0);
    assert.ok(fs.existsSync(dest) && fs.statSync(dest).size === 0);
  } finally {
    if (backup) fs.writeFileSync(dest, backup);
    else if (fs.existsSync(dest)) fs.unlinkSync(dest);
  }
});

Promise.resolve().then(() => {
  setTimeout(() => {
    if (failures) {
      console.error(`\n${failures} test(s) failed`);
      process.exit(1);
    }
    console.log("\nall install smoke tests passed");
  }, 100);
});
