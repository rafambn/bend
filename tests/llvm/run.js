#!/usr/bin/env bun

import * as child from "node:child_process";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import * as url from "node:url";

const DIR = path.dirname(url.fileURLToPath(import.meta.url));
const ROOT = path.resolve(DIR, "../..");
const CLI = path.join(ROOT, "bend2", "main.ts");
const TIMEOUT = 90000;
const WORKERS = 2;
const EXISTING = [
  "tests/compile/erased_argument.bend",
  "tests/compile/erased_partial_closure.bend",
  "tests/compile/closure_partial_app.bend",
  "tests/compile/closure_dynamic_apply.bend",
  "tests/compile/erasure_dead_param.bend",
  "tests/compile/gpu_mark_inert.bend",
  "tests/run/closures_hof.bend",
  "tests/run/fork_shapes.bend",
  "tests/run/array_fork.bend",
  "tests/run/computed_match.bend",
  "tests/run/nested_tuples_records.bend",
  "tests/run/bst_insert.bend",
  "tests/run/list_rebox_sum.bend",
  "tests/run/float_ops.bend",
  "tests/run/nat_native.bend",
  "tests/run/array_wrap_index.bend",
  "tests/run/array_size.bend",
  "tests/run/word_arithmetic.bend",
  "tests/base/u32_divmod_zero.bend",
  "tests/base/nat_divmod_zero.bend",
  "tests/base/show_read.bend",
  "tests/base/string_ops.bend",
  "tests/base/float_roundtrip.bend",
];

function want(file) {
  const lines = fs.readFileSync(file, "utf8").split("\n")
    .filter((line) => line.startsWith("#|")).map((line) => line.slice(2));
  const code = /^exit (\d+)$/.exec(lines.at(-1) ?? "");
  if (code !== null) lines.pop();
  return { output: tidy(lines.join("\n")), status: code === null ? 0 : Number(code[1]) };
}

function tidy(text) {
  return text.replace(/[ \t]+$/gm, "").trim();
}

function bend_files(dir) {
  return fs.readdirSync(dir, { withFileTypes: true }).sort((a, b) =>
    a.name < b.name ? -1 : a.name > b.name ? 1 : 0).flatMap((entry) => {
    const file = path.join(dir, entry.name);
    if (entry.isDirectory()) return bend_files(file);
    return entry.isFile() && entry.name.endsWith(".bend") ? [file] : [];
  });
}

function run(command, args, env = process.env) {
  return new Promise((resolve) => {
    const got = { status: null, signal: null, error: null, stdout: "", stderr: "",
      timedOut: false };
    let proc;
    try {
      proc = child.spawn(command, args, {
        cwd: ROOT,
        env: { ...env, BEND_NO_TELEMETRY: "1" },
        detached: process.platform !== "win32",
        stdio: ["ignore", "pipe", "pipe"],
      });
    } catch (error) {
      resolve({ ...got, error });
      return;
    }
    proc.stdout.setEncoding("utf8").on("data", (text) => { got.stdout += text; });
    proc.stderr.setEncoding("utf8").on("data", (text) => { got.stderr += text; });
    const alarm = setTimeout(() => {
      got.timedOut = true;
      try {
        if (proc.pid !== undefined && process.platform !== "win32") {
          process.kill(-proc.pid, "SIGKILL");
        } else {
          proc.kill("SIGKILL");
        }
      } catch {}
    }, TIMEOUT);
    proc.on("error", (error) => { got.error = error; });
    proc.on("close", (status, signal) => {
      clearTimeout(alarm);
      got.status = status;
      got.signal = signal;
      resolve(got);
    });
  });
}

async function invoke(file, args = [], env = process.env) {
  return run(process.execPath, [CLI, file, ...args], env);
}

function report(label, got, expected) {
  const observed = tidy(got.stdout + (got.status === 0 ? "" : got.stderr));
  const ok = got.error === null && !got.timedOut
    && got.status === expected.status && observed === expected.output;
  if (ok) {
    console.log("PASS " + label);
    return true;
  }
  console.error("FAIL " + label);
  console.error("  expected exit " + expected.status + ": "
    + JSON.stringify(expected.output));
  console.error("  observed: " + JSON.stringify(observed));
  if (got.timedOut) console.error("  timeout after " + TIMEOUT + "ms");
  if (got.error !== null) console.error("  process error: " + got.error.message);
  return false;
}

async function case_pool(cases) {
  let at = 0;
  let passed = 0;
  const workers = Array.from({ length: Math.min(WORKERS, cases.length) }, async () => {
    for (;;) {
      const i = at++;
      if (i >= cases.length) return;
      const test = cases[i];
      const hasForwardedArgs = test.file === path.join(DIR, "args.bend");
      const args = hasForwardedArgs
        ? ["--llvm", "north", "south"] : ["--llvm"];
      const got = await invoke(test.file, args);
      const expected = hasForwardedArgs
        ? { ...test.want, output: "2" } : test.want;
      if (report(test.label, got, expected)) passed += 1;
    }
  });
  await Promise.all(workers);
  return passed;
}

const flags = process.argv.slice(2);
if (flags.length > 1 || flags.some((flag) => !["--existing", "--all"].includes(flag))) {
  throw new Error("usage: bun tests/llvm/run.js [--existing | --all]");
}

const temp = fs.mkdtempSync(path.join(os.tmpdir(), "bend-llvm-test-"));
let passed = 0;
let total = 0;
try {
  const scalar = path.join(DIR, "scalars.bend");
  const ir = path.join(temp, "scalars.ll");
  const emptyPath = path.join(temp, "no-native-tools");
  fs.mkdirSync(emptyPath);
  const emitted = await invoke(scalar, ["-o", ir], {
    ...process.env,
    PATH: emptyPath,
    LLVM_CC: "bend-missing-clang",
    RUSTC: "bend-missing-rustc",
  });
  const emitOk = emitted.error === null && emitted.status === 0
    && fs.existsSync(ir) && /define\s+i32\s+@main\b/.test(fs.readFileSync(ir, "utf8"));
  console.log((emitOk ? "PASS " : "FAIL ") + "LLVM IR emission without native tools");
  if (!emitOk) console.error(emitted.stderr || emitted.error?.message || "invalid IR output");
  passed += Number(emitOk);
  total += 1;

  const binary = path.join(temp, "pure-main");
  const built = await invoke(path.join(DIR, "pure_main.bend"),
    ["--llvm", "-o", binary]);
  const pure = built.status === 0 && built.error === null
    ? await run(binary, []) : built;
  const buildOk = pure.error === null && pure.status === 0
    && tidy(pure.stdout) === want(path.join(DIR, "pure_main.bend")).output;
  console.log((buildOk ? "PASS " : "FAIL ") + "native binary build and run");
  if (!buildOk) {
    console.error("  expected: "
      + JSON.stringify(want(path.join(DIR, "pure_main.bend")).output));
    console.error("  observed: " + JSON.stringify(tidy(pure.stdout + pure.stderr)));
  }
  passed += Number(buildOk);
  total += 1;

  let rejectedTargets = true;
  for (const ext of [".c", ".js", ".mjs", ".cjs", ".bendtt"]) {
    const out = path.join(temp, "rejected" + ext);
    const got = await invoke(path.join(DIR, "pure_main.bend"),
      ["--llvm", "-o", out]);
    const message = got.stdout + got.stderr;
    const ok = got.error === null && !got.timedOut && got.status !== 0
      && message.includes("omit --llvm") && !fs.existsSync(out);
    rejectedTargets &&= ok;
    if (!ok) {
      console.error("  expected --llvm to reject " + ext + ": "
        + JSON.stringify(message));
    }
  }
  console.log((rejectedTargets ? "PASS " : "FAIL ")
    + "incompatible LLVM output extensions rejected");
  passed += Number(rejectedTargets);
  total += 1;

  const cases = fs.readdirSync(DIR).filter((name) => name.endsWith(".bend")).sort()
    .map((name) => ({ label: "llvm/" + name, file: path.join(DIR, name),
      want: want(path.join(DIR, name)) }));
  if (flags.includes("--all")) {
    const files = ["tests/run", "tests/compile", "tests/base"]
      .flatMap((dir) => bend_files(path.join(ROOT, dir)));
    cases.push(...files.map((file) => ({
      label: path.relative(ROOT, file).split(path.sep).join("/"),
      file,
      want: want(file),
    })));
  } else if (flags.includes("--existing")) {
    cases.push(...EXISTING.map((name) => ({ label: name, file: path.join(ROOT, name),
      want: want(path.join(ROOT, name)) })));
  }
  console.log("Running " + cases.length + " cases with at most " + WORKERS + " workers");
  passed += await case_pool(cases);
  total += cases.length;
} finally {
  fs.rmSync(temp, { recursive: true, force: true });
}

console.log("PASS: " + passed + " / " + total);
if (passed !== total) process.exitCode = 1;
