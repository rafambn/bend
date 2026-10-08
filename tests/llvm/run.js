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

function native_scalar_call(ir) {
  return /\bcall\b[^\n]*@bend_(?:(?:u32|f32|nat)\(|op_(?:u32|f32|nat|bool)_[A-Za-z0-9_]+\()/.test(ir);
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
  const scalarIR = fs.existsSync(ir) ? fs.readFileSync(ir, "utf8") : "";
  const emitOk = emitted.error === null && emitted.status === 0
    && /define\s+i32\s+@main\b/.test(scalarIR)
    && /^; bend-runtime: rust$/m.test(scalarIR) && !native_scalar_call(scalarIR);
  console.log((emitOk ? "PASS " : "FAIL ") + "LLVM IR emission without native tools");
  if (!emitOk) console.error(emitted.stderr || emitted.error?.message || "invalid IR output");
  passed += Number(emitOk);
  total += 1;

  const mixedFile = path.join(DIR, "mixed_native_scalar_call.bend");
  const mixedIRFile = path.join(temp, "mixed-native-scalar-call.ll");
  const mixedResult = await invoke(mixedFile, ["-o", mixedIRFile], {
    ...process.env,
    PATH: emptyPath,
    LLVM_CC: "bend-missing-clang",
    RUSTC: "bend-missing-rustc",
  });
  const mixedIR = fs.existsSync(mixedIRFile) ? fs.readFileSync(mixedIRFile, "utf8") : "";
  const mixedFunctions = [...mixedIR.matchAll(
    /^define\s+([^\n{]+)\{\n([\s\S]*?)^\}/gm,
  )];
  const mixedNativeCall = mixedFunctions.some(([, signature, body]) =>
    !signature.includes("@bend_native_")
      && /\bcall\b[^\n]*@bend_native_[A-Za-z0-9_]*/.test(body));
  const mixedIRok = mixedResult.error === null && mixedResult.status === 0
    && /^; bend-runtime: rust$/m.test(mixedIR) && mixedNativeCall;
  console.log((mixedIRok ? "PASS " : "FAIL ")
    + "mixed LLVM path calls typed scalar kernel directly");
  if (!mixedIRok) {
    console.error(mixedResult.stderr || mixedResult.error?.message
      || "mixed IR lacks a direct native-kernel call from a general function");
  }
  passed += Number(mixedIRok);
  total += 1;

  const parallelFile = path.join(DIR, "parallel_native_call.bend");
  const parallelIRFile = path.join(temp, "parallel-native-call.ll");
  const parallelResult = await invoke(parallelFile, ["-o", parallelIRFile], {
    ...process.env,
    PATH: emptyPath,
    LLVM_CC: "bend-missing-clang",
    RUSTC: "bend-missing-rustc",
  });
  const parallelIR = fs.existsSync(parallelIRFile)
    ? fs.readFileSync(parallelIRFile, "utf8") : "";
  const parallelFunctions = [...parallelIR.matchAll(
    /^define\s+([^\n{]+)\{\n([\s\S]*?)^\}/gm,
  )];
  const parallelNativeCall = parallelFunctions.some(([, signature, body]) =>
    !signature.includes("@bend_native_")
      && /\bcall\b[^\n]*@bend_native_[A-Za-z0-9_]*/.test(body));
  const parallelIROk = parallelResult.error === null && parallelResult.status === 0
    && /^; bend-runtime: rust$/m.test(parallelIR)
    && /\bcall\s+i64\s+@bend_cpu_fork\(i64\s/.test(parallelIR)
    && /\bcall\s+i64\s+@bend_cpu_join\(i64\s/.test(parallelIR)
    && parallelNativeCall;
  console.log((parallelIROk ? "PASS " : "FAIL ")
    + "marked native kernels run through CPU fork/join workers");
  if (!parallelIROk) {
    console.error(parallelResult.stderr || parallelResult.error?.message
      || "parallel IR lacks CPU fork/join or a direct native worker call");
  }
  passed += Number(parallelIROk);
  total += 1;

  const arithmeticFile = path.join(DIR, "native_arithmetic.bend");
  const arithmeticIRFile = path.join(temp, "native-arithmetic.ll");
  const nativeIR = await invoke(arithmeticFile, ["-o", arithmeticIRFile], {
    ...process.env,
    PATH: emptyPath,
    LLVM_CC: "bend-missing-clang",
    RUSTC: "bend-missing-rustc",
  });
  const arithmeticIR = fs.existsSync(arithmeticIRFile)
    ? fs.readFileSync(arithmeticIRFile, "utf8") : "";
  const nativeTypes = new Set([...arithmeticIR.matchAll(
    /^define\s+(i32|float|i64|i1)\s+@bend_native_[^\s(]+\(/gm,
  )].map((match) => match[1]));
  const arithmeticIrOk = nativeIR.error === null && nativeIR.status === 0
    && /^; bend-runtime: none$/m.test(arithmeticIR)
    && /^define\s+i32\s+@bend_main\(/m.test(arithmeticIR)
    && ["i32", "float", "i64", "i1"].every((type) => nativeTypes.has(type))
    && /\badd i32\b/.test(arithmeticIR)
    && /\bfadd float\b/.test(arithmeticIR)
    && /\badd i64\b/.test(arithmeticIR)
    && !native_scalar_call(arithmeticIR);
  console.log((arithmeticIrOk ? "PASS " : "FAIL ")
    + "native scalar IR uses typed functions and arithmetic");
  if (!arithmeticIrOk) {
    console.error(nativeIR.stderr || nativeIR.error?.message
      || "native scalar IR lacks typed signatures, arithmetic, or runtime marker");
  }
  passed += Number(arithmeticIrOk);
  total += 1;

  const shiftsFile = path.join(DIR, "native_nat_shifts.bend");
  const shiftsIRFile = path.join(temp, "native-nat-shifts.ll");
  const shiftsResult = await invoke(shiftsFile, ["-o", shiftsIRFile], {
    ...process.env,
    PATH: emptyPath,
    LLVM_CC: "bend-missing-clang",
    RUSTC: "bend-missing-rustc",
  });
  const shiftsIR = fs.existsSync(shiftsIRFile) ? fs.readFileSync(shiftsIRFile, "utf8") : "";
  const natShiftFn = /^define\s+i32\s+@bend_native_[^\s(]+\(\s*i32\s+[^,()]+,\s*i64\s+[^,()]+\s*\)/m
    .test(shiftsIR);
  const shiftsIrOk = shiftsResult.error === null && shiftsResult.status === 0
    && /^; bend-runtime: none$/m.test(shiftsIR)
    && natShiftFn
    && /\bicmp\s+uge\s+i64\b/.test(shiftsIR)
    && /\btrunc\s+i64\s+[^\n]+\s+to\s+i32\b/.test(shiftsIR)
    && !native_scalar_call(shiftsIR);
  console.log((shiftsIrOk ? "PASS " : "FAIL ")
    + "native Nat shifts compare the full count before narrowing");
  if (!shiftsIrOk) {
    console.error(shiftsResult.stderr || shiftsResult.error?.message
      || "Nat shift IR lacks a typed i64 count or full-width bounds check");
  }
  passed += Number(shiftsIrOk);
  total += 1;

  const higherOrderFile = path.join(DIR, "mixed_higher_order_nat_shifts.bend");
  const higherOrderIRFile = path.join(temp, "mixed-higher-order-nat-shifts.ll");
  const higherOrderResult = await invoke(higherOrderFile, ["-o", higherOrderIRFile], {
    ...process.env,
    PATH: emptyPath,
    LLVM_CC: "bend-missing-clang",
    RUSTC: "bend-missing-rustc",
  });
  const higherOrderIR = fs.existsSync(higherOrderIRFile)
    ? fs.readFileSync(higherOrderIRFile, "utf8") : "";
  const higherOrderIRok = higherOrderResult.error === null && higherOrderResult.status === 0
    && /^; bend-runtime: rust$/m.test(higherOrderIR)
    && /\bcall\s+i64\s+@bend_apply\(i64\s/.test(higherOrderIR)
    && /\bdefine\s+i64\s+@bend_op_u32_shln_\d+\(/.test(higherOrderIR)
    && /\bdefine\s+i64\s+@bend_op_u32_shrn_\d+\(/.test(higherOrderIR);
  console.log((higherOrderIRok ? "PASS " : "FAIL ")
    + "higher-order Nat shifts use generic operation callbacks");
  if (!higherOrderIRok) {
    console.error(higherOrderResult.stderr || higherOrderResult.error?.message
      || "higher-order shift IR lacks the Rust callback path");
  }
  passed += Number(higherOrderIRok);
  total += 1;

  const bitsFile = path.join(DIR, "native_f32_u32_bits.bend");
  const bitsIRFile = path.join(temp, "native-f32-u32-bits.ll");
  const bitsResult = await invoke(bitsFile, ["-o", bitsIRFile], {
    ...process.env,
    PATH: emptyPath,
    LLVM_CC: "bend-missing-clang",
    RUSTC: "bend-missing-rustc",
  });
  const bitsIR = fs.existsSync(bitsIRFile) ? fs.readFileSync(bitsIRFile, "utf8") : "";
  const bitsIrOk = bitsResult.error === null && bitsResult.status === 0
    && /^; bend-runtime: none$/m.test(bitsIR)
    && /^define\s+i32\s+@bend_main\(/m.test(bitsIR)
    && /\bbitcast\s+i32\s+[^\n,]+\s+to\s+float\b/.test(bitsIR)
    && /\bbitcast\s+float\s+[^\n,]+\s+to\s+i32\b/.test(bitsIR)
    && !native_scalar_call(bitsIR);
  console.log((bitsIrOk ? "PASS " : "FAIL ")
    + "F32/U32 Word reinterpretation stays in native LLVM IR");
  if (!bitsIrOk) {
    console.error(bitsResult.stderr || bitsResult.error?.message
      || "F32/U32 bitcast IR needs a runtime bridge or lacks native bitcasts");
  }
  passed += Number(bitsIrOk);
  total += 1;

  const unionFile = path.join(DIR, "native_scalar_union.bend");
  const unionIRFile = path.join(temp, "native-scalar-union.ll");
  const unionResult = await invoke(unionFile, ["-o", unionIRFile], {
    ...process.env,
    PATH: emptyPath,
    LLVM_CC: "bend-missing-clang",
    RUSTC: "bend-missing-rustc",
  });
  const unionIR = fs.existsSync(unionIRFile) ? fs.readFileSync(unionIRFile, "utf8") : "";
  const unionIrOk = unionResult.error === null && unionResult.status === 0
    && /^; bend-runtime: none$/m.test(unionIR)
    && /^define\s+i32\s+@bend_main\(/m.test(unionIR)
    && /^define\s+i32\s+@bend_native_[^\s(]+\(/m.test(unionIR)
    && !native_scalar_call(unionIR)
    && !/\bcall\b[^\n]*@bend_(?:ctor|plain_ctor|apply|capture|field|tag)\b/.test(unionIR);
  console.log((unionIrOk ? "PASS " : "FAIL ")
    + "U32/F32 ADT arms lower to a native scalar result");
  if (!unionIrOk) {
    console.error(unionResult.stderr || unionResult.error?.message
      || "U32/F32 union IR uses a runtime bridge or lacks native functions");
  }
  passed += Number(unionIrOk);
  total += 1;

  const wordFile = path.join(DIR, "native_word_bridge.bend");
  const wordIRFile = path.join(temp, "native-word-bridge.ll");
  const wordResult = await invoke(wordFile, ["-o", wordIRFile], {
    ...process.env,
    PATH: emptyPath,
    LLVM_CC: "bend-missing-clang",
    RUSTC: "bend-missing-rustc",
  });
  const wordIR = fs.existsSync(wordIRFile) ? fs.readFileSync(wordIRFile, "utf8") : "";
  const wordValueArg = /^define\s+i32\s+@bend_native_[^\s(]+\(\s*i32\s+[^,()]+\s*\)/m
    .test(wordIR);
  const wordIrOk = wordResult.error === null && wordResult.status === 0
    && /^; bend-runtime: none$/m.test(wordIR)
    && /^define\s+i32\s+@bend_main\(/m.test(wordIR)
    && wordValueArg
    && !native_scalar_call(wordIR)
    && !/\bcall\b[^\n]*@bend_(?:ctor|plain_ctor|apply|capture)\b/.test(wordIR);
  console.log((wordIrOk ? "PASS " : "FAIL ")
    + "Word-to-U32 construction and matching stay native");
  if (!wordIrOk) {
    console.error(wordResult.stderr || wordResult.error?.message
      || "Word/U32 IR uses a runtime bridge or lacks native signatures");
  }
  passed += Number(wordIrOk);
  total += 1;

  const floatFile = path.join(DIR, "native_float_print.bend");
  const floatIRFile = path.join(temp, "native-float-print.ll");
  const floatIRResult = await invoke(floatFile, ["-o", floatIRFile], {
    ...process.env,
    PATH: emptyPath,
    LLVM_CC: "bend-missing-clang",
    RUSTC: "bend-missing-rustc",
  });
  const floatIR = fs.existsSync(floatIRFile) ? fs.readFileSync(floatIRFile, "utf8") : "";
  const floatIrOk = floatIRResult.error === null && floatIRResult.status === 0
    && /^; bend-runtime: none$/m.test(floatIR)
    && !native_scalar_call(floatIR);
  console.log((floatIrOk ? "PASS " : "FAIL ")
    + "native F32 printer emits IR without Rust runtime");
  if (!floatIrOk) {
    console.error(floatIRResult.stderr || floatIRResult.error?.message
      || "F32 printer IR needs the Rust runtime or scalar boxing helpers");
  }
  passed += Number(floatIrOk);
  total += 1;

  const binary = path.join(temp, "native-arithmetic");
  const built = await invoke(arithmeticFile,
    ["--llvm", "-o", binary], { ...process.env, RUSTC: "bend-missing-rustc" });
  const pure = built.status === 0 && built.error === null
    ? await run(binary, []) : built;
  const buildOk = pure.error === null && pure.status === 0
    && tidy(pure.stdout) === want(arithmeticFile).output;
  console.log((buildOk ? "PASS " : "FAIL ")
    + "native binary runs without Rust runtime");
  if (!buildOk) {
    console.error("  expected: "
      + JSON.stringify(want(arithmeticFile).output));
    console.error("  observed: " + JSON.stringify(tidy(pure.stdout + pure.stderr)));
  }
  passed += Number(buildOk);
  total += 1;

  const floatBinary = path.join(temp, "native-float-print");
  const floatBuilt = await invoke(floatFile,
    ["--llvm", "-o", floatBinary], { ...process.env, RUSTC: "bend-missing-rustc" });
  const floatRun = floatBuilt.status === 0 && floatBuilt.error === null
    ? await run(floatBinary, []) : floatBuilt;
  const floatBuildOk = floatRun.error === null && floatRun.status === 0
    && tidy(floatRun.stdout) === want(floatFile).output;
  console.log((floatBuildOk ? "PASS " : "FAIL ")
    + "native F32 printer matches JS output without Rust");
  if (!floatBuildOk) {
    console.error("  expected: " + JSON.stringify(want(floatFile).output));
    console.error("  observed: " + JSON.stringify(tidy(floatRun.stdout + floatRun.stderr)));
  }
  passed += Number(floatBuildOk);
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
