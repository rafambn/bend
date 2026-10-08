// Direct Bend-to-LLVM code generation. This module consumes the checked book
// and emits LLVM IR that calls the Rust boxed-value runtime.

import * as Bend from "./bend.ts";

type Term = Bend.LTerm;
type HTerm = Bend.HTerm;
type Patt = Bend.Patt;
type Body = Bend.Body;
type Name = Bend.Name;

type FnText = { name: string; text: string };

const OPS = new Set((
  "u32_add u32_sub u32_and u32_or u32_xor u32_is_eq u32_is_ne u32_is_lt "
  + "u32_is_le u32_is_gt u32_is_ge u32_mul u32_div u32_mod u32_inc "
  + "u32_shl u32_shr u32_shln u32_shrn u32_not u32_is_zero u32_cmp "
  + "u32_to_f32 u32_to_nat u32_from_nat f32_add f32_sub f32_mul f32_div "
  + "f32_neg f32_is_eq f32_is_ne f32_is_lt f32_is_le f32_is_gt f32_is_ge "
  + "f32_sqrt f32_exp f32_log f32_log2 f32_log10 f32_sin f32_cos f32_tan "
  + "f32_asin f32_acos f32_atan f32_sinh f32_cosh f32_tanh f32_floor "
  + "f32_ceil f32_trunc f32_abs f32_atan2 f32_pow f32_mod f32_to_u32 "
  + "f32_bits f32_show f32_read nat_add nat_mul nat_double nat_cmp nat_sub "
  + "nat_is_lt nat_min nat_max nat_divmod bool_or bool_xor string_append "
  + "string_length array_new array_set array_get array_swap array_size "
  + "array_clone array_atomic_add array_atomic_min array_atomic_max array_atomic_exch "
  + "array_atomic_and array_atomic_or array_atomic_xor array_atomic_cas "
  + "array_atomic_fadd"
).split(/\s+/));

const BOOL_OPS = new Set([
  "u32_is_eq", "u32_is_ne", "u32_is_lt", "u32_is_le", "u32_is_gt",
  "u32_is_ge", "u32_is_zero", "f32_is_eq", "f32_is_ne", "f32_is_lt",
  "f32_is_le", "f32_is_gt", "f32_is_ge", "nat_is_lt", "bool_or",
  "bool_xor",
]);

const CMP_OPS = new Set(["u32_cmp", "nat_cmp"]);
const PAIR_OPS = new Set([
  "nat_divmod", "array_get", "array_size", "array_clone", "array_swap",
  "array_atomic_add", "array_atomic_min", "array_atomic_max", "array_atomic_exch", "array_atomic_and",
  "array_atomic_or", "array_atomic_xor", "array_atomic_cas", "array_atomic_fadd",
]);

const IO_CODES: Record<string, number> = {
  "IO.print": 1,
  "IO.write": 2,
  "IO.print_err": 3,
  "IO.args": 4,
  "IO.get_env": 5,
  "IO.random_u32": 6,
  "IO.sleep": 7,
  "IO.now": 8,
  "IO.thread_count": 9,
};

class Builder {
  readonly lines: string[] = [];
  private tempId = 0;
  private labelId = 0;
  current = "entry";
  terminated = false;

  constructor(readonly name: string, readonly params: string[]) {
    this.lines.push(`define i64 @${name}(${params.join(", ")}) {`, "entry:");
  }

  emit(line: string): void {
    this.lines.push(`  ${line}`);
    this.terminated = /^(br |ret |unreachable$)/.test(line);
  }

  temp(): string {
    return `%v${this.tempId++}`;
  }

  label(prefix = "block"): string {
    return `${prefix}${this.labelId++}`;
  }

  start(label: string): void {
    this.lines.push(`${label}:`);
    this.current = label;
    this.terminated = false;
  }

  finish(value: string): string {
    if (!this.terminated) this.emit(`ret i64 ${value}`);
    this.lines.push("}");
    return this.lines.join("\n");
  }
}

class LLVMCompiler {
  readonly fns: FnText[] = [];
  readonly callbackFns = new Set<string>();
  readonly strings = new Map<string, string>();
  readonly usedOps = new Map<string, number>();
  readonly usedIO = new Set<number>();
  readonly getNames = new Map<Name, string>();
  readonly opNames = new Map<string, string>();
  readonly ioNames = new Map<string, string>();
  readonly reachable = new Set<Name>();
  readonly ctorTags = new Map<Name, number>();
  private fresh = 0;
  private readonly ids = new Map<Name, number>();
  private tupleTag = 0;
  private trueTag = 0;
  private falseTag = 0;
  private ltTag = 0;
  private eqTag = 0;
  private gtTag = 0;
  private activeDef = "";

  constructor(readonly book: Bend.Book) {
    Object.keys(book.ctrs).sort().forEach((k, i) => this.ctorTags.set(k, i + 1));
    this.tupleTag = this.tagSuffix("Tuple");
    this.trueTag = this.tagSuffix("True");
    this.falseTag = this.tagSuffix("False");
    this.ltTag = this.tagSuffix("LT");
    this.eqTag = this.tagSuffix("EQ");
    this.gtTag = this.tagSuffix("GT");
  }

  compile(): string {
    this.collectReachable();
    for (const k of [...this.reachable].sort()) {
      const id = this.ids.get(k)!;
      this.getNames.set(k, `bend_get_${id}`);
    }
    const main = this.book.tlds.main;
    if (main?.$ !== "Def" || main.e === undefined) {
      throw new Error("LLVM backend: no checked main definition to compile");
    }
    this.emitAllGetters();
    const getMain = this.getNames.get("main")!;
    const io = this.mainReturnsIO(main.T);
    const entry = this.emitEntry(getMain, io);
    this.fns.push(entry);
    return this.render();
  }

  private render(): string {
    const declarations = [
      "declare i64 @bend_ctor(i64, i64, i64*)",
      "declare i64 @bend_plain_ctor(i64, i64, i64*)",
      "declare i64 @bend_tag(i64)",
      "declare i64 @bend_field(i64, i64)",
      "declare i64 @bend_nat(i64)",
      "declare i64 @bend_u32(i32)",
      "declare i64 @bend_f32(float)",
      "declare i64 @bend_string(i8*, i64)",
      "declare i64 @bend_array(i64, i64*)",
      "declare i64 @bend_closure(i64, i64*, i64)",
      "declare i64 @bend_capture(i64, i64)",
      "declare i64 @bend_apply(i64, i64)",
      "declare i64 @bend_tail_call(i64, i64)",
      "declare i64 @bend_cpu_fork(i64)",
      "declare i64 @bend_cpu_join(i64)",
      "declare void @bend_register_ctor(i64, i8*, i64)",
      "declare i64 @bend_io_action(i64, i64, i64)",
      "declare i32 @bend_show(i64)",
      "declare i32 @bend_io_run(i64)",
      "declare i32 @bend_runtime_init(i32, i8**)",
      "declare void @bend_runtime_shutdown()",
    ];
    for (const [name, arity] of this.usedOps) {
      declarations.push(`declare i64 @${name}(${Array(arity).fill("i64").join(", ")})`);
    }
    const globals = [...this.strings].map(([value, name]) => {
      const bytes = new TextEncoder().encode(value + "\0");
      return `@${name} = private unnamed_addr constant [${bytes.length} x i8] c"${this.llvmBytes(bytes)}", align 1`;
    });
    return [
      "; generated directly from a checked Bend book",
      ...globals,
      "",
      ...[...new Set(declarations)],
      "",
      ...this.fns.map((f) => f.text),
    ].join("\n");
  }

  private collectReachable(): void {
    const main = this.book.tlds.main;
    if (main?.$ !== "Def") throw new Error("LLVM backend: no main definition");
    const pending = ["main"];
    const seen = new Set<Name>();
    while (pending.length > 0) {
      const k = pending.pop()!;
      if (seen.has(k)) continue;
      seen.add(k);
      const tld = this.book.tlds[k];
      if (tld?.$ !== "Def") continue;
      if (this.isOperation(k)) {
        this.reachable.add(k);
        this.ids.set(k, this.ids.size);
        continue;
      }
      if (tld.i !== undefined) {
        if (!this.isIOHandler(k)) {
          const source = tld.i.join(", ");
          throw new Error(`LLVM backend: reachable foreign effect ${k} (${source}) has no LLVM handler; use a built-in IO operation or add an LLVM runtime handler`);
        }
        this.reachable.add(k);
        this.ids.set(k, this.ids.size);
        continue;
      }
      if (tld.e === undefined) {
        throw new Error(`LLVM backend: reachable definition ${k} has no checked body`);
      }
      this.reachable.add(k);
      this.ids.set(k, this.ids.size);
      for (const ref of this.references(tld.e)) {
        if (this.book.tlds[ref]?.$ === "Def") pending.push(ref);
      }
    }
  }

  private references(term: Term): Set<Name> {
    const refs = new Set<Name>();
    const walk = (node: unknown): void => {
      if (node === null || typeof node !== "object") return;
      if (Array.isArray(node)) {
        node.forEach(walk);
        return;
      }
      const obj = node as Record<string, unknown>;
      if (obj.$ === "Ref" && typeof obj.k === "string") {
        refs.add(obj.k);
        return;
      }
      if (obj.$ === "App") {
        walk(obj.f);
        const fn = obj.f as Term;
        const type = this.termType(fn);
        const all = type && Bend.term_wnf(this.book, type);
        if (!(all?.$ === "All" && all.q.$ === "None")) walk(obj.x);
        return;
      }
      if (obj.$ === "Let") {
        const letTerm = obj as unknown as Extract<Term, { $: "Let" }>;
        letTerm.v.forEach((value, i) => {
          if (letTerm.q[i]?.$ !== "None") walk(value);
        });
        walk(letTerm.f);
        return;
      }
      if (obj.$ === "Ctr") {
        const ctor = obj as unknown as Extract<Term, { $: "Ctr" }>;
        const qs = this.liveCtrQuantities(ctor.k, ctor.x.length);
        ctor.x.forEach((value, i) => {
          if (qs[i]?.$ !== "None") walk(value);
        });
        return;
      }
      if (obj.$ === "Rwt") {
        walk(obj.f);
        return;
      }
      if (obj.$ === "Eql" || obj.$ === "Rfl" || obj.$ === "Typ"
        || obj.$ === "All" || obj.$ === "Qnt" || obj.$ === "Qua"
        || obj.$ === "Min" || obj.$ === "ADT" || obj.$ === "Efq") return;
      for (const [key, value] of Object.entries(obj)) {
        if (key === "s" || key === "T" || key === "A" || key === "g" || key === "q") continue;
        walk(value);
      }
    };
    walk(term);
    return refs;
  }

  private emitAllGetters(): void {
    for (const k of [...this.reachable].sort()) {
      const tld = this.book.tlds[k];
      if (tld?.$ !== "Def") continue;
      if (this.isOperation(k)) {
        this.emitOperationGetter(k, tld);
      } else if (this.isIOHandler(k)) {
        this.emitIOGetter(k, tld);
      } else {
        this.activeDef = k;
        const b = new Builder(this.getNames.get(k)!, []);
        const value = this.emitTerm(b, tld.e!, new Map());
        this.fns.push({ name: b.name, text: b.finish(value) });
      }
    }
  }

  private emitOperationGetter(k: Name, tld: Bend.Def): void {
    const op = this.opName(k);
    const arity = Bend.tele_unbind(this.book, tld.T).doms.filter(([q]) => q.$ !== "None").length;
    const get = this.getNames.get(k)!;
    const b = new Builder(get, []);
    const value = this.emitStage(b, op, 0, arity, []);
    this.fns.push({ name: b.name, text: b.finish(value) });
  }

  private emitStage(parent: Builder, op: string, stage: number, arity: number, prior: string[]): string {
    if (arity === 0) {
      return this.emitOpCall(parent, op, []);
    }
    const name = this.newName(`op_${op}`);
    const callback = new Builder(name, ["i64 %env", "i64 %arg"]);
    const args: string[] = [];
    for (let i = 0; i < prior.length; i++) {
      const cap = callback.temp();
      callback.emit(`${cap} = call i64 @bend_capture(i64 %env, i64 ${i})`);
      args.push(cap);
    }
    args.push("%arg");
    const result = stage + 1 === arity
      ? this.emitOpCall(callback, op, args)
      : this.emitStage(callback, op, stage + 1, arity, args);
    this.addCallback(name, callback.finish(result));
    return this.closure(parent, name, prior);
  }

  private emitOpCall(b: Builder, op: string, args: string[]): string {
    const all = [...args];
    if (BOOL_OPS.has(op)) all.push(String(this.trueTag), String(this.falseTag));
    if (CMP_OPS.has(op)) all.push(String(this.ltTag), String(this.eqTag), String(this.gtTag));
    if (PAIR_OPS.has(op)) all.push(String(this.tupleTag));
    const name = `bend_op_${op}`;
    this.usedOps.set(name, all.length);
    const out = b.temp();
    b.emit(`${out} = call i64 @${name}(${all.map((a) => `i64 ${a}`).join(", ")})`);
    return out;
  }

  private emitIOGetter(k: Name, tld: Bend.Def): void {
    const opcode = IO_CODES[k];
    const io = this.book.tlds.IO;
    if (io?.$ !== "Def" || !io.b) {
      throw new Error(`LLVM backend: built-in effect ${k} requires Base.IO`);
    }
    // Keep IO opaque while reading the foreign definition's arguments. If
    // IO unfolds here, its continuation becomes an apparent extra parameter.
    const tlds = Object.assign(Object.create(this.book.tlds), { IO: { ...io, v: null } });
    const foreignBook = { ...this.book, tlds };
    const arity = Bend.tele_unbind(foreignBook, tld.T).doms
      .filter(([q]) => q.$ !== "None").length;
    const get = this.getNames.get(k)!;
    const b = new Builder(get, []);
    if (arity === 0) {
      const value = b.temp();
      b.emit(`${value} = call i64 @bend_io_action(i64 ${opcode}, i64 0, i64 0)`);
      this.usedIO.add(opcode);
      this.fns.push({ name: b.name, text: b.finish(value) });
      return;
    }
    const value = this.emitIOStage(b, opcode, 0, arity, []);
    this.fns.push({ name: b.name, text: b.finish(value) });
  }

  private emitIOStage(parent: Builder, opcode: number, stage: number, arity: number, prior: string[]): string {
    const name = this.newName(`io_${opcode}`);
    const callback = new Builder(name, ["i64 %env", "i64 %arg"]);
    const args: string[] = [];
    for (let i = 0; i < prior.length; i++) {
      const cap = callback.temp();
      callback.emit(`${cap} = call i64 @bend_capture(i64 %env, i64 ${i})`);
      args.push(cap);
    }
    args.push("%arg");
    let result: string;
    if (stage + 1 === arity) {
      const out = callback.temp();
      callback.emit(`${out} = call i64 @bend_io_action(i64 ${opcode}, i64 ${args[0] ?? "0"}, i64 ${args[1] ?? "0"})`);
      result = out;
      this.usedIO.add(opcode);
    } else {
      result = this.emitIOStage(callback, opcode, stage + 1, arity, args);
    }
    this.addCallback(name, callback.finish(result));
    return this.closure(parent, name, prior);
  }

  private emitEntry(getMain: string, io: boolean): FnText {
    const name = "main";
    const lines = [
      "define i32 @main(i32 %argc, i8** %argv) {",
      "entry:",
      "  %init = call i32 @bend_runtime_init(i32 %argc, i8** %argv)",
    ];
    for (const [k, tag] of this.ctorTags) {
      const tld = this.book.ctrs[k];
      const namePtr = this.stringPointer(k, lines);
      lines.push(`  call void @bend_register_ctor(i64 ${tag}, i8* ${namePtr}, i64 ${tld.n})`);
    }
    lines.push(`  %program = call i64 @${getMain}()`);
    lines.push(`  %status = call i32 @bend_${io ? "io_run" : "show"}(i64 %program)`);
    lines.push("  call void @bend_runtime_shutdown()", "  ret i32 %status", "}");
    return { name, text: lines.join("\n") };
  }

  private emitTerm(b: Builder, input: Term, env: Map<number, string>, hint?: HTerm, tail = false): string {
    const term = Bend.term_force(input);
    switch (term.$) {
      case "Ann":
        return this.emitTerm(b, term.x, env, term.T as HTerm, tail);
      case "Var": {
        const value = env.get(term.i);
        if (value === undefined) throw new Error(`LLVM backend: unbound variable ${term.k} at index ${term.i} in ${this.activeDef}`);
        return value;
      }
      case "Ref":
        return this.emitReference(b, term.k);
      case "Lit":
        return this.emitLiteral(b, term);
      case "Ctr":
        return this.emitConstructor(b, term.k, term.x, env);
      case "App": {
        const ft = this.termType(term.f);
        const all = ft && Bend.term_wnf(this.book, ft);
        if (all?.$ === "All" && all.q.$ === "None") return this.emitTerm(b, term.f, env, undefined, tail);
        const fn = this.emitTerm(b, term.f, env);
        const arg = this.emitTerm(b, term.x, env);
        const value = b.temp();
        const apply = tail ? "bend_tail_call" : "bend_apply";
        b.emit(`${value} = call i64 @${apply}(i64 ${fn}, i64 ${arg})`);
        return value;
      }
      case "Lam": {
        const ty = hint && Bend.term_wnf(this.book, hint);
        const q = ty?.$ === "All" ? ty.q : term.q;
        const local = new Map(env);
        if (q?.$ === "None") {
          local.set(term.i, "0");
        return this.emitTerm(b, term.f, local, undefined, tail);
        }
        return this.emitLambda(b, term.f, env, term.i);
      }
      case "Mat":
        return this.emitMat(b, term, env);
      case "Rwt":
        return this.emitTerm(b, term.f, env, undefined, tail);
      case "Let": {
        const local = new Map(env);
        const live = term.v.map((_, i) => i).filter((i) => term.q[i]?.$ !== "None");
        const values = new Map<number, string>();
        if (live.length > 1) {
          const tasks = live.map((i) => {
            const closure = this.emitThunk(b, term.v[i], env);
            const task = b.temp();
            b.emit(`${task} = call i64 @bend_cpu_fork(i64 ${closure})`);
            return [i, task] as const;
          });
          for (const [i, task] of tasks) {
            const value = b.temp();
            b.emit(`${value} = call i64 @bend_cpu_join(i64 ${task})`);
            values.set(i, value);
          }
        } else {
          for (const i of live) values.set(i, this.emitTerm(b, term.v[i], env));
        }
        for (let i = 0; i < term.v.length; i++) {
          local.set(term.i[i], values.get(i) ?? "0");
        }
        return this.emitTerm(b, term.f as Term, local, undefined, tail);
      }
      case "Eql":
      case "Rfl":
      case "Typ":
    case "All":
      return "0";
    case "Qnt":
      case "Qua":
      case "Min":
      case "ADT":
        return "0";
      case "Efq": {
        b.emit("unreachable");
        return "0";
      }
      case "Hol":
        throw new Error(`LLVM backend: unresolved hole ?${term.k} in ${this.activeDef}`);
      case "Sub":
        throw new Error(`LLVM backend: unsupported checked substitution in ${this.activeDef}`);
    }
  }

  private emitBody(b: Builder, body: Body, env: Map<number, string>, tail = false): string {
    if (body.$ === "Local") {
      const local = new Map(env);
      const values: string[] = [];
      if (body.v.length > 1) {
        const tasks: string[] = [];
        for (const term of body.v) {
          const thunk = this.emitThunk(b, term, env);
          const task = b.temp();
          b.emit(`${task} = call i64 @bend_cpu_fork(i64 ${thunk})`);
          tasks.push(task);
        }
        for (const task of tasks) {
          const value = b.temp();
          b.emit(`${value} = call i64 @bend_cpu_join(i64 ${task})`);
          values.push(value);
        }
      } else if (body.v.length === 1) {
        values.push(this.emitTerm(b, body.v[0], env));
      }
      for (let i = 0; i < body.k.length; i++) this.bindPattern(b, body.k[i], values[i] ?? "0", local);
      return this.emitBody(b, body.f, local, tail);
    }
    if (body.$ === "Match") return this.emitMatch(b, body, env, tail);
    return this.emitTerm(b, body, env, undefined, tail);
  }

  private emitMatch(b: Builder, match: Extract<Body, { $: "Match" }>, env: Map<number, string>, tail = false): string {
    const scrutinees = match.e.map((t) => this.emitTerm(b, t, env));
    const merge = b.label("match_end_");
    const incoming: string[] = [];
    let testBlock = b.current;
    const rows = match.r;
    for (let i = 0; i < rows.length; i++) {
      const row = rows[i];
      const fail = b.label("match_next_");
      const bodyLabel = b.label("match_case_");
      const local = new Map(env);
      let nextPattern = b.label("match_pat_");
      if (row.p.length === 0) {
        b.emit(`br label %${bodyLabel}`);
      } else {
        let current = row.p[0];
        this.emitPatternTest(b, current, scrutinees[0] ?? "0", nextPattern, fail, local);
        b.start(nextPattern);
        for (let j = 1; j < row.p.length; j++) {
          nextPattern = j === row.p.length - 1 ? bodyLabel : b.label("match_pat_");
          this.emitPatternTest(b, row.p[j], scrutinees[j] ?? "0", nextPattern, fail, local);
          if (nextPattern !== bodyLabel) b.start(nextPattern);
        }
        if (row.p.length === 1) b.emit(`br label %${bodyLabel}`);
      }
      b.start(bodyLabel);
      const value = this.emitBody(b, row.f, local, tail);
      if (!b.terminated) {
        const pred = b.current;
        b.emit(`br label %${merge}`);
        incoming.push(`[ ${value}, %${pred} ]`);
      }
      b.start(fail);
      testBlock = fail;
    }
    b.emit("unreachable");
    if (incoming.length === 0) {
      b.start(merge);
      b.emit("unreachable");
      return "0";
    }
    b.start(merge);
    const result = b.temp();
    b.emit(`${result} = phi i64 ${incoming.join(", ")}`);
    return result;
  }

  private emitPatternTest(b: Builder, pattern: Patt, value: string, success: string, failure: string, env: Map<number, string>): void {
    if (pattern.$ === "PVar") {
      env.set(pattern.i, pattern.q.$ === "None" ? "0" : value);
      b.emit(`br label %${success}`);
      return;
    }
    const tag = this.ctorTags.get(pattern.k);
    if (tag === undefined) throw new Error(`LLVM backend: unknown constructor ${pattern.k} in pattern`);
    const actual = b.temp();
    const ok = b.temp();
    b.emit(`${actual} = call i64 @bend_tag(i64 ${value})`);
    b.emit(`${ok} = icmp eq i64 ${actual}, ${tag}`);
    const children = this.patternChildren(pattern.k, pattern.x);
    if (children.length === 0) {
      b.emit(`br i1 ${ok}, label %${success}, label %${failure}`);
      return;
    }
    const first = b.label("pattern_field_");
    b.emit(`br i1 ${ok}, label %${first}, label %${failure}`);
    let fieldLabel = first;
    for (let i = 0; i < children.length; i++) {
      b.start(fieldLabel);
      const field = b.temp();
      b.emit(`${field} = call i64 @bend_field(i64 ${value}, i64 ${i})`);
      const next = i === children.length - 1 ? success : b.label("pattern_field_");
      this.emitPatternTest(b, children[i], field, next, failure, env);
      fieldLabel = next;
    }
  }

  private bindPattern(b: Builder, pattern: Patt, value: string, env: Map<number, string>): void {
    if (pattern.$ === "PVar") {
      env.set(pattern.i, pattern.q.$ === "None" ? "0" : value);
      return;
    }
    const children = this.patternChildren(pattern.k, pattern.x);
    for (let i = 0; i < children.length; i++) {
      const field = b.temp();
      b.emit(`${field} = call i64 @bend_field(i64 ${value}, i64 ${i})`);
      this.bindPattern(b, children[i], field, env);
    }
  }

  private emitConstructor(b: Builder, k: Name, xs: Term[], env: Map<number, string>): string {
    const tag = this.ctorTags.get(k);
    if (tag === undefined) throw new Error(`LLVM backend: unknown constructor ${k} in ${this.activeDef}`);
    const fields: string[] = [];
    const qs = this.liveCtrQuantities(k, xs.length);
    for (let i = 0; i < xs.length; i++) {
      if (qs[i]?.$ === "None") continue;
      fields.push(this.emitTerm(b, xs[i], env));
    }
    const family = Bend.book_fam(this.book, k);
    const builder = this.book.tlds[family]?.b === true ? "bend_ctor" : "bend_plain_ctor";
    return this.callVector(b, builder, [String(tag), String(fields.length)], fields);
  }

  private emitLiteral(b: Builder, term: Extract<Term, { $: "Lit" }>): string {
    const out = b.temp();
    if (term.k === "Nat") {
      b.emit(`${out} = call i64 @bend_nat(i64 ${BigInt(String(term.v))})`);
    } else if (term.k === "U32") {
      b.emit(`${out} = call i64 @bend_u32(i32 ${Number(term.v) >>> 0})`);
    } else if (term.k === "F32") {
      const word = Number(term.v) >>> 0;
      const float = b.temp();
      b.emit(`${float} = bitcast i32 ${word} to float`);
      b.emit(`${out} = call i64 @bend_f32(float ${float})`);
    } else {
      const text = String(term.v);
      const pointer = this.stringPointer(text, b.lines);
      const bytes = new TextEncoder().encode(text);
      b.emit(`${out} = call i64 @bend_string(i8* ${pointer}, i64 ${bytes.length})`);
    }
    return out;
  }

  private emitReference(b: Builder, k: Name): string {
    let get: string | undefined;
    if (this.isOperation(k)) {
      get = this.getNames.get(k) ?? this.operationName(k);
    } else if (this.isIOHandler(k)) {
      get = this.getNames.get(k) ?? this.ioName(k);
    } else {
      get = this.getNames.get(k);
    }
    if (get === undefined) throw new Error(`LLVM backend: unresolved reference ${k} in ${this.activeDef}`);
    const out = b.temp();
    b.emit(`${out} = call i64 @${get}()`);
    return out;
  }

  private emitLambda(b: Builder, body: Term, env: Map<number, string>, binder: number): string {
    const bound = new Set([binder]);
    const free = this.freeVariables(body, bound);
    const captures = [...env.entries()].filter(([index]) => free.has(index))
      .sort(([a], [b]) => a - b);
    const name = this.newName("lambda");
    const callback = new Builder(name, ["i64 %env", "i64 %arg"]);
    const child = new Map<number, string>();
    captures.forEach(([index], i) => {
      const value = callback.temp();
      callback.emit(`${value} = call i64 @bend_capture(i64 %env, i64 ${i})`);
      child.set(index, value);
    });
    child.set(binder, "%arg");
    const value = this.emitTerm(callback, body as Term, child, undefined, true);
    this.addCallback(name, callback.finish(value));
    return this.closure(b, name, captures.map(([, value]) => value));
  }

  private emitThunk(b: Builder, body: Term, env: Map<number, string>): string {
    const free = this.freeVariables(body);
    const captures = [...env.entries()].filter(([index]) => free.has(index))
      .sort(([a], [b]) => a - b);
    const name = this.newName("thunk");
    const callback = new Builder(name, ["i64 %env", "i64 %arg"]);
    const child = new Map<number, string>();
    captures.forEach(([index], i) => {
      const value = callback.temp();
      callback.emit(`${value} = call i64 @bend_capture(i64 %env, i64 ${i})`);
      child.set(index, value);
    });
    const value = this.emitTerm(callback, body, child, undefined, true);
    this.addCallback(name, callback.finish(value));
    return this.closure(b, name, captures.map(([, value]) => value));
  }

  private emitMat(b: Builder, term: Extract<Term, { $: "Mat" }>, env: Map<number, string>): string {
    const name = this.newName("mat");
    const callback = new Builder(name, ["i64 %env", "i64 %arg"]);
    const free = this.freeVariables(term.h);
    this.freeVariables(term.m).forEach((index) => free.add(index));
    const captures = [...env.entries()].filter(([index]) => free.has(index))
      .sort(([a], [b]) => a - b);
    const child = new Map<number, string>();
    captures.forEach(([index], i) => {
      const value = callback.temp();
      callback.emit(`${value} = call i64 @bend_capture(i64 %env, i64 ${i})`);
      child.set(index, value);
    });
    const tag = this.ctorTags.get(term.k);
    if (tag === undefined) throw new Error(`LLVM backend: unknown match constructor ${term.k}`);
    const actual = callback.temp();
    const yes = callback.temp();
    callback.emit(`${actual} = call i64 @bend_tag(i64 %arg)`);
    callback.emit(`${yes} = icmp eq i64 ${actual}, ${tag}`);
    const hit = callback.label("mat_hit_");
    const miss = callback.label("mat_miss_");
    const end = callback.label("mat_end_");
    callback.emit(`br i1 ${yes}, label %${hit}, label %${miss}`);
    callback.start(hit);
    const n = this.liveCtrQuantities(term.k, this.constructorArity(term.k))
      .filter((q) => q.$ !== "None").length;
    let handler = this.emitTerm(callback, term.h, child, undefined, n === 0);
    for (let i = 0; i < n; i++) {
      const field = callback.temp();
      callback.emit(`${field} = call i64 @bend_field(i64 %arg, i64 ${i})`);
      const applied = callback.temp();
      const apply = i + 1 === n ? "bend_tail_call" : "bend_apply";
      callback.emit(`${applied} = call i64 @${apply}(i64 ${handler}, i64 ${field})`);
      handler = applied;
    }
    const hitValue = handler;
    const hitPred = callback.current;
    callback.emit(`br label %${end}`);
    callback.start(miss);
    const missTerm = Bend.term_strip(term.m);
    if (missTerm.$ === "Efq") {
      callback.emit("unreachable");
      callback.start(end);
      this.addCallback(name, callback.finish(hitValue));
      return this.closure(b, name, captures.map(([, value]) => value));
    }
    const fallback = this.emitTerm(callback, term.m, child);
    const missValue = callback.temp();
    callback.emit(`${missValue} = call i64 @bend_tail_call(i64 ${fallback}, i64 %arg)`);
    const missPred = callback.current;
    callback.emit(`br label %${end}`);
    callback.start(end);
    const result = callback.temp();
    callback.emit(`${result} = phi i64 [ ${hitValue}, %${hitPred} ], [ ${missValue}, %${missPred} ]`);
    this.addCallback(name, callback.finish(result));
    return this.closure(b, name, captures.map(([, value]) => value));
  }

  private closure(b: Builder, name: string, captures: string[]): string {
    let ptr = "null";
    if (captures.length > 0) {
      const arr = b.temp();
      b.emit(`${arr} = alloca [${captures.length} x i64], align 8`);
      captures.forEach((v, i) => {
        const slot = b.temp();
        b.emit(`${slot} = getelementptr inbounds [${captures.length} x i64], [${captures.length} x i64]* ${arr}, i64 0, i64 ${i}`);
        b.emit(`store i64 ${v}, i64* ${slot}, align 8`);
      });
      const first = b.temp();
      b.emit(`${first} = getelementptr inbounds [${captures.length} x i64], [${captures.length} x i64]* ${arr}, i64 0, i64 0`);
      ptr = first;
    }
    const fp = b.temp();
    b.emit(`${fp} = ptrtoint i64 (i64, i64)* @${name} to i64`);
    const out = b.temp();
    b.emit(`${out} = call i64 @bend_closure(i64 ${fp}, i64* ${ptr}, i64 ${captures.length})`);
    return out;
  }

  private freeVariables(input: Term, outer = new Set<number>()): Set<number> {
    const free = new Set<number>();
    const walk = (input: Term, bound: Set<number>): void => {
      const term = Bend.term_force(input);
      switch (term.$) {
        case "Var":
          if (!bound.has(term.i)) free.add(term.i);
          return;
        case "Ann":
          walk(term.x, bound);
          return;
        case "App": {
          walk(term.f, bound);
          const type = this.termType(term.f);
          const all = type && Bend.term_wnf(this.book, type);
          if (!(all?.$ === "All" && all.q.$ === "None")) walk(term.x, bound);
          return;
        }
        case "Lam": {
          const local = new Set(bound);
          local.add(term.i);
          walk(term.f, local);
          return;
        }
        case "Let": {
          const quantities = term.q;
          term.v.forEach((value, i) => {
            if (quantities[i]?.$ !== "None") walk(value, bound);
          });
          const local = new Set(bound);
          term.i.forEach((index) => local.add(index));
          walk(term.f as Term, local);
          return;
        }
        case "Ctr": {
          const quantities = this.liveCtrQuantities(term.k, term.x.length);
          term.x.forEach((value, i) => {
            if (quantities[i]?.$ !== "None") walk(value, bound);
          });
          return;
        }
        case "Mat":
          walk(term.h, bound);
          walk(term.m, bound);
          return;
        case "Rwt":
          walk(term.f, bound);
          return;
        case "Eql":
        case "Rfl":
        case "Typ":
        case "Qnt":
        case "Qua":
        case "Min":
        case "ADT":
        case "Lit":
        case "Ref":
        case "Efq":
        case "Hol":
          return;
        case "All":
          return;
        case "Sub":
          walk(term.f, bound);
          return;
      }
    };
    walk(input, outer);
    return free;
  }

  private emitConstructorVector(b: Builder, tag: number, values: string[]): string {
    return this.callVector(b, "bend_ctor", [String(tag), String(values.length)], values);
  }

  private callVector(b: Builder, name: string, leading: string[], values: string[]): string {
    let ptr = "null";
    if (values.length > 0) {
      const arr = b.temp();
      b.emit(`${arr} = alloca [${values.length} x i64], align 8`);
      values.forEach((v, i) => {
        const slot = b.temp();
        b.emit(`${slot} = getelementptr inbounds [${values.length} x i64], [${values.length} x i64]* ${arr}, i64 0, i64 ${i}`);
        b.emit(`store i64 ${v}, i64* ${slot}, align 8`);
      });
      const first = b.temp();
      b.emit(`${first} = getelementptr inbounds [${values.length} x i64], [${values.length} x i64]* ${arr}, i64 0, i64 0`);
      ptr = first;
    }
    const out = b.temp();
    b.emit(`${out} = call i64 @${name}(${leading.map((v) => `i64 ${v}`).join(", ")}, i64* ${ptr})`);
    return out;
  }

  private termType(term: Term): HTerm | undefined {
    const t = Bend.term_force(term);
    return t.$ === "Ann" ? t.T as HTerm : undefined;
  }

  private liveCtrQuantities(k: Name, arity: number): Bend.Quant[] {
    const ctr = this.book.ctrs[k];
    if (!ctr) return Array(arity).fill(Bend.Lone());
    const doms = Bend.tele_unbind(this.book, ctr.T).doms.slice(-ctr.n);
    return Array.from({ length: arity }, (_, i) => doms[i]?.[0] ?? Bend.Lone());
  }

  private livePatternFields(k: Name, count: number): Patt[] {
    const ctr = this.book.ctrs[k];
    if (!ctr) return Array.from({ length: count }, (_, i) => ({ $: "PVar", k: "_", i: -1, q: Bend.Lone() }));
    const qs = this.liveCtrQuantities(k, count);
    return Array.from({ length: count }, (_, i) => i).filter((i) => qs[i]?.$ !== "None").map((i) => ({ $: "PVar", k: "_", i, q: Bend.Lone() }));
  }

  private patternChildren(k: Name, patterns: Patt[]): Patt[] {
    const qs = this.liveCtrQuantities(k, patterns.length);
    return patterns.filter((_, i) => qs[i]?.$ !== "None");
  }

  private constructorArity(k: Name): number {
    return this.book.ctrs[k]?.n ?? 0;
  }

  private mainReturnsIO(T: HTerm): boolean {
    const io = this.book.tlds.IO;
    if (io?.$ !== "Def" || !io.b) return false;
    const tlds = Object.assign(Object.create(this.book.tlds), { IO: { ...io, v: null } });
    const [head, args] = Bend.term_unapply(Bend.term_wnf({ ...this.book, tlds }, T));
    return head.$ === "Ref" && head.k === "IO" && args.length === 1;
  }

  private tagSuffix(name: string): number {
    const exact = this.ctorTags.get(name);
    if (exact !== undefined) return exact;
    const key = Object.keys(this.book.ctrs).find((k) => k === name || k.endsWith(`.${name}`));
    return key === undefined ? 0 : this.ctorTags.get(key) ?? 0;
  }

  private opName(k: Name): string {
    return k.toLowerCase().replace(/[./]/g, "_");
  }

  private isOperation(k: Name): boolean {
    const tld = this.book.tlds[k];
    return tld?.$ === "Def" && tld.b === true && OPS.has(this.opName(k));
  }

  private isIOHandler(k: Name): boolean {
    const tld = this.book.tlds[k];
    return tld?.$ === "Def" && tld.b === true && tld.i !== undefined
      && IO_CODES[k] !== undefined;
  }

  private operationName(k: Name): string {
    let name = this.opNames.get(k);
    if (!name) {
      name = `bend_get_op_${this.opNames.size}`;
      this.opNames.set(k, name);
    }
    return name;
  }

  private ioName(k: Name): string {
    let name = this.ioNames.get(k);
    if (!name) {
      name = `bend_get_io_${this.ioNames.size}`;
      this.ioNames.set(k, name);
    }
    return name;
  }

  private stringPointer(value: string, lines: string[]): string {
    let name = this.strings.get(value);
    if (!name) {
      name = `bend_str_${this.strings.size}`;
      this.strings.set(value, name);
    }
    const bytes = new TextEncoder().encode(value + "\0");
    const ptr = `%strptr${this.fresh++}`;
    lines.push(`  ${ptr} = getelementptr inbounds [${bytes.length} x i8], [${bytes.length} x i8]* @${name}, i64 0, i64 0`);
    return ptr;
  }

  private llvmBytes(bytes: Uint8Array): string {
    return [...bytes].map((b) => b >= 32 && b <= 126 && b !== 34 && b !== 92
      ? String.fromCharCode(b) : `\\${b.toString(16).padStart(2, "0").toUpperCase()}`).join("");
  }

  private newName(prefix: string): string {
    return `bend_${prefix}_${this.fresh++}`;
  }

  private addCallback(name: string, text: string): void {
    this.callbackFns.add(name);
    this.fns.push({ name, text });
  }
}

export function compile_book(book: Bend.Book): string {
  return new LLVMCompiler(book).compile();
}
