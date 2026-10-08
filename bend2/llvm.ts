// Direct Bend-to-LLVM code generation from the checked book. Native kernels
// use typed LLVM values; heap values, closures, effects, and CPU tasks cross
// the tagged-value Rust runtime boundary.

import * as Bend from "./bend.ts";

type Term = Bend.LTerm;
type HTerm = Bend.HTerm;
type Patt = Bend.Patt;
type Body = Bend.Body;
type Name = Bend.Name;

type FnText = { name: string; text: string };

type NativeKind = "i32" | "i64" | "float" | "i1";
type NativeLay = {
  kind: "u32" | "f32" | "nat" | "bool" | "word" | "adt" | "unit";
  ks: NativeKind[];
  width?: number;
  family?: Name;
  arms?: Map<Name, NativeArm>;
  tagIndex?: number;
};
type NativeArm = { fields: NativeLay[]; tag: number; offsets: number[] };
type NativeVal = { ws: string[]; lay: NativeLay; type: HTerm };
type NativeDom = { q: Bend.Quant; k: Name; A: HTerm; lay: NativeLay | null; fixed?: HTerm };
type NativeSpec = {
  key: string;
  name: string;
  k: Name;
  def: Bend.Def;
  erased: HTerm[];
  doms: NativeDom[];
  live: NativeDom[];
  ret: HTerm;
  retLay: NativeLay;
  state: "new" | "compiling" | "done" | "failed";
};

class NativeUnsupported extends Error {}

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

  constructor(readonly name: string, readonly params: string[], readonly returnType = "i64") {
    this.lines.push(`define ${returnType} @${name}(${params.join(", ")}) {`, "entry:");
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
    if (!this.terminated) this.emit(`ret ${this.returnType} ${value}`);
    this.lines.push("}");
    return this.lines.join("\n");
  }
}

class LLVMCompiler {
  readonly fns: FnText[] = [];
  readonly callbackFns = new Set<string>();
  readonly strings = new Map<string, string>();
  readonly usedOps = new Map<string, number>();
  readonly usedMath = new Set<string>();
  needsTrap = false;
  readonly usedIO = new Set<number>();
  readonly getNames = new Map<Name, string>();
  readonly opNames = new Map<string, string>();
  readonly ioNames = new Map<string, string>();
  readonly reachable = new Set<Name>();
  readonly ctorTags = new Map<Name, number>();
  readonly nativeSpecs = new Map<string, NativeSpec>();
  readonly nativeDeps = new Map<string, Set<string>>();
  readonly nativeFns: FnText[] = [];
  readonly nativeTypes = new Map<string, NativeLay | null>();
  private activeNativeSpec: string | null = null;
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
    this.prepareNativeSpecs();
    const main = this.book.tlds.main;
    if (main?.$ !== "Def" || main.e === undefined) {
      throw new Error("LLVM backend: no checked main definition to compile");
    }
    const mainSpec = this.nativeSpecs.get(this.nativeSpecKey("main", []));
    if (mainSpec?.state === "done" && mainSpec.doms.length === 0 && this.canRenderNativeOnly()
      && this.canPrintNative(mainSpec.retLay)) {
      return this.renderNativeMain(mainSpec);
    }
    this.emitAllGetters();
    this.pruneNativeSpecs();
    this.fns.push(...this.nativeFns);
    const getMain = this.getNames.get("main")!;
    const io = this.mainReturnsIO(main.T);
    const entry = this.emitEntry(getMain, io);
    this.fns.push(entry);
    return this.render();
  }

  private prepareNativeSpecs(): void {
    const add = (k: Name, args: HTerm[]): NativeSpec | null => {
      const spec = this.makeNativeSpec(k, args);
      if (!spec) return null;
      if (!this.nativeSpecs.has(spec.key)) this.nativeSpecs.set(spec.key, spec);
      return this.nativeSpecs.get(spec.key)!;
    };
    const main = add("main", []);
    if (main) this.ensureNativeSpec(main);
    for (const k of [...this.reachable].sort()) {
      const tld = this.book.tlds[k];
      if (tld?.$ !== "Def" || tld.e === undefined || this.isOperation(k)
        || this.nativeOpName(k) !== null || this.isIOHandler(k)) continue;
      this.walkTerms(tld.e, (term) => {
        const [head, args] = Bend.term_unapply(term);
        const h = this.stripTerm(head);
        if (h.$ !== "Ref" || args.length === 0) return;
        const tld = this.book.tlds[h.k];
        if (tld?.$ !== "Def" || this.isOperation(h.k) || this.isIOHandler(h.k)) return;
        const spec = add(h.k, args);
        if (spec) this.ensureNativeSpec(spec);
      });
    }
    this.pruneNativeSpecs();
  }

  private ensureNativeSpec(spec: NativeSpec): boolean {
    if (spec.state === "done" || spec.state === "compiling") return true;
    if (spec.state === "failed") return false;
    spec.state = "compiling";
    const previous = this.activeNativeSpec;
    this.activeNativeSpec = spec.key;
    try {
      const text = this.emitNativeSpec(spec);
      spec.state = "done";
      this.nativeFns.push({ name: spec.name, text });
      return true;
    } catch (error) {
      if (!(error instanceof NativeUnsupported)) throw error;
      spec.state = "failed";
      return false;
    } finally {
      this.activeNativeSpec = previous;
    }
  }

  private pruneNativeSpecs(): void {
    let changed = true;
    while (changed) {
      changed = false;
      for (const spec of this.nativeSpecs.values()) {
        if (spec.state !== "done") continue;
        const deps = this.nativeDeps.get(spec.key);
        if (deps && [...deps].some((key) => this.nativeSpecs.get(key)?.state !== "done")) {
          spec.state = "failed";
          changed = true;
        }
      }
    }
    const liveNames = new Set([...this.nativeSpecs.values()]
      .filter((spec) => spec.state === "done").map((spec) => spec.name));
    for (let i = this.nativeFns.length - 1; i >= 0; i--) {
      if (!liveNames.has(this.nativeFns[i].name)) this.nativeFns.splice(i, 1);
    }
  }

  private noteNativeDependency(spec: NativeSpec): void {
    if (this.activeNativeSpec === null || this.activeNativeSpec === spec.key) return;
    let deps = this.nativeDeps.get(this.activeNativeSpec);
    if (!deps) this.nativeDeps.set(this.activeNativeSpec, deps = new Set());
    deps.add(spec.key);
  }

  private makeNativeSpec(k: Name, args: HTerm[]): NativeSpec | null {
    const tld = this.book.tlds[k];
    if (tld?.$ !== "Def" || tld.e === undefined || tld.i !== undefined
      || this.isOperation(k) || this.nativeOpName(k) !== null || this.isIOHandler(k)
      || args.length !== tld.n) return null;
    const keyArgs: HTerm[] = [];
    const domains: NativeDom[] = [];
    let tel = tld.T;
    for (let i = 0; i < tld.n; i++) {
      const all = Bend.tele_open(this.book, tel);
      if (!all) return null;
      const A = Bend.term_wnf(this.book, all.A);
      if (all.q.$ === "None") {
        const erased = args[i];
        if (erased === undefined) return null;
        keyArgs.push(erased);
        tel = all.B(erased);
        domains.push({ q: all.q, k: all.k, A, lay: null, fixed: erased });
      } else {
        const placeholder = Bend.Var(`__native_${i}`, -300000 - i);
        const actual = args[i];
        let fixed: HTerm | undefined;
        if (actual && this.natLiteral(actual) !== null) {
          const apply = all.B as unknown as (term: any) => HTerm;
          if (this.nativeNeedsStaticArgument(tld, i, apply(placeholder) as HTerm,
            apply(actual) as HTerm, args)) {
            if (this.natConstant(actual) !== null) {
              fixed = actual;
              keyArgs.push(actual);
            }
          }
        }
        const lay = this.nativeLayout(A);
        if (!lay) return null;
        domains.push({ q: all.q, k: all.k, A, lay, fixed });
        tel = (all.B as unknown as (term: any) => HTerm)(fixed ?? placeholder) as HTerm;
      }
    }
    const ret = Bend.term_wnf(this.book, tel);
    const retLay = this.nativeLayout(ret);
    if (!retLay) return null;
    const key = this.nativeSpecKey(k, keyArgs);
    const count = [...this.nativeSpecs.values()].filter((s) => s.k === k).length;
    const id = this.ids.get(k) ?? count;
    const name = k === "main" && tld.n === 0 && domains.length === 0
      ? "bend_main" : `bend_native_${id}_${count}`;
    return {
      key, name, k, def: tld,
      erased: keyArgs, doms: domains, live: domains.filter((d) => d.lay !== null && d.fixed === undefined),
      ret, retLay, state: "new",
    };
  }

  private nativeNeedsStaticArgument(tld: Bend.Def, index: number, placeholder: HTerm,
    actual: HTerm, args: HTerm[]): boolean {
    let p = placeholder;
    let a = actual;
    for (let i = index + 1; i < tld.n; i++) {
      const pDom = Bend.tele_open(this.book, p);
      const aDom = Bend.tele_open(this.book, a);
      if (!pDom || !aDom) return false;
      const pLay = this.nativeLayout(Bend.term_wnf(this.book, pDom.A));
      const aLay = this.nativeLayout(Bend.term_wnf(this.book, aDom.A));
      if (pLay === null && aLay !== null) return true;
      const arg = args[i];
      if (arg === undefined) return false;
      p = (pDom.B as unknown as (term: any) => HTerm)(arg) as HTerm;
      a = (aDom.B as unknown as (term: any) => HTerm)(arg) as HTerm;
    }
    return this.nativeLayout(Bend.term_wnf(this.book, p)) === null
      && this.nativeLayout(Bend.term_wnf(this.book, a)) !== null;
  }

  private nativeSpecKey(k: Name, erased: HTerm[]): string {
    return `${k}|${erased.map((arg) => {
      try {
        return Bend.term_key(Bend.term_lower(Bend.term_wnf(this.book, arg)));
      } catch {
        return JSON.stringify(Bend.term_lower(arg));
      }
    }).join("|")}`;
  }

  private nativeLayout(input: HTerm, stack = new Set<Name>()): NativeLay | null {
    const T = Bend.term_wnf(this.book, input);
    if (T.$ !== "ADT") return null;
    const key = Bend.term_key(Bend.term_lower(T));
    const cached = this.nativeTypes.get(key);
    if (cached) return cached;
    const cache = (lay: NativeLay): NativeLay => {
      this.nativeTypes.set(key, lay);
      return lay;
    };
    const tld = this.book.tlds[T.k];
    const builtin = tld?.b === true;
    const family = T.k.split(".").pop()!;
    if (builtin && family === "U32") return cache({ kind: "u32", ks: ["i32"], family: T.k });
    if (builtin && family === "F32") return cache({ kind: "f32", ks: ["float"], family: T.k });
    if (builtin && family === "Nat") return cache({ kind: "nat", ks: ["i64"], family: T.k });
    if (builtin && family === "Bool") return cache({ kind: "bool", ks: ["i1"], family: T.k });
    if (builtin && family === "Word") {
      const width = T.x[0] ? this.natConstant(T.x[0]) : null;
      if (width === null || width > 32) return null;
      return cache({ kind: "word", ks: ["i32"], width, family: T.k });
    }
    if (builtin && (T.k === "Word.Nil" || T.k.endsWith(".Word.Nil"))) {
      return cache({ kind: "word", ks: ["i32"], width: 0, family: T.k });
    }
    if (builtin && (T.k === "Word.Con" || T.k.endsWith(".Word.Con"))) {
      const tail = T.x[0] ? this.natConstant(T.x[0]) : null;
      if (tail === null || tail >= 32) return null;
      return cache({ kind: "word", ks: ["i32"], width: tail + 1, family: T.k });
    }
    if (T.k === "Array" || family === "String" || family === "IO" || stack.has(key)) return null;
    if (tld?.$ !== "ADT" || tld.c.length === 0) return null;
    const next = new Set(stack).add(key);
    const arms = new Map<Name, NativeArm>();
    const payloadKinds: NativeKind[] = [];
    let tagOffset = tld.c.length > 1 ? 1 : 0;
    for (let index = 0; index < tld.c.length; index++) {
      const ctr = tld.c[index];
      const filled = Bend.tele_fill(this.book, ctr.T, T.x, Bend.ctx_nil());
      const doms = Bend.tele_unbind(this.book, filled).doms.slice(-ctr.n);
      const fields: NativeLay[] = [];
      for (const [q, , A] of doms) {
        if (q.$ === "None") continue;
        const lay = this.nativeLayout(A, next);
        if (!lay) return null;
        fields.push(lay);
      }
      let offset = tagOffset;
      const offsets: number[] = [];
      for (const field of fields) {
        offsets.push(offset);
        field.ks.forEach((kind, j) => {
          payloadKinds[offset - tagOffset + j] = this.mergeNativeKind(payloadKinds[offset - tagOffset + j], kind);
        });
        offset += field.ks.length;
        if (offset > 247) return null;
      }
      arms.set(ctr.k, { fields, tag: index, offsets });
    }
    const max = payloadKinds.length;
    const ks: NativeKind[] = tld.c.length > 1 ? ["i32"] : [];
    for (let i = 0; i < max; i++) ks.push(payloadKinds[i] ?? "i32");
    if (ks.length > 247) return null;
    const out: NativeLay = { kind: "adt", ks, family: T.k, arms, tagIndex: tld.c.length > 1 ? 0 : -1 };
    return cache(out);
  }

  private mergeNativeKind(a: NativeKind | undefined, b: NativeKind): NativeKind {
    if (a === undefined || a === b) return b;
    if (a === "i64" || b === "i64") return "i64";
    if (a === "i1" && b === "i1") return "i1";
    if (a === "float" && b === "float") return "float";
    return "i32";
  }

  private natConstant(input: HTerm): number | null {
    const raw = this.stripTerm(input);
    if (raw.$ === "Lit" && raw.k === "Nat") return Number(raw.v);
    if (raw.$ === "Ctr") {
      const family = Bend.book_fam(this.book, raw.k);
      const tld = this.book.tlds[family];
      if (tld?.b === true && family.split(".").pop() === "Nat") {
        const name = raw.k.split(".").pop();
        if (name === "Zero") return 0;
        if (name === "Succ" && raw.x[0]) {
          const pred = this.natConstant(raw.x[0] as HTerm);
          return pred === null ? null : pred + 1;
        }
      }
    }
    const term = Bend.term_wnf(this.book, input);
    if (term.$ === "Lit" && term.k === "Nat") return Number(term.v);
    if (term.$ !== "Ctr") return null;
    const family = Bend.book_fam(this.book, term.k).split(".").pop();
    if (family !== "Nat") return null;
    const name = term.k.split(".").pop();
    if (name === "Zero") return 0;
    if (name === "Succ" && term.x[0]) {
      const pred = this.natConstant(term.x[0] as HTerm);
      return pred === null ? null : pred + 1;
    }
    return null;
  }

  private natLiteral(input: HTerm): number | null {
    let term = Bend.term_force(input);
    while (term.$ === "Ann" || term.$ === "Rwt") {
      term = Bend.term_force(term.$ === "Ann" ? term.x : term.f);
    }
    if (term.$ === "Lit" && term.k === "Nat") return Number(term.v);
    if (term.$ !== "Ctr") return null;
    const tld = this.book.tlds[Bend.book_fam(this.book, term.k)];
    if (tld?.b !== true || Bend.book_fam(this.book, term.k).split(".").pop() !== "Nat") return null;
    const name = term.k.split(".").pop();
    if (name === "Zero") return 0;
    if (name === "Succ" && term.x[0]) {
      const pred = this.natLiteral(term.x[0] as HTerm);
      return pred === null ? null : pred + 1;
    }
    return null;
  }

  private stripTerm(input: HTerm): HTerm {
    let term = Bend.term_force(input);
    while (term.$ === "Ann" || term.$ === "Rwt") term = Bend.term_force(term.$ === "Ann" ? term.x : term.f);
    return term;
  }

  private walkTerms(root: Term, visit: (term: HTerm) => void): void {
    const seen = new Set<HTerm>();
    const walk = (input: HTerm): void => {
      const term = Bend.term_force(input);
      if (seen.has(term)) return;
      seen.add(term);
      visit(term);
      switch (term.$) {
        case "Ann": walk(term.x); return;
        case "App": walk(term.f); walk(term.x); return;
        case "Lam": walk(term.f(Bend.Var(term.k, -400000 - seen.size))); return;
        case "Let": {
          term.v.forEach((value) => walk(value as HTerm));
          walk(term.f(term.i.map((i) => Bend.Var(`__let_${i}`, -410000 - seen.size - i))) as HTerm);
          return;
        }
        case "Ctr": term.x.forEach((value) => walk(value as HTerm)); return;
        case "Mat": walk(term.h); walk(term.m); return;
        case "Rwt": walk(term.e); walk(term.p); walk(term.f); return;
        case "Sub": walk(term.f); return;
        default: return;
      }
    };
    walk(Bend.term_higher(root));
  }

  private emitNativeSpec(spec: NativeSpec): string {
    const params: string[] = [];
    const env = new Map<number, NativeVal>();
    let erasedAt = 0;
    let liveAt = 0;
    let body: HTerm = Bend.term_higher(spec.def.e as Term);
    for (let i = 0; i < spec.doms.length; i++) {
      const dom = spec.doms[i];
      if (dom.fixed !== undefined) {
        body = Bend.term_apply(body, dom.fixed);
        if (dom.q.$ === "None") erasedAt++;
        continue;
      }
      if (!dom.lay) throw new NativeUnsupported();
      const ws = dom.lay.ks.map((kind, j) => {
        const name = `%arg${liveAt}_${j}`;
        params.push(`${kind} ${name}`);
        return name;
      });
      const id = -500000 - liveAt;
      const arg = Bend.Var(`__native_arg_${liveAt}`, id) as HTerm;
      const value: NativeVal = { ws, lay: dom.lay, type: dom.A };
      env.set(id, value);
      body = this.nativeApplyTerm(body, this.nativeAnnotation(arg, dom.A as HTerm));
      liveAt++;
    }
    void erasedAt;
    const builder = new Builder(spec.name, params, this.nativeReturnType(spec.retLay));
    const result = this.emitNativeTerm(builder, body, env, spec.retLay, spec.ret);
    return builder.finish(this.nativeReturnValue(builder, spec.retLay, result.ws));
  }

  private canRenderNativeOnly(): boolean {
    const root = this.nativeSpecs.get(this.nativeSpecKey("main", []));
    if (!root || root.state !== "done") return false;
    const pending = [root.key];
    const seen = new Set<string>();
    while (pending.length > 0) {
      const key = pending.pop()!;
      if (seen.has(key)) continue;
      seen.add(key);
      const spec = this.nativeSpecs.get(key);
      if (!spec || spec.state !== "done") return false;
      this.nativeDeps.get(key)?.forEach((dep) => pending.push(dep));
    }
    return true;
  }

  private nativeIsTuple(lay: NativeLay): boolean {
    if (lay.kind !== "adt" || !this.nativeBuiltinFamily(lay) || lay.arms?.size !== 1) return false;
    const [ctor] = [...lay.arms.keys()];
    return ctor !== undefined && this.nativeCtorShort(ctor) === "Tuple";
  }

  private canPrintNative(lay: NativeLay, seen = new Set<NativeLay>()): boolean {
    if (["u32", "nat", "f32", "bool"].includes(lay.kind)) return true;
    if (!this.nativeIsTuple(lay) || seen.has(lay)) return false;
    const next = new Set(seen).add(lay);
    const arms = lay.arms;
    const arm = arms && [...arms.values()][0];
    return !!arm && arm.fields.every((field) => this.canPrintNative(field, next));
  }

  private nativeTupleElements(lay: NativeLay, base = 0): { index: number; lay: NativeLay }[] {
    const arm = [...(lay.arms?.values() ?? [])][0];
    if (!arm || arm.fields.length !== 2) return [];
    const first = { index: base + arm.offsets[0], lay: arm.fields[0] };
    const second = arm.fields[1];
    const tail = this.nativeIsTuple(second)
      ? this.nativeTupleElements(second, base + arm.offsets[1])
      : [{ index: base + arm.offsets[1], lay: second }];
    return [first, ...tail];
  }

  private emitNativeTuplePrint(lay: NativeLay, base: number, components: string[], lines: string[]): void {
    lines.push(`  %open${this.fresh++} = call i32 @putchar(i32 40)`);
    const elements = this.nativeTupleElements(lay, base);
    elements.forEach((element, i) => {
      if (i > 0) lines.push(`  %comma${this.fresh++} = call i32 @putchar(i32 44)`, `  %space${this.fresh++} = call i32 @putchar(i32 32)`);
      if (this.nativeIsTuple(element.lay)) this.emitNativeTuplePrint(element.lay, element.index, components, lines);
      else this.emitNativeLeafPrint(element.lay, components[element.index] ?? "0", lines);
    });
    lines.push(`  %close${this.fresh++} = call i32 @putchar(i32 41)`);
  }

  private nativeContainsF32(lay: NativeLay): boolean {
    return lay.kind === "f32" || (lay.kind === "adt" && [...(lay.arms?.values() ?? [])]
      .some((arm) => arm.fields.some((field) => this.nativeContainsF32(field))));
  }

  private emitNativeLeafPrint(lay: NativeLay, value: string, lines: string[]): void {
    if (lay.kind === "u32") {
      const fmt = this.stringPointer("%u", lines);
      lines.push(`  %printed${this.fresh++} = call i32 (i8*, ...) @printf(i8* ${fmt}, i32 ${value})`);
    } else if (lay.kind === "nat") {
      const fmt = this.stringPointer("%llun", lines);
      lines.push(`  %printed${this.fresh++} = call i32 (i8*, ...) @printf(i8* ${fmt}, i64 ${value})`);
    } else if (lay.kind === "f32") {
      lines.push(`  call void @bend_native_print_f32(float ${value})`);
    } else if (lay.kind === "bool") {
      const yes = this.stringPointer("True{}", lines);
      const no = this.stringPointer("False{}", lines);
      const fmt = this.stringPointer("%s", lines);
      const chosen = `%boolstr${this.fresh++}`;
      lines.push(`  ${chosen} = select i1 ${value}, i8* ${yes}, i8* ${no}`);
      lines.push(`  %printed${this.fresh++} = call i32 (i8*, ...) @printf(i8* ${fmt}, i8* ${chosen})`);
    }
  }

  private renderNativeMain(spec: NativeSpec): string {
    const lines = [
      "define i32 @main(i32 %argc, i8** %argv) {",
      "entry:",
      `  %value = call ${this.nativeReturnType(spec.retLay)} @${spec.name}()`,
    ];
    const kind = spec.retLay.kind;
    if (kind === "u32") {
      const fmt = this.stringPointer("%u\n", lines);
      lines.push(`  %printed = call i32 (i8*, ...) @printf(i8* ${fmt}, i32 %value)`);
    } else if (kind === "nat") {
      const fmt = this.stringPointer("%llun\n", lines);
      lines.push(`  %printed = call i32 (i8*, ...) @printf(i8* ${fmt}, i64 %value)`);
    } else if (kind === "f32") {
      lines.push("  call void @bend_native_print_f32(float %value)", "  %newline = call i32 @putchar(i32 10)");
    } else if (kind === "adt") {
      const components = spec.retLay.ks.map((component, i) => {
        const name = `%show${i}`;
        lines.push(`  ${name} = extractvalue ${this.nativeReturnType(spec.retLay)} %value, ${i}`);
        return name;
      });
      this.emitNativeTuplePrint(spec.retLay, 0, components, lines);
      lines.push(`  %newline${this.fresh++} = call i32 @putchar(i32 10)`);
    } else {
      const yes = this.stringPointer("True{}\n", lines);
      const no = this.stringPointer("False{}\n", lines);
      lines.push("  br i1 %value, label %show_true, label %show_false", "show_true:");
      lines.push(`  %printed_true = call i32 (i8*, ...) @printf(i8* ${yes})`, "  br label %show_end", "show_false:");
      lines.push(`  %printed_false = call i32 (i8*, ...) @printf(i8* ${no})`, "  br label %show_end", "show_end:");
    }
    lines.push("  ret i32 0", "}");
    const globals = [...this.strings].map(([value, name]) => {
      const bytes = new TextEncoder().encode(value + "\0");
      return `@${name} = private unnamed_addr constant [${bytes.length} x i8] c"${this.llvmBytes(bytes)}", align 1`;
    });
    const declarations = ["declare i32 @printf(i8*, ...)"];
    if (kind === "f32" || kind === "adt") declarations.push("declare i32 @putchar(i32)");
    for (const name of [...this.usedMath].sort()) {
      declarations.push(`declare double @${name}(${name === "atan2" || name === "pow" || name === "fmod" ? "double, double" : "double"})`);
    }
    if (this.needsTrap) declarations.push("declare i64 @write(i32, i8*, i64)", "declare void @exit(i32)");
    return [
      "; generated directly from a checked Bend book",
      "; bend-runtime: none",
      ...globals,
      "",
      ...declarations,
      "",
      ...this.nativeFns.map((fn) => fn.text),
      ...(this.nativeContainsF32(spec.retLay) ? [native_f32_print_support()] : []),
      lines.join("\n"),
    ].join("\n");
  }

  private nativeReturnType(lay: NativeLay): string {
    if (lay.ks.length === 0) return "i8";
    if (lay.ks.length === 1) return lay.ks[0];
    return `{ ${lay.ks.join(", ")} }`;
  }

  private nativeReturnValue(b: Builder, lay: NativeLay, ws: string[]): string {
    if (lay.ks.length === 0) return "0";
    if (lay.ks.length === 1) return ws[0] ?? "0";
    let value = "zeroinitializer";
    const aggregate = this.nativeReturnType(lay);
    for (let i = 0; i < lay.ks.length; i++) {
      const next = b.temp();
      b.emit(`${next} = insertvalue ${aggregate} ${value}, ${lay.ks[i]} ${ws[i] ?? "0"}, ${i}`);
      value = next;
    }
    return value;
  }

  private nativeCallResult(b: Builder, lay: NativeLay, call: string): string[] {
    if (lay.ks.length === 0) return [];
    if (lay.ks.length === 1) return [call];
    return lay.ks.map((kind, i) => {
      const out = b.temp();
      b.emit(`${out} = extractvalue ${this.nativeReturnType(lay)} ${call}, ${i}`);
      return out;
    });
  }

  private emitNativeTerm(b: Builder, input: HTerm, env: Map<number, NativeVal>,
    expected: NativeLay, expectedType: HTerm): NativeVal {
    const term = Bend.term_force(input);
    switch (term.$) {
      case "Ann": {
        const T = Bend.term_wnf(this.book, term.T);
        const lay = this.nativeLayout(T) ?? expected;
        return this.emitNativeTerm(b, term.x, env, lay, T);
      }
      case "Var": {
        const value = env.get(term.i);
        if (!value) throw new NativeUnsupported(`native unbound variable ${term.k}`);
        return value;
      }
      case "Ref": {
        const spec = this.makeNativeSpec(term.k, []);
        if (!spec || spec.doms.length !== 0) throw new NativeUnsupported(`native value reference ${term.k}`);
        if (!this.nativeSpecs.has(spec.key)) this.nativeSpecs.set(spec.key, spec);
        const known = this.nativeSpecs.get(spec.key)!;
        this.noteNativeDependency(known);
        if (!this.ensureNativeSpec(known) && known.state !== "compiling") {
          throw new NativeUnsupported(`native value ${term.k} is outside the typed subset`);
        }
        const call = b.temp();
        const signature = this.nativeReturnType(known.retLay);
        b.emit(`${call} = call ${signature} @${known.name}()`);
        return { ws: this.nativeCallResult(b, known.retLay, call), lay: known.retLay, type: known.ret };
      }
      case "Lit": {
        const T = this.nativeTypeTerm(term.k);
        const lay = this.nativeLayout(T);
        if (!lay) throw new NativeUnsupported(`native literal ${term.k}`);
        if (term.k === "Nat") {
          const value = b.temp();
          b.emit(`${value} = add i64 0, ${term.v}`);
          return { ws: [value], lay, type: T };
        }
        if (term.k === "U32") {
          const value = b.temp();
          b.emit(`${value} = add i32 0, ${Number(term.v) >>> 0}`);
          return { ws: [value], lay, type: T };
        }
        if (term.k === "F32") {
          const value = b.temp();
          b.emit(`${value} = bitcast i32 ${Number(term.v) >>> 0} to float`);
          return { ws: [value], lay, type: T };
        }
        throw new NativeUnsupported("native strings are not scalar values");
      }
      case "Ctr":
        return this.emitNativeConstructor(b, term, env, expected, expectedType);
      case "App": {
        const [head, args] = Bend.term_unapply(term);
        return this.emitNativeApply(b, head, args, env, expected, expectedType);
      }
      case "Let": {
        if (term.v.filter((_, i) => term.q[i]?.$ !== "None").length > 1) {
          throw new NativeUnsupported("native multi-binding requires the CPU fork runtime");
        }
        const local = new Map(env);
        for (let i = 0; i < term.v.length; i++) {
          const index = term.i[i];
          const type = this.inferNativeType(term.v[i], local, expectedType);
          const lay = this.nativeLayout(type);
          if (!lay) throw new NativeUnsupported("native let value has open layout");
          const value = term.q[i]?.$ === "None"
            ? { ws: [], lay, type }
            : this.emitNativeTerm(b, term.v[i], local, lay, type);
          local.set(index, value);
        }
        const body = term.f(term.i.map((index, i) => Bend.Var(term.k[i] ?? "__native_let", index))) as HTerm;
        return this.emitNativeTerm(b, body, local, expected, expectedType);
      }
      case "Rwt":
        return this.emitNativeTerm(b, term.f, env, expected, expectedType);
      case "Efq":
        b.emit("unreachable");
        return { ws: [], lay: expected, type: expectedType };
      case "Mat":
      case "Ref":
      case "Sub":
      case "Eql":
      case "Rfl":
      case "Typ":
      case "All":
      case "Qnt":
      case "Qua":
      case "Min":
      case "ADT":
      case "Hol":
        throw new NativeUnsupported(`native expression ${term.$}`);
    }
    throw new NativeUnsupported("unsupported native expression");
  }

  private emitNativeApply(b: Builder, functionTerm: HTerm, args: HTerm[], env: Map<number, NativeVal>,
    expected: NativeLay, expectedType: HTerm): NativeVal {
    if (args.length === 0) return this.emitNativeTerm(b, functionTerm, env, expected, expectedType);
    const bareFunction = this.stripTerm(functionTerm);
    const [appliedHead, appliedArgs] = Bend.term_unapply(bareFunction);
    if (appliedArgs.length > 0) {
      return this.emitNativeApply(b, appliedHead, [...appliedArgs, ...args], env, expected, expectedType);
    }
    const head = bareFunction;
    if (head.$ === "Ref") {
      const op = this.nativeOpName(head.k);
      if (op) return this.emitNativeOperation(b, head.k, op, args, env, expected, expectedType);
      const tld = this.book.tlds[head.k];
      if (tld?.$ === "Def") {
        const spec = this.makeNativeSpec(head.k, args);
      if (!spec) throw new NativeUnsupported(`native call ${head.k} has no finite layout`);
      if (!this.nativeSpecs.has(spec.key)) this.nativeSpecs.set(spec.key, spec);
      const known = this.nativeSpecs.get(spec.key)!;
      this.noteNativeDependency(known);
        if (!this.ensureNativeSpec(known) && known.state !== "compiling") {
          throw new NativeUnsupported(`native call ${head.k} is outside the typed subset`);
        }
        const callArgs: string[] = [];
        for (let i = 0; i < known.doms.length; i++) {
          const dom = known.doms[i];
          if (dom.fixed !== undefined || dom.q.$ === "None") continue;
          const lay = dom.lay!;
          const value = this.emitNativeTerm(b, args[i], env, lay, dom.A);
          for (let j = 0; j < lay.ks.length; j++) {
            callArgs.push(`${lay.ks[j]} ${this.coerceNative(b, value.ws[j] ?? "0", lay.ks[j], lay.ks[j])}`);
          }
        }
        const result = b.temp();
        const sig = this.nativeReturnType(known.retLay);
        b.emit(`${result} = call ${sig} @${known.name}(${callArgs.join(", ")})`);
        return { ws: this.nativeCallResult(b, known.retLay, result), lay: known.retLay, type: known.ret };
      }
    }
    if (head.$ === "Lam") {
      const applied = Bend.term_apply(functionTerm, args[0]);
      return this.emitNativeApply(b, applied, args.slice(1), env, expected, expectedType);
    }
    if (head.$ === "Mat") {
      return this.emitNativeMatApply(b, head, args, env, expected, expectedType);
    }
    throw new NativeUnsupported("native higher-order application");
  }

  private inferNativeType(input: HTerm, env: Map<number, NativeVal>, expected: HTerm): HTerm {
    const term = Bend.term_force(input);
    if (term.$ === "Ann") return Bend.term_wnf(this.book, term.T);
    if (term.$ === "Var") return env.get(term.i)?.type ?? expected;
    if (term.$ === "Lit") return this.nativeTypeTerm(term.k);
    if (term.$ === "Ctr") {
      const family = Bend.book_fam(this.book, term.k);
      return Bend.ADT(family, []);
    }
    if (term.$ === "App") {
      const [head, args] = Bend.term_unapply(term);
      const h = this.stripTerm(head);
      if (h.$ === "Ref") {
        const op = this.nativeOpName(h.k);
        if (op) return this.nativeCallType(h.k, args);
        const spec = this.makeNativeSpec(h.k, args);
        if (spec) return spec.ret;
      }
    }
    return expected;
  }

  private nativeTypeTerm(name: string): HTerm {
    const exact = this.book.tlds[name]?.b === true ? name : undefined;
    const key = exact ?? Object.keys(this.book.tlds).find((k) =>
      this.book.tlds[k]?.b === true && k.split(".").pop() === name);
    if (!key) throw new NativeUnsupported(`no built-in type ${name}`);
    return Bend.ADT(key, []);
  }

  private nativeOpName(k: Name): string | null {
    if (this.isOperation(k)) return this.opName(k);
    const tld = this.book.tlds[k];
    if (tld?.$ !== "Def" || tld.b !== true) return null;
    const op = this.opName(k);
    return [
      "bool_and", "bool_not", "nat_is_eq", "word_zero", "word_inc", "word_add",
      "word_sub", "word_and", "word_or", "word_xor", "word_not", "word_shl",
      "word_shr", "word_to_nat",
    ].includes(op) ? op : null;
  }

  private nativeCallType(k: Name, args: HTerm[]): HTerm {
    const tld = this.book.tlds[k];
    if (tld?.$ !== "Def") throw new NativeUnsupported(`missing operation type ${k}`);
    let tel = tld.T;
    for (let i = 0; i < args.length; i++) {
      const all = Bend.tele_open(this.book, tel);
      if (!all) break;
      tel = all.B(args[i]);
    }
    return Bend.term_wnf(this.book, tel);
  }

  private nativeApplyTerm(fn: HTerm, arg: HTerm): HTerm {
    return (Bend.term_apply as (fn: any, arg: any) => HTerm)(fn, arg);
  }

  private nativeAnnotation(term: HTerm, type: HTerm): HTerm {
    return (Bend.Ann as (term: HTerm, type: HTerm) => HTerm)(term, type);
  }

  private nativeLiveArgs(k: Name, args: HTerm[]): { term: HTerm; type: HTerm }[] {
    const tld = this.book.tlds[k];
    if (tld?.$ !== "Def") return [];
    const live: { term: HTerm; type: HTerm }[] = [];
    let tel = tld.T;
    for (const arg of args) {
      const dom = Bend.tele_open(this.book, tel);
      if (!dom) break;
      if (dom.q.$ !== "None") live.push({ term: arg, type: Bend.term_wnf(this.book, dom.A) });
      tel = dom.B(arg);
    }
    return live;
  }

  private nativeCtorFields(k: Name, type: HTerm): { type: HTerm; lay: NativeLay }[] {
    const ctr = this.book.ctrs[k];
    if (!ctr) return [];
    const familyType = Bend.term_wnf(this.book, type);
    const params = familyType.$ === "ADT" ? familyType.x : [];
    const filled = Bend.tele_fill(this.book, ctr.T, params, Bend.ctx_nil());
    const doms = Bend.tele_unbind(this.book, filled).doms.slice(-ctr.n);
    const out: { type: HTerm; lay: NativeLay }[] = [];
    for (const [q, , A] of doms) {
      if (q.$ === "None") continue;
      const fieldType = Bend.term_wnf(this.book, A) as HTerm;
      const lay = this.nativeLayout(fieldType);
      if (!lay) throw new NativeUnsupported(`native constructor field in ${k} has no finite layout`);
      out.push({ type: fieldType, lay });
    }
    return out;
  }

  private nativeBuiltinFamily(lay: NativeLay): boolean {
    return !!lay.family && this.book.tlds[lay.family]?.b === true;
  }

  private nativeCtorShort(k: Name): string {
    return k.split(".").pop() ?? k;
  }

  private emitNativeConstructor(b: Builder, term: Extract<HTerm, { $: "Ctr" }>,
    env: Map<number, NativeVal>, expected: NativeLay, expectedType: HTerm): NativeVal {
    const type = Bend.term_wnf(this.book, expectedType);
    const family = type.$ === "ADT" ? type.k : Bend.book_fam(this.book, term.k);
    const builtin = this.book.tlds[family]?.b === true;
    const short = this.nativeCtorShort(term.k);

    if (builtin && family.split(".").pop() === "Nat" && (short === "Zero" || short === "Succ")) {
      if (short === "Zero") {
        const value = b.temp();
        b.emit(`${value} = add i64 0, 0`);
        return { ws: [value], lay: expected, type: expectedType };
      }
      const fields = this.nativeCtorFields(term.k, expectedType);
      const input = term.x.find((_, i) => this.liveCtrQuantities(term.k, term.x.length)[i]?.$ !== "None");
      const pred = input === undefined ? { ws: ["0"], lay: expected, type: expectedType }
        : this.emitNativeTerm(b, input as HTerm, env, fields[0]?.lay ?? expected, fields[0]?.type ?? expectedType);
      const next = b.temp();
      b.emit(`${next} = add i64 ${pred.ws[0] ?? "0"}, 1`);
      this.guardNat(b, next);
      return { ws: [next], lay: expected, type: expectedType };
    }
    if (builtin && family.split(".").pop() === "Bool" && (short === "True" || short === "False")) {
      const value = b.temp();
      b.emit(`${value} = add i1 0, ${short === "True" ? 1 : 0}`);
      return { ws: [value], lay: expected, type: expectedType };
    }
    if (builtin && (family.split(".").pop() === "U32" || family.split(".").pop() === "F32")
      && short === family.split(".").pop()) {
      const fields = this.nativeCtorFields(term.k, expectedType);
      const input = term.x.find((_, i) => this.liveCtrQuantities(term.k, term.x.length)[i]?.$ !== "None");
      const word = fields[0];
      if (input === undefined || !word) throw new NativeUnsupported(`missing Word(32) field in ${term.k}`);
      const value = this.emitNativeTerm(b, input as HTerm, env, word.lay, word.type);
      if (word.lay.kind !== "word" || word.lay.width !== 32) {
        throw new NativeUnsupported(`${term.k} requires a native Word(32) value`);
      }
      if (family.split(".").pop() === "U32") return { ws: [value.ws[0] ?? "0"], lay: expected, type: expectedType };
      const out = b.temp();
      b.emit(`${out} = bitcast i32 ${value.ws[0] ?? "0"} to float`);
      return { ws: [out], lay: expected, type: expectedType };
    }
    if (builtin && expected.kind === "word" && (short === "WNil" || short === "WCon")) {
      const width = expected.width ?? 0;
      if (short === "WNil") {
        const value = b.temp();
        b.emit(`${value} = add i32 0, 0`);
        return { ws: [value], lay: expected, type: expectedType };
      }
      if (width <= 0 || width > 32) throw new NativeUnsupported("native WCon requires Word width 1..32");
      const fields = this.nativeCtorFields(term.k, expectedType);
      const live = this.liveCtrQuantities(term.k, term.x.length);
      const indexes = term.x.map((_, i) => i).filter((i) => live[i]?.$ !== "None");
      if (indexes.length < 2 || fields.length < 2) throw new NativeUnsupported("malformed Word constructor");
      const bit = this.emitNativeTerm(b, term.x[indexes[0]] as HTerm, env, fields[0].lay, fields[0].type);
      const tail = this.emitNativeTerm(b, term.x[indexes[1]] as HTerm, env, fields[1].lay, fields[1].type);
      const shifted = b.temp();
      const low = b.temp();
      const out = b.temp();
      b.emit(`${shifted} = shl i32 ${tail.ws[0] ?? "0"}, 1`);
      const bit32 = this.coerceNative(b, bit.ws[0] ?? "0", "i1", "i32");
      b.emit(`${low} = and i32 ${bit32}, 1`);
      b.emit(`${out} = or i32 ${shifted}, ${low}`);
      const mask = width === 32 ? 0xffff_ffff : (2 ** width) - 1;
      if (mask !== 0xffff_ffff) {
        const masked = b.temp();
        b.emit(`${masked} = and i32 ${out}, ${mask}`);
        return { ws: [masked], lay: expected, type: expectedType };
      }
      return { ws: [out], lay: expected, type: expectedType };
    }

    if (expected.kind !== "adt" || expected.family !== family) {
      throw new NativeUnsupported(`native constructor ${term.k} does not inhabit its expected layout`);
    }
    const arm = expected.arms?.get(term.k);
    if (!arm) throw new NativeUnsupported(`native constructor ${term.k} is absent from its family layout`);
    const fieldInfo = this.nativeCtorFields(term.k, expectedType);
    const live = this.liveCtrQuantities(term.k, term.x.length);
    const indexes = term.x.map((_, i) => i).filter((i) => live[i]?.$ !== "None");
    const ws = Array.from({ length: expected.ks.length }, (_, i) => {
      const zero = b.temp();
      b.emit(`${zero} = ${expected.ks[i] === "float" ? "fadd float 0.0, 0.0" : `add ${expected.ks[i]} 0, 0`}`);
      return zero;
    });
    if (expected.tagIndex !== undefined && expected.tagIndex >= 0) {
      const tag = b.temp();
      b.emit(`${tag} = add i32 0, ${arm.tag}`);
      ws[expected.tagIndex] = tag;
    }
    for (let i = 0; i < Math.min(arm.fields.length, indexes.length); i++) {
      const field = this.emitNativeTerm(b, term.x[indexes[i]] as HTerm, env, arm.fields[i], fieldInfo[i]?.type ?? expectedType);
      const offset = arm.offsets[i];
      for (let j = 0; j < arm.fields[i].ks.length; j++) {
        ws[offset + j] = this.coerceNative(b, field.ws[j] ?? "0", arm.fields[i].ks[j], expected.ks[offset + j]);
      }
    }
    return { ws, lay: expected, type: expectedType };
  }

  private nativeProjectFields(b: Builder, ctor: Name, value: NativeVal, type: HTerm): NativeVal[] {
    const lay = value.lay;
    const family = lay.family ?? Bend.book_fam(this.book, ctor);
    const builtin = this.book.tlds[family]?.b === true;
    const short = this.nativeCtorShort(ctor);
    if (builtin && family.split(".").pop() === "Nat" && short === "Succ") {
      const fieldType = Bend.ADT(family, []) as HTerm;
      const one = b.temp();
      const pred = b.temp();
      b.emit(`${one} = icmp ugt i64 ${value.ws[0] ?? "0"}, 0`);
      b.emit(`${pred} = sub i64 ${value.ws[0] ?? "0"}, 1`);
      void one;
      return [{ ws: [pred], lay, type: fieldType }];
    }
    if (builtin && (family.split(".").pop() === "U32" || family.split(".").pop() === "F32")
      && short === family.split(".").pop()) {
      const word = this.nativeCtorFields(ctor, type)[0];
      if (!word) return [];
      if (family.split(".").pop() === "U32") return [{ ws: [value.ws[0] ?? "0"], lay: word.lay, type: word.type }];
      const bits = b.temp();
      b.emit(`${bits} = bitcast float ${value.ws[0] ?? "0.0"} to i32`);
      return [{ ws: [bits], lay: word.lay, type: word.type }];
    }
    if (builtin && lay.kind === "word" && short === "WCon") {
      const width = lay.width ?? 0;
      const fields = this.nativeCtorFields(ctor, type);
      const low = b.temp();
      const bit = b.temp();
      const tail = b.temp();
      b.emit(`${low} = and i32 ${value.ws[0] ?? "0"}, 1`);
      b.emit(`${bit} = trunc i32 ${low} to i1`);
      b.emit(`${tail} = lshr i32 ${value.ws[0] ?? "0"}, 1`);
      return [
        { ws: [bit], lay: fields[0]?.lay ?? { kind: "bool", ks: ["i1"] }, type: fields[0]?.type ?? Bend.ADT("Bool", []) },
        { ws: [tail], lay: fields[1]?.lay ?? { kind: "word", ks: ["i32"], width: width - 1, family }, type: fields[1]?.type ?? type },
      ];
    }
    if (lay.kind === "adt") {
      const arm = lay.arms?.get(ctor);
      if (!arm) return [];
      const fieldInfo = this.nativeCtorFields(ctor, type);
      return arm.fields.map((field, i) => ({
        ws: field.ks.map((kind, j) => this.coerceNative(b,
          value.ws[arm.offsets[i] + j] ?? "0", lay.ks[arm.offsets[i] + j], kind)),
        lay: field,
        type: fieldInfo[i]?.type ?? type,
      }));
    }
    return [];
  }

  private nativeTagTest(b: Builder, ctor: Name, value: NativeVal): string {
    const lay = value.lay;
    const family = lay.family ?? Bend.book_fam(this.book, ctor);
    const builtin = this.book.tlds[family]?.b === true;
    const short = this.nativeCtorShort(ctor);
    const out = b.temp();
    if (builtin && lay.kind === "bool") {
      const isTrue = short === "True";
      b.emit(`${out} = icmp eq i1 ${value.ws[0] ?? "false"}, ${isTrue ? "true" : "false"}`);
      return out;
    }
    if (builtin && lay.kind === "nat") {
      b.emit(`${out} = icmp ${short === "Zero" ? "eq" : "ne"} i64 ${value.ws[0] ?? "0"}, 0`);
      return out;
    }
    if (builtin && lay.kind === "word") {
      const yes = (short === "WNil" && lay.width === 0) || (short === "WCon" && (lay.width ?? 0) > 0);
      b.emit(`${out} = add i1 0, ${yes ? 1 : 0}`);
      return out;
    }
    if (lay.kind !== "adt" || lay.tagIndex === undefined || lay.tagIndex < 0) {
      b.emit(`${out} = add i1 0, 1`);
      return out;
    }
    const tag = lay.arms?.get(ctor)?.tag;
    if (tag === undefined) throw new NativeUnsupported(`unknown native constructor pattern ${ctor}`);
    b.emit(`${out} = icmp eq i32 ${value.ws[lay.tagIndex] ?? "0"}, ${tag}`);
    return out;
  }

  private emitNativeMatApply(b: Builder, matcher: Extract<HTerm, { $: "Mat" }>, args: HTerm[],
    env: Map<number, NativeVal>, expected: NativeLay, expectedType: HTerm): NativeVal {
    if (args.length === 0) throw new NativeUnsupported("native matcher missing scrutinee");
    const scrutineeType = this.inferNativeType(args[0], env, expectedType);
    const scrutineeLay = this.nativeLayout(scrutineeType);
    if (!scrutineeLay) throw new NativeUnsupported("native match scrutinee has no finite layout");
    const scrutinee = this.emitNativeTerm(b, args[0], env, scrutineeLay, scrutineeType);
    const test = this.nativeTagTest(b, matcher.k, scrutinee);
    const hitLabel = b.label("native_match_hit_");
    const missLabel = b.label("native_match_next_");
    const mergeLabel = b.label("native_match_end_");
    b.emit(`br i1 ${test}, label %${hitLabel}, label %${missLabel}`);

    b.start(hitLabel);
    const local = new Map(env);
    const fields = this.nativeProjectFields(b, matcher.k, scrutinee, scrutineeType);
    const fieldTerms = this.nativeCtorFields(matcher.k, scrutineeType);
    let branch: HTerm = matcher.h;
    fields.forEach((field, i) => {
      const id = -700000 - this.fresh++;
      const variable = Bend.Var(`__native_field_${id}`, id) as HTerm;
      local.set(id, field);
      branch = this.nativeApplyTerm(branch, this.nativeAnnotation(variable, (fieldTerms[i]?.type ?? field.type) as HTerm));
    });
    let hit = args.length > 1
      ? this.emitNativeApply(b, branch, args.slice(1), local, expected, expectedType)
      : this.emitNativeTerm(b, branch, local, expected, expectedType);
    const incoming: { pred: string; ws: string[] }[] = [];
    if (!b.terminated) {
      incoming.push({ pred: b.current, ws: hit.ws });
      b.emit(`br label %${mergeLabel}`);
    }

    b.start(missLabel);
    const missTerm = this.stripTerm(matcher.m);
    if (missTerm.$ === "Efq") {
      b.emit("unreachable");
      if (incoming.length === 0) {
        b.start(mergeLabel);
        b.emit("unreachable");
        return { ws: [], lay: expected, type: expectedType };
      }
      b.start(mergeLabel);
      return {
        ws: incoming[0].ws,
        lay: expected,
        type: expectedType,
      };
    }
    let miss: NativeVal;
    if (missTerm.$ === "Mat") {
      const missArgs = [args[0], ...args.slice(1)];
      miss = this.emitNativeMatApply(b, missTerm, missArgs, env, expected, expectedType);
    } else {
      const id = -710000 - this.fresh++;
      const variable = Bend.Var(`__native_match_${id}`, id) as HTerm;
      const fallbackEnv = new Map(env).set(id, scrutinee);
      const fallback = this.nativeApplyTerm(missTerm, this.nativeAnnotation(variable, scrutineeType));
      miss = args.length > 1
        ? this.emitNativeApply(b, fallback, args.slice(1), fallbackEnv, expected, expectedType)
        : this.emitNativeTerm(b, fallback, fallbackEnv, expected, expectedType);
    }
    if (!b.terminated) {
      incoming.push({ pred: b.current, ws: miss.ws });
      b.emit(`br label %${mergeLabel}`);
    }
    b.start(mergeLabel);
    if (incoming.length === 0) {
      b.emit("unreachable");
      return { ws: [], lay: expected, type: expectedType };
    }
    const ws: string[] = [];
    for (let i = 0; i < expected.ks.length; i++) {
      if (incoming.length === 1) {
        ws.push(incoming[0].ws[i] ?? "0");
        continue;
      }
      const value = b.temp();
      const incomingText = incoming.map((item) => `[ ${item.ws[i] ?? "0"}, %${item.pred} ]`).join(", ");
      b.emit(`${value} = phi ${expected.ks[i]} ${incomingText}`);
      ws.push(value);
    }
    return { ws, lay: expected, type: expectedType };
  }

  private emitNativeOperation(b: Builder, k: Name, op: string, args: HTerm[], env: Map<number, NativeVal>,
    expected: NativeLay, expectedType: HTerm): NativeVal {
    const inputs = this.nativeLiveArgs(k, args);
    let values: NativeVal[] = inputs.map(({ term, type }) => {
      const lay = this.nativeLayout(type) ?? this.nativeLayout(this.inferNativeType(term, env, expectedType));
      if (!lay) throw new NativeUnsupported(`native operation ${op} argument has no scalar layout`);
      return this.emitNativeTerm(b, term, env, lay, type);
    });
    let wordWidth: number | null = null;
    if (op.startsWith("word_")) {
      wordWidth = inputs.length > 0 ? this.natConstant(inputs[0].term) : null;
      if (wordWidth === null || wordWidth < 0 || wordWidth > 32) {
        throw new NativeUnsupported(`native Word operation ${op} needs a static width in 0..32`);
      }
      values = values.slice(1);
    }
    const outType = this.nativeCallType(k, args);
    const outLay = this.nativeLayout(outType) ?? expected;
    const x = values[0]?.ws[0] ?? "0";
    const y = values[1]?.ws[0] ?? "0";
    const emit = (instruction: string, type: NativeKind, left: string, right?: string): string => {
      const result = b.temp();
      b.emit(right === undefined ? `${result} = ${instruction} ${type} ${left}` : `${result} = ${instruction} ${type} ${left}, ${right}`);
      return result;
    };
    const maskWord = (raw: string): string => {
      const width = outLay.width ?? wordWidth ?? values[0]?.lay.width ?? 32;
      if (width >= 32) return raw;
      return emit("and", "i32", raw, String((2 ** width) - 1));
    };
    let result: string;
    if (op === "u32_add" || op === "u32_sub" || op === "u32_mul" || op === "u32_and"
      || op === "u32_or" || op === "u32_xor") {
      const inst = op.slice(4);
      result = emit(inst, "i32", x, y);
    } else if (op === "u32_inc") result = emit("add", "i32", x, "1");
    else if (op === "u32_not") result = emit("xor", "i32", x, "-1");
    else if (op === "u32_div" || op === "u32_mod") {
      const zero = emit("icmp eq", "i32", y, "0");
      const safe = this.select(b, zero, "1", y, "i32");
      const raw = emit(op === "u32_div" ? "udiv" : "urem", "i32", x, safe);
      result = this.select(b, zero, op === "u32_div" ? "0" : x, raw, "i32");
    } else if (op === "u32_shl" || op === "u32_shln" || op === "u32_shr" || op === "u32_shrn") {
      const left = op === "u32_shl" || op === "u32_shln";
      const shift = op === "u32_shl" || op === "u32_shr" ? "1" : y;
      const wideShift = (op === "u32_shln" || op === "u32_shrn")
        && values[1]?.lay.kind === "nat";
      const shiftType = wideShift ? "i64" : "i32";
      const big = emit("icmp uge", shiftType, shift, "32");
      let narrowedShift = shift;
      if (wideShift) {
        narrowedShift = b.temp();
        b.emit(`${narrowedShift} = trunc i64 ${shift} to i32`);
      }
      const safe = this.select(b, big, "0", narrowedShift, "i32");
      const raw = emit(left ? "shl" : "lshr", "i32", x, safe);
      result = this.select(b, big, "0", raw, "i32");
    } else if (op === "u32_is_eq" || op === "u32_is_ne" || op === "u32_is_lt" || op === "u32_is_le"
      || op === "u32_is_gt" || op === "u32_is_ge" || op === "nat_is_eq" || op === "nat_is_lt"
      || op === "f32_is_eq" || op === "f32_is_ne" || op === "f32_is_lt" || op === "f32_is_le"
      || op === "f32_is_gt" || op === "f32_is_ge" || op === "bool_and" || op === "bool_or"
      || op === "bool_xor" || op === "bool_not") {
      if (op === "bool_not") result = emit("xor", "i1", x, "true");
      else if (op === "bool_and" || op === "bool_or" || op === "bool_xor") result = emit(op === "bool_and" ? "and" : op === "bool_or" ? "or" : "xor", "i1", x, y);
      else if (op.startsWith("f32_")) {
        const pred = ({ f32_is_eq: "oeq", f32_is_ne: "une", f32_is_lt: "olt", f32_is_le: "ole", f32_is_gt: "ogt", f32_is_ge: "oge" } as Record<string, string>)[op];
        result = emit(`fcmp ${pred}`, "float", x, y);
      } else {
        const pred = ({ u32_is_eq: "eq", u32_is_ne: "ne", u32_is_lt: "ult", u32_is_le: "ule", u32_is_gt: "ugt", u32_is_ge: "uge", nat_is_eq: "eq", nat_is_lt: "ult" } as Record<string, string>)[op];
        const ty = values[0]?.lay.kind === "nat" ? "i64" : "i32";
        result = emit(`icmp ${pred}`, ty, x, y);
      }
    } else if (op === "nat_add" || op === "nat_double" || op === "nat_mul" || op === "nat_sub"
      || op === "nat_min" || op === "nat_max") {
      if (op === "nat_sub" || op === "nat_min" || op === "nat_max") {
        const pred = emit("icmp", "i64", x, y);
        const cmp = b.temp();
        b.lines[b.lines.length - 1] = `  ${cmp} = icmp ${op === "nat_sub" || op === "nat_min" ? "ult" : "ugt"} i64 ${x}, ${y}`;
        if (op === "nat_sub") {
          const sub = emit("sub", "i64", x, y);
          result = this.select(b, cmp, "0", sub, "i64");
        } else result = this.select(b, cmp, x, y, "i64");
      } else if (op === "nat_mul") {
        const zero = emit("icmp eq", "i64", y, "0");
        const safe = this.select(b, zero, "1", y, "i64");
        const limit = emit("udiv", "i64", String((1n << 48n) - 1n), safe);
        const tooLarge = emit("icmp ugt", "i64", x, limit);
        const nonzero = emit("xor", "i1", zero, "true");
        const invalid = emit("and", "i1", tooLarge, nonzero);
        this.guardIf(b, invalid);
        result = emit("mul", "i64", x, y);
      } else {
        result = emit("add", "i64", x, op === "nat_double" ? x : y);
        this.guardNat(b, result);
      }
    } else if (op.startsWith("word_")) {
      if (op === "word_zero") result = "0";
      else if (op === "word_inc") result = maskWord(emit("add", "i32", x, "1"));
      else if (op === "word_add") result = maskWord(emit("add", "i32", x, y));
      else if (op === "word_sub") result = maskWord(emit("sub", "i32", x, y));
      else if (op === "word_and") result = emit("and", "i32", x, y);
      else if (op === "word_or") result = emit("or", "i32", x, y);
      else if (op === "word_xor") result = emit("xor", "i32", x, y);
      else if (op === "word_not") result = maskWord(emit("xor", "i32", x, "-1"));
      else if (op === "word_shl" || op === "word_shr") {
        result = maskWord(emit(op === "word_shl" ? "shl" : "lshr", "i32", x, "1"));
      } else if (op === "word_to_nat") {
        result = b.temp();
        b.emit(`${result} = zext i32 ${x} to i64`);
      }
      else throw new NativeUnsupported(`native operation ${op} is not implemented`);
    } else if (op === "u32_to_nat") {
      result = b.temp();
      b.emit(`${result} = zext i32 ${x} to i64`);
    } else if (op === "nat_to_u32" || op === "u32_from_nat") {
      result = b.temp();
      b.emit(`${result} = trunc i64 ${x} to i32`);
    } else if (op === "u32_to_f32") {
      result = b.temp();
      b.emit(`${result} = uitofp i32 ${x} to float`);
    } else if (op === "f32_to_u32") {
      const ge = emit("fcmp oge", "float", x, "1.0");
      const lt = emit("fcmp olt", "float", x, "0x41F0000000000000");
      const valid = emit("and", "i1", ge, lt);
      const safe = this.select(b, valid, x, "0.0", "float");
      result = b.temp();
      b.emit(`${result} = fptoui float ${safe} to i32`);
    } else if (op === "f32_add" || op === "f32_sub" || op === "f32_mul" || op === "f32_div") {
      result = emit(({ f32_add: "fadd", f32_sub: "fsub", f32_mul: "fmul", f32_div: "fdiv" } as Record<string, string>)[op], "float", x, y);
    } else if (op === "f32_neg") result = emit("fneg", "float", x);
    else if (op === "nat_is_eq") result = emit("icmp eq", "i64", x, y);
    else throw new NativeUnsupported(`native operation ${op} is not implemented`);
    return { ws: [result], lay: outLay, type: outType };
  }

  private coerceNative(b: Builder, value: string, from: NativeKind, to: NativeKind): string {
    if (from === to) return value;
    const result = b.temp();
    if (from === "i1" && (to === "i32" || to === "i64")) b.emit(`${result} = zext ${from} ${value} to ${to}`);
    else if ((from === "i32" || from === "i64") && to === "i1") {
      const wide = b.temp();
      b.emit(`${wide} = icmp ne ${from} ${value}, 0`);
      return wide;
    } else if (from === "i32" && to === "i64") b.emit(`${result} = zext i32 ${value} to i64`);
    else if (from === "i64" && to === "i32") b.emit(`${result} = trunc i64 ${value} to i32`);
    else if (from === "i32" && to === "float") b.emit(`${result} = bitcast i32 ${value} to float`);
    else if (from === "float" && to === "i32") b.emit(`${result} = bitcast float ${value} to i32`);
    else throw new NativeUnsupported(`cannot convert native LLVM ${from} to ${to}`);
    return result;
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
    if (this.needsTrap) {
      declarations.push("declare i64 @write(i32, i8*, i64)", "declare void @exit(i32)");
    }
    for (const name of [...this.usedMath].sort()) {
      declarations.push(`declare double @${name}(${name === "atan2" || name === "pow" || name === "fmod" ? "double, double" : "double"})`);
    }
    for (const [name, arity] of this.usedOps) {
      declarations.push(`declare i64 @${name}(${Array(arity).fill("i64").join(", ")})`);
    }
    const globals = [...this.strings].map(([value, name]) => {
      const bytes = new TextEncoder().encode(value + "\0");
      return `@${name} = private unnamed_addr constant [${bytes.length} x i8] c"${this.llvmBytes(bytes)}", align 1`;
    });
    return [
      "; generated directly from a checked Bend book",
      "; bend-runtime: rust",
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
    const scalar = this.emitScalarOpCall(b, op, args);
    if (scalar !== undefined) return scalar;
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

  private emitScalarOpCall(b: Builder, op: string, args: string[]): string | undefined {
    const arg = (i: number) => args[i] ?? "0";
    const u32 = (i: number) => this.unpackU32(b, arg(i));
    const nat = (i: number) => this.unpackNat(b, arg(i));
    const f32 = (i: number) => this.unpackF32(b, arg(i));
    const outU32 = (raw: string) => this.packU32(b, raw);
    const outNat = (raw: string) => this.packNat(b, raw);
    const outF32 = (raw: string) => this.packF32(b, raw);
    const outBool = (raw: string) => this.packBool(b, raw);
    const bin = (instr: string, left: string, right: string, ty = "i32") => {
      const out = b.temp();
      b.emit(`${out} = ${instr} ${ty} ${left}, ${right}`);
      return out;
    };

    if (op.startsWith("u32_")) {
      const x = u32(0);
      const y = u32(1);
      const binary: Record<string, string> = {
        u32_add: "add", u32_sub: "sub", u32_and: "and", u32_or: "or",
        u32_xor: "xor", u32_mul: "mul",
      };
      if (binary[op]) return outU32(bin(binary[op], x, y));
      if (["u32_is_eq", "u32_is_ne", "u32_is_lt", "u32_is_le", "u32_is_gt", "u32_is_ge"].includes(op)) {
        const pred: Record<string, string> = {
          u32_is_eq: "eq", u32_is_ne: "ne", u32_is_lt: "ult", u32_is_le: "ule",
          u32_is_gt: "ugt", u32_is_ge: "uge",
        };
        const cmp = b.temp();
        b.emit(`${cmp} = icmp ${pred[op]} i32 ${x}, ${y}`);
        return outBool(cmp);
      }
      if (op === "u32_is_zero") {
        const cmp = b.temp();
        b.emit(`${cmp} = icmp eq i32 ${x}, 0`);
        return outBool(cmp);
      }
      if (op === "u32_inc") return outU32(bin("add", x, "1"));
      if (op === "u32_not") return outU32(bin("xor", x, "-1"));
      if (op === "u32_shl" || op === "u32_shln") {
        const shift = op === "u32_shl" ? "1" : nat(1);
        const amount = op === "u32_shl" ? null : b.temp();
        if (amount !== null) b.emit(`${amount} = icmp uge i64 ${shift}, 32`);
        const narrowed = op === "u32_shl" ? shift : b.temp();
        if (op !== "u32_shl") b.emit(`${narrowed} = trunc i64 ${shift} to i32`);
        const safe = op === "u32_shl" ? shift : this.select(b, amount!, "0", narrowed, "i32");
        const shifted = bin("shl", x, safe);
        return outU32(amount === null ? shifted : this.select(b, amount, "0", shifted, "i32"));
      }
      if (op === "u32_shr" || op === "u32_shrn") {
        const shift = op === "u32_shr" ? "1" : nat(1);
        const amount = op === "u32_shr" ? null : b.temp();
        if (amount !== null) b.emit(`${amount} = icmp uge i64 ${shift}, 32`);
        const narrowed = op === "u32_shr" ? shift : b.temp();
        if (op !== "u32_shr") b.emit(`${narrowed} = trunc i64 ${shift} to i32`);
        const safe = op === "u32_shr" ? shift : this.select(b, amount!, "0", narrowed, "i32");
        const shifted = bin("lshr", x, safe);
        return outU32(amount === null ? shifted : this.select(b, amount, "0", shifted, "i32"));
      }
      if (op === "u32_div" || op === "u32_mod") {
        const zero = b.temp();
        b.emit(`${zero} = icmp eq i32 ${y}, 0`);
        const safe = this.select(b, zero, "1", y, "i32");
        const result = bin(op === "u32_div" ? "udiv" : "urem", x, safe);
        return outU32(this.select(b, zero, op === "u32_div" ? "0" : x, result, "i32"));
      }
      if (op === "u32_to_f32") {
        const cast = b.temp();
        b.emit(`${cast} = uitofp i32 ${x} to float`);
        return outF32(cast);
      }
      if (op === "u32_to_nat") {
        const cast = b.temp();
        b.emit(`${cast} = zext i32 ${x} to i64`);
        return outNat(cast);
      }
      if (op === "u32_from_nat") {
        const wide = nat(0);
        const cast = b.temp();
        b.emit(`${cast} = trunc i64 ${wide} to i32`);
        return outU32(cast);
      }
      return undefined;
    }

    if (op.startsWith("f32_")) {
      const x = f32(0);
      const y = f32(1);
      const binary: Record<string, string> = {
        f32_add: "fadd", f32_sub: "fsub", f32_mul: "fmul", f32_div: "fdiv",
      };
      if (binary[op]) return outF32(bin(binary[op], x, y, "float"));
      if (op === "f32_neg") {
        const out = b.temp();
        b.emit(`${out} = fneg float ${x}`);
        return outF32(out);
      }
      const pred: Record<string, string> = {
        f32_is_eq: "oeq", f32_is_ne: "une", f32_is_lt: "olt",
        f32_is_le: "ole", f32_is_gt: "ogt", f32_is_ge: "oge",
      };
      if (pred[op]) {
        const cmp = b.temp();
        b.emit(`${cmp} = fcmp ${pred[op]} float ${x}, ${y}`);
        return outBool(cmp);
      }
      if (op === "f32_bits") {
        const bits = b.temp();
        b.emit(`${bits} = bitcast float ${x} to i32`);
        return outU32(bits);
      }
      if (op === "f32_to_u32") {
        const ge = b.temp();
        const lt = b.temp();
        const valid = b.temp();
        b.emit(`${ge} = fcmp oge float ${x}, 1.0`);
        b.emit(`${lt} = fcmp olt float ${x}, 0x41F0000000000000`);
        b.emit(`${valid} = and i1 ${ge}, ${lt}`);
        const safe = this.select(b, valid, x, "0.0", "float");
        const truncated = b.temp();
        b.emit(`${truncated} = fptoui float ${safe} to i32`);
        return outU32(truncated);
      }
      const mathName: Record<string, string> = {
        f32_sqrt: "sqrt", f32_exp: "exp", f32_log: "log", f32_log2: "log2",
        f32_log10: "log10", f32_sin: "sin", f32_cos: "cos", f32_tan: "tan",
        f32_asin: "asin", f32_acos: "acos", f32_atan: "atan", f32_sinh: "sinh",
        f32_cosh: "cosh", f32_tanh: "tanh", f32_floor: "floor", f32_ceil: "ceil",
        f32_trunc: "trunc", f32_abs: "fabs", f32_atan2: "atan2", f32_pow: "pow",
        f32_mod: "fmod",
      };
      const fn = mathName[op];
      if (fn) {
        const dx = b.temp();
        b.emit(`${dx} = fpext float ${x} to double`);
        let callArgs = `double ${dx}`;
        if (["atan2", "pow", "fmod"].includes(fn)) {
          const dy = b.temp();
          b.emit(`${dy} = fpext float ${y} to double`);
          callArgs += `, double ${dy}`;
        }
        this.usedMath.add(fn);
        const result = b.temp();
        b.emit(`${result} = call double @${fn}(${callArgs})`);
        const narrowed = b.temp();
        b.emit(`${narrowed} = fptrunc double ${result} to float`);
        return outF32(narrowed);
      }
      return undefined;
    }

    if (op.startsWith("nat_") && !["nat_cmp", "nat_divmod"].includes(op)) {
      const x = nat(0);
      const y = nat(1);
      if (op === "nat_add" || op === "nat_double") {
        const sum = bin("add", x, op === "nat_double" ? x : y, "i64");
        this.guardNat(b, sum);
        return outNat(sum);
      }
      if (op === "nat_mul") {
        const zero = b.temp();
        b.emit(`${zero} = icmp eq i64 ${y}, 0`);
        const safe = this.select(b, zero, "1", y, "i64");
        const quotient = bin("udiv", String((1n << 48n) - 1n), safe, "i64");
        const tooLarge = b.temp();
        const nonzero = b.temp();
        const invalid = b.temp();
        b.emit(`${tooLarge} = icmp ugt i64 ${x}, ${quotient}`);
        b.emit(`${nonzero} = xor i1 ${zero}, true`);
        b.emit(`${invalid} = and i1 ${tooLarge}, ${nonzero}`);
        this.guardIf(b, invalid);
        return outNat(bin("mul", x, y, "i64"));
      }
      if (op === "nat_sub") {
        const under = b.temp();
        b.emit(`${under} = icmp ult i64 ${x}, ${y}`);
        return outNat(this.select(b, under, "0", bin("sub", x, y, "i64"), "i64"));
      }
      if (op === "nat_is_lt") {
        const cmp = b.temp();
        b.emit(`${cmp} = icmp ult i64 ${x}, ${y}`);
        return outBool(cmp);
      }
      if (op === "nat_is_eq") {
        const cmp = b.temp();
        b.emit(`${cmp} = icmp eq i64 ${x}, ${y}`);
        return outBool(cmp);
      }
      if (op === "nat_min" || op === "nat_max") {
        const cmp = b.temp();
        b.emit(`${cmp} = icmp ${op === "nat_min" ? "ule" : "uge"} i64 ${x}, ${y}`);
        return outNat(this.select(b, cmp, x, y, "i64"));
      }
      return undefined;
    }

    if (op === "nat_cmp") {
      const x = nat(0);
      const y = nat(1);
      const lt = b.temp();
      const eq = b.temp();
      const notLt = b.temp();
      const pick = b.temp();
      const tag = b.temp();
      b.emit(`${lt} = icmp ult i64 ${x}, ${y}`);
      b.emit(`${eq} = icmp eq i64 ${x}, ${y}`);
      b.emit(`${notLt} = select i1 ${eq}, i64 ${this.eqTag}, i64 ${this.gtTag}`);
      b.emit(`${tag} = select i1 ${lt}, i64 ${this.ltTag}, i64 ${notLt}`);
      return this.callVector(b, "bend_ctor", [tag, "0"], []);
    }

    if (op === "nat_divmod") {
      const x = nat(0);
      const y = nat(1);
      const zero = b.temp();
      const safe = b.temp();
      const quotient = b.temp();
      const remainder = b.temp();
      const q = b.temp();
      const r = b.temp();
      b.emit(`${zero} = icmp eq i64 ${y}, 0`);
      b.emit(`${safe} = select i1 ${zero}, i64 1, i64 ${y}`);
      b.emit(`${quotient} = udiv i64 ${x}, ${safe}`);
      b.emit(`${remainder} = urem i64 ${x}, ${safe}`);
      b.emit(`${q} = select i1 ${zero}, i64 0, i64 ${quotient}`);
      b.emit(`${r} = select i1 ${zero}, i64 ${x}, i64 ${remainder}`);
      return this.callVector(b, "bend_ctor", [String(this.tupleTag), "2"], [outNat(q), outNat(r)]);
    }

    if (op === "bool_and" || op === "bool_or" || op === "bool_xor" || op === "bool_not") {
      const x = this.unpackBool(b, arg(0));
      const result = b.temp();
      if (op === "bool_not") {
        b.emit(`${result} = xor i1 ${x}, true`);
      } else {
        const y = this.unpackBool(b, arg(1));
        b.emit(`${result} = ${op === "bool_and" ? "and" : op === "bool_or" ? "or" : "xor"} i1 ${x}, ${y}`);
      }
      return outBool(result);
    }
    return undefined;
  }

  private unpackU32(b: Builder, value: string): string {
    const low = b.temp();
    const out = b.temp();
    b.emit(`${low} = and i64 ${value}, 4294967295`);
    b.emit(`${out} = trunc i64 ${low} to i32`);
    return out;
  }

  private unpackNat(b: Builder, value: string): string {
    const out = b.temp();
    b.emit(`${out} = and i64 ${value}, ${(1n << 48n) - 1n}`);
    return out;
  }

  private unpackF32(b: Builder, value: string): string {
    const bits = b.temp();
    const out = b.temp();
    b.emit(`${bits} = trunc i64 ${value} to i32`);
    b.emit(`${out} = bitcast i32 ${bits} to float`);
    return out;
  }

  private unpackBool(b: Builder, value: string): string {
    const bit = b.temp();
    const out = b.temp();
    b.emit(`${bit} = and i64 ${value}, 1`);
    b.emit(`${out} = trunc i64 ${bit} to i1`);
    return out;
  }

  private packU32(b: Builder, value: string): string {
    return this.packTagged32(b, value, 0x1001n);
  }

  private packF32(b: Builder, value: string): string {
    const bits = b.temp();
    b.emit(`${bits} = bitcast float ${value} to i32`);
    return this.packTagged32(b, bits, 0x1002n);
  }

  private packNat(b: Builder, value: string): string {
    const payload = b.temp();
    const tagged = b.temp();
    b.emit(`${payload} = and i64 ${value}, ${(1n << 48n) - 1n}`);
    b.emit(`${tagged} = or i64 ${payload}, ${0x1003n << 48n}`);
    return tagged;
  }

  private packNatConst(b: Builder, value: string): string {
    const out = b.temp();
    b.emit(`${out} = or i64 ${value}, ${0x1003n << 48n}`);
    return out;
  }

  private packBool(b: Builder, value: string): string {
    const payload = b.temp();
    const tagged = b.temp();
    b.emit(`${payload} = zext i1 ${value} to i64`);
    b.emit(`${tagged} = or i64 ${payload}, ${0x1005n << 48n}`);
    return tagged;
  }

  private packBoolConst(b: Builder, value: boolean): string {
    const out = b.temp();
    b.emit(`${out} = or i64 ${value ? 1 : 0}, ${0x1005n << 48n}`);
    return out;
  }

  private packU32Const(b: Builder, value: string): string {
    const out = b.temp();
    b.emit(`${out} = or i64 ${value}, ${0x1001n << 48n}`);
    return out;
  }

  private emitWordBits(b: Builder, word: string): string {
    let current = word;
    let result = "0";
    const wcon = this.tagSuffix("WCon");
    for (let i = 0; i < 32; i++) {
      const tail = b.temp();
      const bit = b.temp();
      const low = b.temp();
      const narrowed = b.temp();
      const shifted = b.temp();
      const combined = b.temp();
      const tag = b.temp();
      b.emit(`${tag} = call i64 @bend_tag(i64 ${current})`);
      b.emit(`${bit} = call i64 @bend_field(i64 ${current}, i64 0)`);
      b.emit(`${tail} = call i64 @bend_field(i64 ${current}, i64 1)`);
      b.emit(`${low} = and i64 ${bit}, 1`);
      b.emit(`${narrowed} = trunc i64 ${low} to i32`);
      if (i === 0) {
        b.emit(`${shifted} = shl i32 ${narrowed}, 0`);
      } else {
        b.emit(`${shifted} = shl i32 ${narrowed}, ${i}`);
      }
      b.emit(`${combined} = or i32 ${result}, ${shifted}`);
      result = combined;
      current = tail;
      // The type checker guarantees the length. Keeping this cheap tag read
      // also makes a malformed foreign value fail through the normal helper.
      void wcon;
    }
    return result;
  }

  private packTagged32(b: Builder, value: string, tag: bigint): string {
    const payload = b.temp();
    const tagged = b.temp();
    b.emit(`${payload} = zext i32 ${value} to i64`);
    b.emit(`${tagged} = or i64 ${payload}, ${tag << 48n}`);
    return tagged;
  }

  private select(b: Builder, cond: string, yes: string, no: string, type: string): string {
    const out = b.temp();
    b.emit(`${out} = select i1 ${cond}, ${type} ${yes}, ${type} ${no}`);
    return out;
  }

  private guardNat(b: Builder, value: string): void {
    const tooLarge = b.temp();
    b.emit(`${tooLarge} = icmp ugt i64 ${value}, ${(1n << 48n) - 1n}`);
    this.guardIf(b, tooLarge);
  }

  private guardIf(b: Builder, cond: string): void {
    const failed = b.label("nat_overflow_");
    const good = b.label("nat_ok_");
    b.emit(`br i1 ${cond}, label %${failed}, label %${good}`);
    b.start(failed);
    const message = "bend: a Nat past the largest immediate 2^48-1\n";
    const pointer = this.stringPointer(message, b.lines);
    const length = new TextEncoder().encode(message).length;
    const written = b.temp();
    b.emit(`${written} = call i64 @write(i32 2, i8* ${pointer}, i64 ${length})`);
    b.emit("call void @exit(i32 1)");
    b.emit("unreachable");
    b.start(good);
    this.needsTrap = true;
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
        const [head, args] = Bend.term_unapply(term);
        let rawHead: Term = term;
        const rawArgs: Term[] = [];
        for (;;) {
          const part = Bend.term_force(rawHead);
          if (part.$ === "Ann") rawHead = part.x as Term;
          else if (part.$ === "App") {
            rawArgs.unshift(part.x as Term);
            rawHead = part.f as Term;
          } else break;
        }
        const direct = this.stripTerm(Bend.term_higher(rawHead));
        const directArgs = rawArgs.map((arg) => Bend.term_higher(arg));
        if (direct.$ === "Ref" && directArgs.length > 0) {
          const tld = this.book.tlds[direct.k];
          if (tld?.$ === "Def" && !this.isOperation(direct.k) && !this.isIOHandler(direct.k)) {
            const spec = this.makeNativeSpec(direct.k, directArgs);
            if (spec) {
              if (!this.nativeSpecs.has(spec.key)) this.nativeSpecs.set(spec.key, spec);
              const known = this.nativeSpecs.get(spec.key)!;
              const nativeReady = this.ensureNativeSpec(known);
              if (known.state !== "compiling") this.pruneNativeSpecs();
              if (nativeReady && known.state === "done") {
                return this.emitNativeBoundaryCall(b, known, rawArgs, env);
              }
            }
          }
        }
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
    const builtin = this.book.tlds[family]?.b === true;
    const fam = family.split(".").pop();
    const short = k.split(".").pop();
    if (builtin && fam === "Nat" && (short === "Zero" || short === "Succ")) {
      if (short === "Zero") return this.packNatConst(b, "0");
      const value = this.unpackNat(b, fields[0] ?? "0");
      const next = b.temp();
      b.emit(`${next} = add i64 ${value}, 1`);
      this.guardNat(b, next);
      return this.packNat(b, next);
    }
    if (builtin && fam === "Bool" && (short === "True" || short === "False")) {
      return this.packBoolConst(b, short === "True");
    }
    if (builtin && (fam === "U32" || fam === "F32") && short === fam) {
      const bits = this.emitWordBits(b, fields[0] ?? "0");
      if (fam === "U32") return this.packU32(b, bits);
      const float = b.temp();
      b.emit(`${float} = bitcast i32 ${bits} to float`);
      return this.packF32(b, float);
    }
    if (builtin && fam === "Char" && short === "Chr") {
      const code = this.unpackU32(b, fields[0] ?? "0");
      const wide = b.temp();
      const tagged = b.temp();
      b.emit(`${wide} = zext i32 ${code} to i64`);
      b.emit(`${tagged} = or i64 ${wide}, ${0x1004n << 48n}`);
      return tagged;
    }
    const builder = builtin ? "bend_ctor" : "bend_plain_ctor";
    return this.callVector(b, builder, [String(tag), String(fields.length)], fields);
  }

  private emitLiteral(b: Builder, term: Extract<Term, { $: "Lit" }>): string {
    if (term.k === "Nat") {
      const value = BigInt(String(term.v));
      if (value < 0n || value >= (1n << 48n)) {
        throw new Error(`LLVM backend: Nat literal ${value} exceeds immediate range 0..${(1n << 48n) - 1n}`);
      }
      return this.packNatConst(b, String(value));
    } else if (term.k === "U32") {
      return this.packU32Const(b, String(Number(term.v) >>> 0));
    } else if (term.k === "F32") {
      const word = Number(term.v) >>> 0;
      const float = b.temp();
      b.emit(`${float} = bitcast i32 ${word} to float`);
      return this.packF32(b, float);
    }
    const out = b.temp();
    const text = String(term.v);
    const pointer = this.stringPointer(text, b.lines);
    const bytes = new TextEncoder().encode(text);
    b.emit(`${out} = call i64 @bend_string(i8* ${pointer}, i64 ${bytes.length})`);
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

  private emitNativeBoundaryCall(b: Builder, spec: NativeSpec, args: Term[], env: Map<number, string>): string {
    const callArgs: string[] = [];
    for (let i = 0; i < spec.doms.length; i++) {
      const dom = spec.doms[i];
      if (dom.fixed !== undefined || dom.q.$ === "None") continue;
      const boxed = this.emitTerm(b, args[i], env);
      if (!dom.lay || dom.lay.ks.length === 0) continue;
      const lay = dom.lay;
      const raw = this.unboxNativeBoundary(b, lay, boxed, dom.A);
      raw.forEach((value, j) => callArgs.push(`${lay.ks[j]} ${value}`));
    }
    const call = b.temp();
    const resultType = this.nativeReturnType(spec.retLay);
    b.emit(`${call} = call ${resultType} @${spec.name}(${callArgs.join(", ")})`);
    const raw = this.nativeCallResult(b, spec.retLay, call);
    return this.packNativeBoundary(b, spec.retLay, raw, spec.ret);
  }

  private unboxNativeBoundary(b: Builder, lay: NativeLay, boxed: string, type: HTerm): string[] {
    if (lay.kind === "u32") return [this.unpackU32(b, boxed)];
    if (lay.kind === "nat") return [this.unpackNat(b, boxed)];
    if (lay.kind === "f32") return [this.unpackF32(b, boxed)];
    if (lay.kind === "bool") return [this.unpackBool(b, boxed)];
    if (lay.kind === "word") {
      const raw = this.unpackU32(b, boxed);
      if ((lay.width ?? 33) > 32) throw new NativeUnsupported("native Word boundary exceeds 32 bits");
      return [raw];
    }
    if (lay.kind !== "adt") throw new NativeUnsupported("unsupported native argument boundary");
    const arms = [...(lay.arms?.entries() ?? [])];
    if (arms.length === 0) return [];
    if (arms.length === 1) return this.unboxNativeArm(b, lay, arms[0][0], arms[0][1], boxed, type);
    const tag = b.temp();
    b.emit(`${tag} = call i64 @bend_tag(i64 ${boxed})`);
    const merge = b.label("native_unbox_end_");
    const phis = lay.ks.map((kind) => ({ kind, name: b.temp(), incoming: [] as string[] }));
    let next = b.current;
    for (let i = 0; i < arms.length; i++) {
      const [ctor, arm] = arms[i];
      const hit = b.label("native_unbox_arm_");
      const miss = i + 1 === arms.length ? b.label("native_unbox_bad_") : b.label("native_unbox_next_");
      const runtimeTag = this.ctorTags.get(ctor);
      if (runtimeTag === undefined) throw new NativeUnsupported(`missing runtime tag for ${ctor}`);
      const equal = b.temp();
      b.emit(`${equal} = icmp eq i64 ${tag}, ${runtimeTag}`);
      b.emit(`br i1 ${equal}, label %${hit}, label %${miss}`);
      b.start(hit);
      const values = this.unboxNativeArm(b, lay, ctor, arm, boxed, type);
      const pred = b.current;
      if (!b.terminated) b.emit(`br label %${merge}`);
      values.forEach((value, j) => phis[j].incoming.push(`[ ${value}, %${pred} ]`));
      if (i + 1 === arms.length) {
        b.start(miss);
        b.emit("unreachable");
      } else {
        b.start(miss);
      }
      next = miss;
    }
    void next;
    b.start(merge);
    phis.forEach((phi) => b.emit(`${phi.name} = phi ${phi.kind} ${phi.incoming.join(", ")}`));
    return phis.map((phi) => phi.name);
  }

  private unboxNativeArm(b: Builder, lay: NativeLay, ctor: Name, arm: NativeArm, boxed: string,
    type: HTerm): string[] {
    const values: string[] = Array.from({ length: lay.ks.length }, (_, i) => lay.ks[i] === "float" ? "0.0" : "0");
    if (lay.tagIndex !== undefined && lay.tagIndex >= 0) values[lay.tagIndex] = String(arm.tag);
    const info = this.nativeCtorFields(ctor, type);
    for (let i = 0; i < arm.fields.length; i++) {
      const fieldBox = b.temp();
      b.emit(`${fieldBox} = call i64 @bend_field(i64 ${boxed}, i64 ${i})`);
      const field = this.unboxNativeBoundary(b, arm.fields[i], fieldBox, info[i]?.type ?? type);
      for (let j = 0; j < field.length; j++) {
        const offset = arm.offsets[i] + j;
        values[offset] = this.coerceNative(b, field[j], arm.fields[i].ks[j], lay.ks[offset]);
      }
    }
    return values;
  }

  private packNativeBoundary(b: Builder, lay: NativeLay, raw: string[], type: HTerm): string {
    if (lay.kind === "u32") return this.packU32(b, raw[0] ?? "0");
    if (lay.kind === "nat") return this.packNat(b, raw[0] ?? "0");
    if (lay.kind === "f32") return this.packF32(b, raw[0] ?? "0.0");
    if (lay.kind === "bool") return this.packBool(b, raw[0] ?? "false");
    if (lay.kind === "word") {
      const width = lay.width ?? 33;
      if (width > 32) throw new NativeUnsupported("native Word boundary exceeds 32 bits");
      const low = b.temp();
      const payload = b.temp();
      const tagged = b.temp();
      b.emit(`${low} = zext i32 ${raw[0] ?? "0"} to i64`);
      b.emit(`${payload} = or i64 ${low}, ${BigInt(width) << 32n}`);
      b.emit(`${tagged} = or i64 ${payload}, ${0x1006n << 48n}`);
      return tagged;
    }
    if (lay.kind !== "adt") throw new NativeUnsupported("unsupported native return boundary");
    const arms = [...(lay.arms?.entries() ?? [])];
    if (arms.length === 0) return "0";
    const make = (ctor: Name, arm: NativeArm): string => {
      const fields: string[] = [];
      const runtimeCtor = this.book.ctrs[ctor];
      if (!runtimeCtor) throw new NativeUnsupported(`missing constructor metadata for ${ctor}`);
      const quantities = this.liveCtrQuantities(ctor, runtimeCtor.n);
      let fieldAt = 0;
      for (let i = 0; i < runtimeCtor.n; i++) {
        if (quantities[i]?.$ === "None") continue;
        const fieldLay = arm.fields[fieldAt];
        const offset = arm.offsets[fieldAt];
        const fieldRaw = raw.slice(offset, offset + fieldLay.ks.length).map((value, j) =>
          this.coerceNative(b, value, lay.ks[offset + j], fieldLay.ks[j]));
        fields.push(this.packNativeBoundary(b, fieldLay, fieldRaw, this.nativeCtorFields(ctor, type)[fieldAt]?.type ?? type));
        fieldAt++;
      }
      const builder = this.book.tlds[lay.family ?? ""]?.b === true ? "bend_ctor" : "bend_plain_ctor";
      const tag = this.ctorTags.get(ctor);
      if (tag === undefined) throw new NativeUnsupported(`missing runtime tag for ${ctor}`);
      return this.callVector(b, builder, [String(tag), String(fields.length)], fields);
    };
    if (arms.length === 1) return make(arms[0][0], arms[0][1]);
    const tag = raw[lay.tagIndex ?? 0] ?? "0";
    const merge = b.label("native_pack_end_");
    const incoming: string[] = [];
    for (let i = 0; i < arms.length; i++) {
      const [ctor, arm] = arms[i];
      const hit = b.label("native_pack_arm_");
      const miss = i + 1 === arms.length ? b.label("native_pack_bad_") : b.label("native_pack_next_");
      const equal = b.temp();
      b.emit(`${equal} = icmp eq i32 ${tag}, ${arm.tag}`);
      b.emit(`br i1 ${equal}, label %${hit}, label %${miss}`);
      b.start(hit);
      const value = make(ctor, arm);
      const pred = b.current;
      b.emit(`br label %${merge}`);
      incoming.push(`[ ${value}, %${pred} ]`);
      b.start(miss);
    }
    b.emit("unreachable");
    b.start(merge);
    const result = b.temp();
    b.emit(`${result} = phi i64 ${incoming.join(", ")}`);
    return result;
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

// The standalone printer uses the same shortest round-trip decimal rule as
// comp.ts's f32_text, then adds the literal's .0 before an exponent when needed.
// Its caller declares printf and prints the enclosing value's final newline.
export function native_f32_print_support(): string {
  return String.raw`
@bend_f32_scientific = private unnamed_addr constant [5 x i8] c"%.*e\00"
@bend_f32_fixed = private unnamed_addr constant [5 x i8] c"%.*f\00"
@bend_f32_exponent = private unnamed_addr constant [6 x i8] c"e%c%d\00"
@bend_f32_nan = private unnamed_addr constant [4 x i8] c"nan\00"
@bend_f32_contains = private unnamed_addr constant [4 x i8] c".ni\00"
@bend_f32_dot = private unnamed_addr constant [9 x i8] c"%.*s.0%s\00"
@bend_f32_text = private unnamed_addr constant [3 x i8] c"%s\00"
@bend_f32_e = private unnamed_addr constant [2 x i8] c"e\00"

declare i32 @snprintf(i8*, i64, i8*, ...)
declare i32 @sprintf(i8*, i8*, ...)
declare float @strtof(i8*, i8**)
declare i8* @strchr(i8*, i32)
declare i32 @atoi(i8*)
declare i8* @memmove(i8*, i8*, i64)
declare i8* @memset(i8*, i32, i64)
declare i8* @strpbrk(i8*, i8*)
declare i64 @strcspn(i8*, i8*)

define void @bend_native_print_f32(float %value) {
entry:
  %storage = alloca [40 x i8], align 1
  %buf = getelementptr inbounds [40 x i8], [40 x i8]* %storage, i64 0, i64 0
  %wide = fpext float %value to double
  %isnan = fcmp uno float %value, %value
  br i1 %isnan, label %nan, label %precision
nan:
  %nanprint = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @bend_f32_text, i64 0, i64 0), i8* getelementptr inbounds ([4 x i8], [4 x i8]* @bend_f32_nan, i64 0, i64 0))
  ret void
precision:
  %p = phi i32 [ 0, %entry ], [ %next, %more ]
  %written = call i32 (i8*, i64, i8*, ...) @snprintf(i8* %buf, i64 40, i8* getelementptr inbounds ([5 x i8], [5 x i8]* @bend_f32_scientific, i64 0, i64 0), i32 %p, double %wide)
  %parsed = call float @strtof(i8* %buf, i8** null)
  %same = fcmp oeq float %parsed, %value
  %last = icmp eq i32 %p, 8
  %done = or i1 %same, %last
  br i1 %done, label %format, label %more
more:
  %next = add i32 %p, 1
  br label %precision
format:
  %ep = call i8* @strchr(i8* %buf, i32 101)
  %noexp = icmp eq i8* %ep, null
  br i1 %noexp, label %ready, label %exponent
exponent:
  %digits = getelementptr inbounds i8, i8* %ep, i64 1
  %ex = call i32 @atoi(i8* %digits)
  %large = icmp sge i32 %ex, 21
  %small = icmp sle i32 %ex, -7
  %scientific = or i1 %large, %small
  br i1 %scientific, label %scientific_format, label %ordinary_format
scientific_format:
  %negative = icmp slt i32 %ex, 0
  %negex = sub i32 0, %ex
  %abs = select i1 %negative, i32 %negex, i32 %ex
  %sign = select i1 %negative, i32 45, i32 43
  %expprint = call i32 (i8*, i8*, ...) @sprintf(i8* %ep, i8* getelementptr inbounds ([6 x i8], [6 x i8]* @bend_f32_exponent, i64 0, i64 0), i32 %sign, i32 %abs)
  br label %ready
ordinary_format:
  %fraction = icmp sle i32 %ex, %p
  br i1 %fraction, label %fixed_format, label %integer_format
fixed_format:
  %places = sub i32 %p, %ex
  %fixedprint = call i32 (i8*, i64, i8*, ...) @snprintf(i8* %buf, i64 40, i8* getelementptr inbounds ([5 x i8], [5 x i8]* @bend_f32_fixed, i64 0, i64 0), i32 %places, double %wide)
  br label %ready
integer_format:
  %first = load i8, i8* %buf, align 1
  %minus = icmp eq i8 %first, 45
  %signed = zext i1 %minus to i64
  %destindex = add i64 %signed, 1
  %sourceindex = add i64 %signed, 2
  %dest = getelementptr inbounds i8, i8* %buf, i64 %destindex
  %source = getelementptr inbounds i8, i8* %buf, i64 %sourceindex
  %count = zext i32 %p to i64
  %moved = call i8* @memmove(i8* %dest, i8* %source, i64 %count)
  %zeroindex = add i64 %destindex, %count
  %zerostart = getelementptr inbounds i8, i8* %buf, i64 %zeroindex
  %zeros = sub i32 %ex, %p
  %zerocount = zext i32 %zeros to i64
  %filled = call i8* @memset(i8* %zerostart, i32 48, i64 %zerocount)
  %exwide = zext i32 %ex to i64
  %endindex = add i64 %destindex, %exwide
  %end = getelementptr inbounds i8, i8* %buf, i64 %endindex
  store i8 0, i8* %end, align 1
  br label %ready
ready:
  %present = call i8* @strpbrk(i8* %buf, i8* getelementptr inbounds ([4 x i8], [4 x i8]* @bend_f32_contains, i64 0, i64 0))
  %needsdot = icmp eq i8* %present, null
  br i1 %needsdot, label %dot, label %text
dot:
  %prefix = call i64 @strcspn(i8* %buf, i8* getelementptr inbounds ([2 x i8], [2 x i8]* @bend_f32_e, i64 0, i64 0))
  %length = trunc i64 %prefix to i32
  %tail = getelementptr inbounds i8, i8* %buf, i64 %prefix
  %dotprint = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([9 x i8], [9 x i8]* @bend_f32_dot, i64 0, i64 0), i32 %length, i8* %buf, i8* %tail)
  ret void
text:
  %textprint = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @bend_f32_text, i64 0, i64 0), i8* %buf)
  ret void
}
`.trim();
}
