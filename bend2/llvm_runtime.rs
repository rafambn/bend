//! Small native runtime for the direct LLVM backend.
//!
//! Values crossing the LLVM ABI are opaque `i64` handles. Zero is erased unit;
//! all other handles point at a Rust `Object`. Generated code only calls this
//! file through the exported C ABI below.

#![allow(clippy::missing_safety_doc)]

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::ffi::{c_char, CStr};
use std::io::{self, Read, Write};
use std::mem;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock, RwLock};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

type Value = u64;
type BendFn = unsafe extern "C" fn(Value, Value) -> Value;

struct Ctor {
    tag: Value,
    fields: Vec<Value>,
}

struct Closure {
    call: BendFn,
    captures: Vec<Value>,
}

struct Array {
    values: Mutex<Vec<ArrayCell>>,
}

#[derive(Clone)]
struct ArrayCell {
    value: Value,
    _hold: Option<Arc<RuntimeObject>>,
}

struct TaskState {
    join: Mutex<Option<JoinHandle<TaskResult>>>,
    result: Mutex<Option<TaskResult>>,
    ready: Condvar,
}

struct TaskResult {
    value: Value,
    hold: Option<Arc<RuntimeObject>>,
    joined: bool,
}

enum Object {
    Nat(u64),
    U32(u32),
    F32(f32),
    Char(char),
    String(String),
    Ctor(Ctor),
    Array(Array),
    Closure(Closure),
    TailCall {
        fun: Value,
        arg: Value,
    },
    Task(Arc<TaskState>),
    IoAction {
        opcode: u64,
        arg0: Value,
        arg1: Value,
    },
}

struct RuntimeObject {
    object: Object,
    _edges: Vec<Arc<RuntimeObject>>,
}

#[derive(Default)]
struct Region {
    objects: Vec<Arc<RuntimeObject>>,
    owned: HashSet<usize>,
}

thread_local! {
    static REGIONS: RefCell<Vec<Region>> = const { RefCell::new(Vec::new()) };
}

static ROOT_OBJECTS: OnceLock<Mutex<Vec<Arc<RuntimeObject>>>> = OnceLock::new();
static CTORS: OnceLock<RwLock<HashMap<Value, String>>> = OnceLock::new();
static ARGUMENTS: OnceLock<Mutex<Vec<String>>> = OnceLock::new();
static THREAD_LIMIT: AtomicUsize = AtomicUsize::new(1);
static ACTIVE_THREADS: AtomicUsize = AtomicUsize::new(1);
static STARTED: OnceLock<Instant> = OnceLock::new();

fn roots() -> &'static Mutex<Vec<Arc<RuntimeObject>>> {
    ROOT_OBJECTS.get_or_init(|| Mutex::new(Vec::new()))
}

fn ctor_names() -> &'static RwLock<HashMap<Value, String>> {
    CTORS.get_or_init(|| RwLock::new(HashMap::new()))
}

fn argv_store() -> &'static Mutex<Vec<String>> {
    ARGUMENTS.get_or_init(|| Mutex::new(Vec::new()))
}

fn alloc(object: Object) -> Value {
    let edges = object_edges(&object)
        .into_iter()
        .filter_map(retained_object)
        .collect();
    let runtime_object = Arc::new(RuntimeObject {
        object,
        _edges: edges,
    });
    let raw = Arc::as_ptr(&runtime_object) as usize;
    REGIONS.with(|regions| {
        if let Some(region) = regions.borrow_mut().last_mut() {
            region.objects.push(runtime_object);
            region.owned.insert(raw);
        } else {
            roots()
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push(runtime_object);
        }
    });
    raw as Value
}

fn is_handle(value: Value) -> bool {
    value > 4096 && value as usize % std::mem::align_of::<RuntimeObject>() == 0
}

unsafe fn object<'a>(value: Value) -> Option<&'a Object> {
    is_handle(value).then(|| &(*(value as *const RuntimeObject)).object)
}

fn object_raw(object: &Arc<RuntimeObject>) -> usize {
    Arc::as_ptr(object) as usize
}

fn retained_object(value: Value) -> Option<Arc<RuntimeObject>> {
    if !is_handle(value) {
        return None;
    }
    unsafe {
        let raw = value as *const RuntimeObject;
        Arc::increment_strong_count(raw);
        Some(Arc::from_raw(raw))
    }
}

fn object_edges(object: &Object) -> Vec<Value> {
    match object {
        Object::Ctor(ctor) => ctor.fields.clone(),
        Object::Closure(closure) => closure.captures.clone(),
        Object::TailCall { fun, arg } => vec![*fun, *arg],
        Object::IoAction { arg0, arg1, .. } => vec![*arg0, *arg1],
        _ => Vec::new(),
    }
}

fn retain_in_current_region(value: Value) {
    if value == 0 {
        return;
    }
    let raw = value as usize;
    let owned_here = REGIONS.with(|regions| {
        regions
            .borrow()
            .last()
            .is_some_and(|region| region.owned.contains(&raw))
    });
    if owned_here {
        return;
    }
    let Some(object) = retained_object(value) else {
        return;
    };
    REGIONS.with(|regions| {
        if let Some(region) = regions.borrow_mut().last_mut() {
            if region.owned.insert(raw) {
                region.objects.push(object);
            }
        } else {
            let mut roots = roots().lock().unwrap_or_else(|e| e.into_inner());
            if !roots.iter().any(|root| object_raw(root) == raw) {
                roots.push(object);
            }
        }
    });
}

fn array_cell(value: Value) -> ArrayCell {
    ArrayCell {
        value,
        _hold: retained_object(value),
    }
}

fn array_cells(values: Vec<Value>) -> Vec<ArrayCell> {
    values.into_iter().map(array_cell).collect()
}

fn with_object<R>(value: Value, fallback: R, f: impl FnOnce(&Object) -> R) -> R {
    unsafe { object(value).map(f).unwrap_or(fallback) }
}

fn fail(message: &str) -> ! {
    let _ = writeln!(io::stderr().lock(), "bend: {message}");
    std::process::exit(1)
}

fn report_panic() -> ! {
    fail("native runtime operation failed")
}

fn locked<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

fn push_region() {
    REGIONS.with(|regions| regions.borrow_mut().push(Region::default()));
}

fn pop_region() -> Region {
    REGIONS.with(|regions| regions.borrow_mut().pop().unwrap_or_default())
}

fn replay_tasks_in_current_region() {
    let tasks = REGIONS.with(|regions| {
        regions
            .borrow()
            .last()
            .map(|region| {
                region
                    .objects
                    .iter()
                    .filter_map(|runtime_object| match &runtime_object.object {
                        Object::Task(task) => Some(task.clone()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default()
    });
    for task in tasks {
        task.ensure_finished();
    }
}

fn region_from_objects(objects: Vec<Arc<RuntimeObject>>) -> Region {
    let owned = objects.iter().map(object_raw).collect();
    Region { objects, owned }
}

unsafe fn drop_region(region: Region) {
    // A worker can still be using values borrowed from this region. Join it
    // before releasing any of the region's objects.
    for runtime_object in &region.objects {
        if let Object::Task(task) = &runtime_object.object {
            task.ensure_finished();
        }
    }
    drop(region.objects);
}

fn transfer_region(value: Value, region: Region) {
    transfer_region_values(&[value], region);
}

fn retain_region_values(region: Region, values: &[Value]) -> Region {
    let retained = retain_values(values);
    unsafe { drop_region(region) };
    region_from_objects(retained)
}

fn transfer_region_values(values: &[Value], region: Region) {
    let retained = retain_values(values);
    unsafe { drop_region(region) };
    promote_objects(retained);
}

fn retain_values(values: &[Value]) -> Vec<Arc<RuntimeObject>> {
    let mut seen = HashSet::new();
    values
        .iter()
        .filter_map(|value| retained_object(*value))
        .filter(|object| seen.insert(object_raw(object)))
        .collect()
}

fn promote_objects(objects: Vec<Arc<RuntimeObject>>) {
    REGIONS.with(|regions| {
        if let Some(parent) = regions.borrow_mut().last_mut() {
            for object in objects {
                let raw = object_raw(&object);
                if parent.owned.insert(raw) {
                    parent.objects.push(object);
                }
            }
        } else {
            let mut root_objects = locked(roots());
            for object in objects {
                let raw = object_raw(&object);
                if !root_objects.iter().any(|root| object_raw(root) == raw) {
                    root_objects.push(object);
                }
            }
        }
    });
}

fn apply_inner(mut fun: Value, mut arg: Value) -> Value {
    enum Callable {
        Closure(BendFn),
        Io(u64, Value, Value),
    }
    push_region();
    loop {
        let callable = with_object(fun, None, |object| match object {
            Object::Closure(closure) => Some(Callable::Closure(closure.call)),
            Object::IoAction { opcode, arg0, arg1 } => Some(Callable::Io(*opcode, *arg0, *arg1)),
            _ => None,
        });
        let Some(callable) = callable else {
            fail("attempted to apply a non-function value");
        };

        push_region();
        let result = match callable {
            Callable::Closure(call) => unsafe { call(fun, arg) },
            Callable::Io(opcode, arg0, arg1) => execute_io(opcode, arg0, arg1, arg),
        };
        replay_tasks_in_current_region();
        let region = pop_region();
        let tail = with_object(result, None, |object| match object {
            Object::TailCall { fun, arg } => Some((*fun, *arg)),
            _ => None,
        });
        if let Some((next_fun, next_arg)) = tail {
            transfer_region_values(&[next_fun, next_arg], region);
            replay_tasks_in_current_region();
            let carry = retain_region_values(pop_region(), &[next_fun, next_arg]);
            REGIONS.with(|regions| regions.borrow_mut().push(carry));
            fun = next_fun;
            arg = next_arg;
        } else {
            transfer_region(result, region);
            replay_tasks_in_current_region();
            let carry = pop_region();
            transfer_region(result, carry);
            return result;
        }
    }
}

unsafe impl Send for Object {}
unsafe impl Sync for Object {}

impl TaskState {
    fn pending(join: JoinHandle<TaskResult>) -> Self {
        Self {
            join: Mutex::new(Some(join)),
            result: Mutex::new(None),
            ready: Condvar::new(),
        }
    }

    fn ready(result: TaskResult) -> Self {
        Self {
            join: Mutex::new(None),
            result: Mutex::new(Some(result)),
            ready: Condvar::new(),
        }
    }

    fn ensure_finished(&self) {
        let mut result = locked(&self.result);
        if result.is_some() {
            return;
        }
        if let Some(join) = locked(&self.join).take() {
            *result = Some(
                join.join()
                    .unwrap_or_else(|_| fail("parallel task panicked")),
            );
            self.ready.notify_all();
        }
    }

    fn take_result(&self) -> Option<TaskResult> {
        self.ensure_finished();
        let mut result = locked(&self.result);
        let Some(value) = result.as_mut() else {
            return None;
        };
        if value.joined {
            return None;
        }
        value.joined = true;
        Some(TaskResult {
            value: value.value,
            hold: value.hold.take(),
            joined: true,
        })
    }
}

impl Drop for TaskState {
    fn drop(&mut self) {
        self.ensure_finished();
    }
}

fn take_region_result(closure: Value) -> TaskResult {
    push_region();
    let value = apply_inner(closure, 0);
    replay_tasks_in_current_region();
    let region = pop_region();
    let hold = retained_object(value);
    unsafe { drop_region(region) };
    TaskResult {
        value,
        hold,
        joined: false,
    }
}

unsafe impl Send for TaskResult {}

fn alloc_named_ctor(tag: Value, fields: Vec<Value>) -> Value {
    alloc(Object::Ctor(Ctor { tag, fields }))
}

fn ctor_tag_named(name: &str) -> Option<Value> {
    let names = locked_read(ctor_names());
    names
        .iter()
        .find_map(|(tag, got)| (got == name).then_some(*tag))
        .or_else(|| {
            names
                .iter()
                .find_map(|(tag, got)| got.ends_with(&format!(".{name}")).then_some(*tag))
        })
}

fn locked_read<T>(m: &RwLock<T>) -> std::sync::RwLockReadGuard<'_, T> {
    m.read().unwrap_or_else(|e| e.into_inner())
}

fn locked_write<T>(m: &RwLock<T>) -> std::sync::RwLockWriteGuard<'_, T> {
    m.write().unwrap_or_else(|e| e.into_inner())
}

fn nat(value: Value) -> u64 {
    with_object(value, 0, |o| match o {
        Object::Nat(n) => *n,
        Object::U32(n) => *n as u64,
        _ => 0,
    })
}

fn u32_value(value: Value) -> u32 {
    with_object(value, 0, |o| match o {
        Object::U32(n) => *n,
        Object::Nat(n) => *n as u32,
        _ => 0,
    })
}

fn f32_value(value: Value) -> f32 {
    with_object(value, 0.0, |o| match o {
        Object::F32(n) => *n,
        Object::U32(n) => *n as f32,
        _ => 0.0,
    })
}

fn bool_value(value: Value) -> bool {
    with_object(value, false, |o| match o {
        Object::Ctor(c) => locked_read(ctor_names())
            .get(&c.tag)
            .is_some_and(|name| name == "True" || name.ends_with(".True")),
        _ => false,
    })
}

fn string_value(value: Value) -> String {
    with_object(value, String::new(), |o| match o {
        Object::String(s) => s.clone(),
        Object::Char(c) => c.to_string(),
        _ => String::new(),
    })
}

fn bool_ctor(value: bool, yes: Value, no: Value) -> Value {
    alloc_named_ctor(if value { yes } else { no }, Vec::new())
}

fn cmp_ctor(ord: std::cmp::Ordering, lt: Value, eq: Value, gt: Value) -> Value {
    alloc_named_ctor(
        match ord {
            std::cmp::Ordering::Less => lt,
            std::cmp::Ordering::Equal => eq,
            std::cmp::Ordering::Greater => gt,
        },
        Vec::new(),
    )
}

fn tuple(tag: Value, a: Value, b: Value) -> Value {
    alloc_named_ctor(tag, vec![a, b])
}

fn f32_to_string(value: f32) -> String {
    if value.is_nan() {
        return "nan".to_owned();
    }
    if value == f32::INFINITY {
        return "inf".to_owned();
    }
    if value == f32::NEG_INFINITY {
        return "-inf".to_owned();
    }
    if value == 0.0 && value.is_sign_negative() {
        return "-0".to_owned();
    }
    let s = value.to_string();
    s.replace("e+", "e")
}

fn nat_text(value: u64) -> String {
    format!("{value}n")
}

fn escaped_char(c: char, quote: char) -> String {
    match c {
        '\n' => "\\n".to_owned(),
        '\t' => "\\t".to_owned(),
        '\r' => "\\r".to_owned(),
        '\0' => "\\0".to_owned(),
        '\\' => "\\\\".to_owned(),
        c if c == quote => format!("\\{c}"),
        c if c.is_control() => format!("\\u{{{:x}}}", c as u32),
        c => c.to_string(),
    }
}

fn show_inner(value: Value, depth: usize, seen: &mut HashSet<Value>) -> String {
    if value == 0 {
        return "()".to_owned();
    }
    if depth > 256 {
        return "…".to_owned();
    }
    if !seen.insert(value) {
        return "<cycle>".to_owned();
    }
    let out = with_object(value, "<?>".to_owned(), |object| match object {
        Object::Nat(n) => nat_text(*n),
        Object::U32(n) => n.to_string(),
        Object::F32(n) => {
            let text = f32_to_string(*n);
            if text.contains('.') || text.contains('e') || text == "nan" || text.contains("inf") {
                text
            } else {
                format!("{text}.0")
            }
        }
        Object::Char(c) => format!("'{}'", escaped_char(*c, '\'')),
        Object::String(s) => format!(
            "\"{}\"",
            s.chars().map(|c| escaped_char(c, '"')).collect::<String>()
        ),
        Object::Ctor(ctor) => {
            let name = locked_read(ctor_names())
                .get(&ctor.tag)
                .cloned()
                .unwrap_or_else(|| format!("Ctor{}", ctor.tag));
            let fields = ctor
                .fields
                .iter()
                .map(|field| show_inner(*field, depth + 1, seen))
                .collect::<Vec<_>>();
            if name == "True" || name.ends_with(".True") {
                "True{}".to_owned()
            } else if name == "False" || name.ends_with(".False") {
                "False{}".to_owned()
            } else if name.ends_with("Tuple") || name == "Tuple" || name == "Pair" {
                format!("({})", fields.join(", "))
            } else if fields.is_empty() {
                format!("{name}{{}}")
            } else {
                format!("{name}{{{}}}", fields.join(", "))
            }
        }
        Object::Array(array) => {
            let values = locked(&array.values).clone();
            format!(
                "[{}]",
                values
                    .iter()
                    .map(|cell| show_inner(cell.value, depth + 1, seen))
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        }
        Object::Closure(_) => "<function>".to_owned(),
        Object::TailCall { .. } => "<tail call>".to_owned(),
        Object::Task(_) => "<task>".to_owned(),
        Object::IoAction { .. } => "<io action>".to_owned(),
    });
    seen.remove(&value);
    out
}

fn show(value: Value) -> String {
    show_inner(value, 0, &mut HashSet::new())
}

fn error_constructor(code: u32, message: String, fail_tag: Value, pair_tag: Value) -> Value {
    let code = alloc(Object::U32(code));
    let message = alloc(Object::String(message));
    let pair = tuple(pair_tag, code, message);
    alloc_named_ctor(fail_tag, vec![pair])
}

fn io_print(value: Value, newline: bool, stderr: bool) {
    let text = string_value(value);
    if stderr {
        let mut out = io::stderr().lock();
        let _ = if newline {
            writeln!(out, "{text}")
        } else {
            write!(out, "{text}")
        };
    } else {
        let mut out = io::stdout().lock();
        let _ = if newline {
            writeln!(out, "{text}")
        } else {
            write!(out, "{text}")
        };
    }
}

fn ctor_value(tag: Value, fields: Vec<Value>) -> Value {
    let name = locked_read(ctor_names())
        .get(&tag)
        .cloned()
        .unwrap_or_default();
    let suffix = name.rsplit('.').next().unwrap_or(&name);
    match suffix {
        "Unit" => 0,
        "Zero" => alloc(Object::Nat(0)),
        "Succ" => alloc(Object::Nat(
            nat(fields.first().copied().unwrap_or(0)).saturating_add(1),
        )),
        "True" => alloc_named_ctor(tag, Vec::new()),
        "False" => alloc_named_ctor(tag, Vec::new()),
        "Chr" => {
            let code = u32_value(fields.first().copied().unwrap_or(0));
            char::from_u32(code)
                .map(|c| alloc(Object::Char(c)))
                .unwrap_or_else(|| fail("invalid Unicode scalar value"))
        }
        "SNil" => alloc(Object::String(String::new())),
        "SCon" => {
            let head = fields.first().copied().unwrap_or(0);
            let tail = fields.get(1).copied().unwrap_or(0);
            alloc(Object::String(format!(
                "{}{}",
                char_value(head),
                string_value(tail)
            )))
        }
        "ALeaf" => alloc(Object::Array(Array {
            values: Mutex::new(array_cells(fields.first().copied().into_iter().collect())),
        })),
        "ANode" => {
            let left = fields.first().copied().unwrap_or(0);
            let right = fields.get(1).copied().unwrap_or(0);
            let mut joined = array_values(left);
            joined.extend(array_values(right));
            alloc(Object::Array(Array {
                values: Mutex::new(joined),
            }))
        }
        "U32" => alloc(Object::U32(word_value(
            fields.first().copied().unwrap_or(0),
        ))),
        "F32" => alloc(Object::F32(f32::from_bits(word_value(
            fields.first().copied().unwrap_or(0),
        )))),
        _ => alloc_named_ctor(tag, fields),
    }
}

fn char_value(value: Value) -> char {
    with_object(value, '\0', |object| match object {
        Object::Char(c) => *c,
        Object::U32(n) => char::from_u32(*n).unwrap_or('\0'),
        Object::Ctor(ctor) if ctor.fields.first().is_some() => char_value(ctor.fields[0]),
        _ => '\0',
    })
}

fn array_values(value: Value) -> Vec<ArrayCell> {
    with_object(value, Vec::new(), |object| match object {
        Object::Array(array) => locked(&array.values).clone(),
        _ => Vec::new(),
    })
}

fn word_value(value: Value) -> u32 {
    fn bits(value: Value, bit: u32, out: &mut u32) {
        if bit >= 32 {
            return;
        }
        let (is_word, head, tail) = with_object(value, (false, false, 0), |object| match object {
            Object::Ctor(ctor) => {
                let name = locked_read(ctor_names())
                    .get(&ctor.tag)
                    .cloned()
                    .unwrap_or_default();
                if name.ends_with("WCon") {
                    (
                        true,
                        bool_value(ctor.fields.first().copied().unwrap_or(0)),
                        ctor.fields.get(1).copied().unwrap_or(0),
                    )
                } else {
                    (false, false, 0)
                }
            }
            _ => (false, false, 0),
        });
        if is_word {
            if head {
                *out |= 1 << bit;
            }
            bits(tail, bit + 1, out);
        }
    }
    let direct = with_object(value, None, |object| match object {
        Object::U32(n) => Some(*n),
        _ => None,
    });
    if let Some(n) = direct {
        return n;
    }
    let mut out = 0;
    bits(value, 0, &mut out);
    out
}

fn word_from_u32(value: u32) -> Value {
    let nil = ctor_tag_named("WNil").unwrap_or(0);
    let cons = ctor_tag_named("WCon").unwrap_or(0);
    let mut word = alloc_named_ctor(nil, Vec::new());
    for bit in (0..32).rev() {
        let yes = ctor_tag_named("True").unwrap_or(0);
        let no = ctor_tag_named("False").unwrap_or(0);
        let head = alloc_named_ctor(if value & (1 << bit) != 0 { yes } else { no }, Vec::new());
        word = alloc_named_ctor(cons, vec![head, word]);
    }
    word
}

fn io_action_result(opcode: u64, a: Value, _b: Value) -> Value {
    match opcode {
        1 => {
            io_print(a, true, false);
            0
        }
        2 => {
            io_print(a, false, false);
            0
        }
        3 => {
            io_print(a, true, true);
            0
        }
        4 => {
            let cons = ctor_tag_named("Con").unwrap_or(0);
            let nil = ctor_tag_named("Nil").unwrap_or(0);
            let mut list = alloc_named_ctor(nil, Vec::new());
            let args = locked(argv_store()).clone();
            for arg in args.into_iter().rev() {
                let text = alloc(Object::String(arg));
                list = alloc_named_ctor(cons, vec![text, list]);
            }
            list
        }
        5 => {
            let name = string_value(a);
            let done = ctor_tag_named("Done").unwrap_or(0);
            let fail_tag = ctor_tag_named("Fail").unwrap_or(0);
            let pair = ctor_tag_named("Tuple").unwrap_or(0);
            match std::env::var(&name) {
                Ok(value) => alloc_named_ctor(done, vec![alloc(Object::String(value))]),
                Err(error) => error_constructor(1, error.to_string(), fail_tag, pair),
            }
        }
        6 => {
            let done = ctor_tag_named("Done").unwrap_or(0);
            let fail_tag = ctor_tag_named("Fail").unwrap_or(0);
            let pair = ctor_tag_named("Tuple").unwrap_or(0);
            let got = std::fs::File::open("/dev/urandom").and_then(|mut file| {
                let mut bytes = [0u8; 4];
                file.read_exact(&mut bytes)
                    .map(|_| u32::from_ne_bytes(bytes))
            });
            match got {
                Ok(value) => alloc_named_ctor(done, vec![alloc(Object::U32(value))]),
                Err(error) => error_constructor(
                    error.raw_os_error().unwrap_or(1) as u32,
                    error.to_string(),
                    fail_tag,
                    pair,
                ),
            }
        }
        7 => {
            thread::sleep(Duration::from_millis(u32_value(a) as u64));
            0
        }
        8 => nat_result(
            STARTED
                .get_or_init(Instant::now)
                .elapsed()
                .as_millis()
                .min((1u128 << 48) - 1) as u64,
        ),
        9 => u32_result(THREAD_LIMIT.load(Ordering::SeqCst) as u32),
        _ => fail("unknown native IO action"),
    }
}

fn execute_io(opcode: u64, a: Value, b: Value, continuation: Value) -> Value {
    let result = io_action_result(opcode, a, b);
    bend_tail_call(continuation, result)
}

unsafe extern "C" fn io_emit(env: Value, value: Value) -> Value {
    let tag = bend_capture(env, 0);
    alloc_named_ctor(tag, vec![value])
}

fn u32_bool(value: bool, yes: Value, no: Value) -> Value {
    bool_ctor(value, yes, no)
}
fn u32_cmp(a: u32, b: u32, lt: Value, eq: Value, gt: Value) -> Value {
    cmp_ctor(a.cmp(&b), lt, eq, gt)
}
fn nat_cmp(a: u64, b: u64, lt: Value, eq: Value, gt: Value) -> Value {
    cmp_ctor(a.cmp(&b), lt, eq, gt)
}

fn array_new_value(size: u64, fill: Value) -> Value {
    if size > 24 {
        fail("array is too large for the native runtime");
    }
    let len = 1usize << size;
    alloc(Object::Array(Array {
        values: Mutex::new(array_cells(vec![fill; len])),
    }))
}

fn array_clone_value(value: Value) -> Value {
    with_object(value, 0, |object| match object {
        Object::Array(array) => alloc(Object::Array(Array {
            values: Mutex::new(locked(&array.values).clone()),
        })),
        _ => 0,
    })
}

fn array_pair(tag: Value, array: Value, value: Value) -> Value {
    tuple(tag, array, value)
}

fn array_get_value(array: Value, index: Value, tuple_tag: Value, swap: Option<Value>) -> Value {
    let old = with_object(array, 0, |object| match object {
        Object::Array(a) => {
            let mut values = locked(&a.values);
            let at = if values.is_empty() {
                return 0;
            } else {
                u32_value(index) as usize % values.len()
            };
            let old = values[at].value;
            retain_in_current_region(old);
            if let Some(value) = swap {
                values[at] = array_cell(value);
            }
            old
        }
        _ => 0,
    });
    array_pair(tuple_tag, array, old)
}

fn f32_result(value: f32) -> Value {
    alloc(Object::F32(value))
}

fn parse_f32_prefix(text: &str) -> Option<f32> {
    let start = text
        .bytes()
        .take_while(|byte| matches!(byte, b'\t' | b'\n' | 0x0b | 0x0c | b'\r' | b' '))
        .count();
    let token = &text[start..];
    let bytes = token.as_bytes();
    if bytes.is_empty() {
        return None;
    }
    let sign = match bytes[0] {
        b'+' => 1.0f32,
        b'-' => -1.0f32,
        _ => 1.0f32,
    };
    let offset = usize::from(matches!(bytes[0], b'+' | b'-'));
    let special = &bytes[offset..];
    if special.eq_ignore_ascii_case(b"inf") || special.eq_ignore_ascii_case(b"infinity") {
        return Some(if sign.is_sign_negative() {
            f32::NEG_INFINITY
        } else {
            f32::INFINITY
        });
    }
    if special.eq_ignore_ascii_case(b"nan") {
        let bits = if sign.is_sign_negative() {
            0xffc0_0000
        } else {
            f32::NAN.to_bits()
        };
        return Some(f32::from_bits(bits));
    }

    let mut index = offset;
    let mut digits = 0;
    while bytes.get(index).is_some_and(u8::is_ascii_digit) {
        digits += 1;
        index += 1;
    }
    if bytes.get(index) == Some(&b'.') {
        index += 1;
        while bytes.get(index).is_some_and(u8::is_ascii_digit) {
            digits += 1;
            index += 1;
        }
    }
    if digits == 0 {
        return None;
    }
    if bytes
        .get(index)
        .is_some_and(|byte| matches!(byte, b'e' | b'E'))
    {
        index += 1;
        if bytes
            .get(index)
            .is_some_and(|byte| matches!(byte, b'+' | b'-'))
        {
            index += 1;
        }
        let exponent = index;
        while bytes.get(index).is_some_and(u8::is_ascii_digit) {
            index += 1;
        }
        if exponent == index {
            return None;
        }
    }
    (index == bytes.len())
        .then(|| token.parse::<f32>().ok())
        .flatten()
}

fn u32_result(value: u32) -> Value {
    alloc(Object::U32(value))
}
fn nat_result(value: u64) -> Value {
    alloc(Object::Nat(value))
}
fn check_nat(value: u128) -> u64 {
    const NAT_MAX: u128 = (1u128 << 48) - 1;
    if value > NAT_MAX {
        fail("a Nat past the largest immediate 2^48-1");
    }
    value as u64
}

#[no_mangle]
pub extern "C" fn bend_op_u32_add(a: Value, b: Value) -> Value {
    u32_result(u32_value(a).wrapping_add(u32_value(b)))
}
#[no_mangle]
pub extern "C" fn bend_op_u32_sub(a: Value, b: Value) -> Value {
    u32_result(u32_value(a).wrapping_sub(u32_value(b)))
}
#[no_mangle]
pub extern "C" fn bend_op_u32_and(a: Value, b: Value) -> Value {
    u32_result(u32_value(a) & u32_value(b))
}
#[no_mangle]
pub extern "C" fn bend_op_u32_or(a: Value, b: Value) -> Value {
    u32_result(u32_value(a) | u32_value(b))
}
#[no_mangle]
pub extern "C" fn bend_op_u32_xor(a: Value, b: Value) -> Value {
    u32_result(u32_value(a) ^ u32_value(b))
}
#[no_mangle]
pub extern "C" fn bend_op_u32_is_eq(a: Value, b: Value, yes: Value, no: Value) -> Value {
    u32_bool(u32_value(a) == u32_value(b), yes, no)
}
#[no_mangle]
pub extern "C" fn bend_op_u32_is_ne(a: Value, b: Value, yes: Value, no: Value) -> Value {
    u32_bool(u32_value(a) != u32_value(b), yes, no)
}
#[no_mangle]
pub extern "C" fn bend_op_u32_is_lt(a: Value, b: Value, yes: Value, no: Value) -> Value {
    u32_bool(u32_value(a) < u32_value(b), yes, no)
}
#[no_mangle]
pub extern "C" fn bend_op_u32_is_le(a: Value, b: Value, yes: Value, no: Value) -> Value {
    u32_bool(u32_value(a) <= u32_value(b), yes, no)
}
#[no_mangle]
pub extern "C" fn bend_op_u32_is_gt(a: Value, b: Value, yes: Value, no: Value) -> Value {
    u32_bool(u32_value(a) > u32_value(b), yes, no)
}
#[no_mangle]
pub extern "C" fn bend_op_u32_is_ge(a: Value, b: Value, yes: Value, no: Value) -> Value {
    u32_bool(u32_value(a) >= u32_value(b), yes, no)
}
#[no_mangle]
pub extern "C" fn bend_op_u32_mul(a: Value, b: Value) -> Value {
    u32_result(u32_value(a).wrapping_mul(u32_value(b)))
}
#[no_mangle]
pub extern "C" fn bend_op_u32_div(a: Value, b: Value) -> Value {
    let (a, b) = (u32_value(a), u32_value(b));
    u32_result(if b == 0 { 0 } else { a / b })
}
#[no_mangle]
pub extern "C" fn bend_op_u32_mod(a: Value, b: Value) -> Value {
    let (a, b) = (u32_value(a), u32_value(b));
    u32_result(if b == 0 { a } else { a % b })
}
#[no_mangle]
pub extern "C" fn bend_op_u32_inc(a: Value) -> Value {
    u32_result(u32_value(a).wrapping_add(1))
}
#[no_mangle]
pub extern "C" fn bend_op_u32_shl(a: Value) -> Value {
    u32_result(u32_value(a).wrapping_shl(1))
}
#[no_mangle]
pub extern "C" fn bend_op_u32_shr(a: Value) -> Value {
    u32_result(u32_value(a) >> 1)
}
#[no_mangle]
pub extern "C" fn bend_op_u32_shln(a: Value, n: Value) -> Value {
    let n = u32_value(n);
    u32_result(if n >= 32 { 0 } else { u32_value(a) << n })
}
#[no_mangle]
pub extern "C" fn bend_op_u32_shrn(a: Value, n: Value) -> Value {
    let n = u32_value(n);
    u32_result(if n >= 32 { 0 } else { u32_value(a) >> n })
}
#[no_mangle]
pub extern "C" fn bend_op_u32_not(a: Value) -> Value {
    u32_result(!u32_value(a))
}
#[no_mangle]
pub extern "C" fn bend_op_u32_is_zero(a: Value, yes: Value, no: Value) -> Value {
    u32_bool(u32_value(a) == 0, yes, no)
}
#[no_mangle]
pub extern "C" fn bend_op_u32_cmp(a: Value, b: Value, lt: Value, eq: Value, gt: Value) -> Value {
    u32_cmp(u32_value(a), u32_value(b), lt, eq, gt)
}
#[no_mangle]
pub extern "C" fn bend_op_u32_to_f32(a: Value) -> Value {
    f32_result(u32_value(a) as f32)
}
#[no_mangle]
pub extern "C" fn bend_op_u32_to_nat(a: Value) -> Value {
    nat_result(u32_value(a) as u64)
}
#[no_mangle]
pub extern "C" fn bend_op_u32_from_nat(a: Value) -> Value {
    u32_result(nat(a) as u32)
}

#[no_mangle]
pub extern "C" fn bend_op_f32_add(a: Value, b: Value) -> Value {
    f32_result(f32_value(a) + f32_value(b))
}
#[no_mangle]
pub extern "C" fn bend_op_f32_sub(a: Value, b: Value) -> Value {
    f32_result(f32_value(a) - f32_value(b))
}
#[no_mangle]
pub extern "C" fn bend_op_f32_mul(a: Value, b: Value) -> Value {
    f32_result(f32_value(a) * f32_value(b))
}
#[no_mangle]
pub extern "C" fn bend_op_f32_div(a: Value, b: Value) -> Value {
    f32_result(f32_value(a) / f32_value(b))
}
#[no_mangle]
pub extern "C" fn bend_op_f32_neg(a: Value) -> Value {
    f32_result(-f32_value(a))
}
#[no_mangle]
pub extern "C" fn bend_op_f32_is_eq(a: Value, b: Value, yes: Value, no: Value) -> Value {
    bool_ctor(f32_value(a) == f32_value(b), yes, no)
}
#[no_mangle]
pub extern "C" fn bend_op_f32_is_ne(a: Value, b: Value, yes: Value, no: Value) -> Value {
    bool_ctor(f32_value(a) != f32_value(b), yes, no)
}
#[no_mangle]
pub extern "C" fn bend_op_f32_is_lt(a: Value, b: Value, yes: Value, no: Value) -> Value {
    bool_ctor(f32_value(a) < f32_value(b), yes, no)
}
#[no_mangle]
pub extern "C" fn bend_op_f32_is_le(a: Value, b: Value, yes: Value, no: Value) -> Value {
    bool_ctor(f32_value(a) <= f32_value(b), yes, no)
}
#[no_mangle]
pub extern "C" fn bend_op_f32_is_gt(a: Value, b: Value, yes: Value, no: Value) -> Value {
    bool_ctor(f32_value(a) > f32_value(b), yes, no)
}
#[no_mangle]
pub extern "C" fn bend_op_f32_is_ge(a: Value, b: Value, yes: Value, no: Value) -> Value {
    bool_ctor(f32_value(a) >= f32_value(b), yes, no)
}
#[no_mangle]
pub extern "C" fn bend_op_f32_sqrt(a: Value) -> Value {
    f32_result((f64::from(f32_value(a))).sqrt() as f32)
}
#[no_mangle]
pub extern "C" fn bend_op_f32_exp(a: Value) -> Value {
    f32_result((f64::from(f32_value(a))).exp() as f32)
}
#[no_mangle]
pub extern "C" fn bend_op_f32_log(a: Value) -> Value {
    f32_result((f64::from(f32_value(a))).ln() as f32)
}
#[no_mangle]
pub extern "C" fn bend_op_f32_log2(a: Value) -> Value {
    f32_result((f64::from(f32_value(a))).log2() as f32)
}
#[no_mangle]
pub extern "C" fn bend_op_f32_log10(a: Value) -> Value {
    f32_result((f64::from(f32_value(a))).log10() as f32)
}
#[no_mangle]
pub extern "C" fn bend_op_f32_sin(a: Value) -> Value {
    f32_result((f64::from(f32_value(a))).sin() as f32)
}
#[no_mangle]
pub extern "C" fn bend_op_f32_cos(a: Value) -> Value {
    f32_result((f64::from(f32_value(a))).cos() as f32)
}
#[no_mangle]
pub extern "C" fn bend_op_f32_tan(a: Value) -> Value {
    f32_result((f64::from(f32_value(a))).tan() as f32)
}
#[no_mangle]
pub extern "C" fn bend_op_f32_asin(a: Value) -> Value {
    f32_result((f64::from(f32_value(a))).asin() as f32)
}
#[no_mangle]
pub extern "C" fn bend_op_f32_acos(a: Value) -> Value {
    f32_result((f64::from(f32_value(a))).acos() as f32)
}
#[no_mangle]
pub extern "C" fn bend_op_f32_atan(a: Value) -> Value {
    f32_result((f64::from(f32_value(a))).atan() as f32)
}
#[no_mangle]
pub extern "C" fn bend_op_f32_sinh(a: Value) -> Value {
    f32_result((f64::from(f32_value(a))).sinh() as f32)
}
#[no_mangle]
pub extern "C" fn bend_op_f32_cosh(a: Value) -> Value {
    f32_result((f64::from(f32_value(a))).cosh() as f32)
}
#[no_mangle]
pub extern "C" fn bend_op_f32_tanh(a: Value) -> Value {
    f32_result((f64::from(f32_value(a))).tanh() as f32)
}
#[no_mangle]
pub extern "C" fn bend_op_f32_floor(a: Value) -> Value {
    f32_result((f64::from(f32_value(a))).floor() as f32)
}
#[no_mangle]
pub extern "C" fn bend_op_f32_ceil(a: Value) -> Value {
    f32_result((f64::from(f32_value(a))).ceil() as f32)
}
#[no_mangle]
pub extern "C" fn bend_op_f32_trunc(a: Value) -> Value {
    f32_result((f64::from(f32_value(a))).trunc() as f32)
}
#[no_mangle]
pub extern "C" fn bend_op_f32_abs(a: Value) -> Value {
    f32_result((f64::from(f32_value(a))).abs() as f32)
}
#[no_mangle]
pub extern "C" fn bend_op_f32_atan2(a: Value, b: Value) -> Value {
    f32_result((f64::from(f32_value(a))).atan2(f64::from(f32_value(b))) as f32)
}
#[no_mangle]
pub extern "C" fn bend_op_f32_pow(a: Value, b: Value) -> Value {
    f32_result((f64::from(f32_value(a))).powf(f64::from(f32_value(b))) as f32)
}
#[no_mangle]
pub extern "C" fn bend_op_f32_mod(a: Value, b: Value) -> Value {
    f32_result((f64::from(f32_value(a)) % f64::from(f32_value(b))) as f32)
}
#[no_mangle]
pub extern "C" fn bend_op_f32_to_u32(a: Value) -> Value {
    let x = f32_value(a);
    u32_result(if x >= 0.0 && x < 4294967296.0 {
        x as u32
    } else {
        0
    })
}
#[no_mangle]
pub extern "C" fn bend_op_f32_bits(a: Value) -> Value {
    u32_result(f32_value(a).to_bits())
}
#[no_mangle]
pub extern "C" fn bend_op_f32_show(a: Value) -> Value {
    alloc(Object::String(f32_to_string(f32_value(a))))
}
#[no_mangle]
pub extern "C" fn bend_op_f32_read(a: Value) -> Value {
    let text = string_value(a);
    let parsed = parse_f32_prefix(&text);
    let tag = ctor_tag_named(if parsed.is_some() { "Some" } else { "None" }).unwrap_or(0);
    alloc_named_ctor(
        tag,
        parsed
            .map(|x| vec![alloc(Object::F32(x))])
            .unwrap_or_default(),
    )
}

#[no_mangle]
pub extern "C" fn bend_op_nat_add(a: Value, b: Value) -> Value {
    nat_result(check_nat(nat(a) as u128 + nat(b) as u128))
}
#[no_mangle]
pub extern "C" fn bend_op_nat_mul(a: Value, b: Value) -> Value {
    nat_result(check_nat(nat(a) as u128 * nat(b) as u128))
}
#[no_mangle]
pub extern "C" fn bend_op_nat_double(a: Value) -> Value {
    nat_result(check_nat(nat(a) as u128 * 2))
}
#[no_mangle]
pub extern "C" fn bend_op_nat_cmp(a: Value, b: Value, lt: Value, eq: Value, gt: Value) -> Value {
    nat_cmp(nat(a), nat(b), lt, eq, gt)
}
#[no_mangle]
pub extern "C" fn bend_op_nat_sub(a: Value, b: Value) -> Value {
    nat_result(nat(a).saturating_sub(nat(b)))
}
#[no_mangle]
pub extern "C" fn bend_op_nat_is_lt(a: Value, b: Value, yes: Value, no: Value) -> Value {
    bool_ctor(nat(a) < nat(b), yes, no)
}
#[no_mangle]
pub extern "C" fn bend_op_nat_min(a: Value, b: Value) -> Value {
    nat_result(nat(a).min(nat(b)))
}
#[no_mangle]
pub extern "C" fn bend_op_nat_max(a: Value, b: Value) -> Value {
    nat_result(nat(a).max(nat(b)))
}
#[no_mangle]
pub extern "C" fn bend_op_nat_divmod(a: Value, b: Value, tag: Value) -> Value {
    let (a, b) = (nat(a), nat(b));
    tuple(
        tag,
        alloc(Object::Nat(if b == 0 { 0 } else { a / b })),
        alloc(Object::Nat(if b == 0 { a } else { a % b })),
    )
}

#[no_mangle]
pub extern "C" fn bend_op_bool_or(a: Value, b: Value, yes: Value, no: Value) -> Value {
    bool_ctor(bool_value(a) || bool_value(b), yes, no)
}
#[no_mangle]
pub extern "C" fn bend_op_bool_xor(a: Value, b: Value, yes: Value, no: Value) -> Value {
    bool_ctor(bool_value(a) ^ bool_value(b), yes, no)
}
#[no_mangle]
pub extern "C" fn bend_op_string_append(a: Value, b: Value) -> Value {
    alloc(Object::String(format!(
        "{}{}",
        string_value(a),
        string_value(b)
    )))
}
#[no_mangle]
pub extern "C" fn bend_op_string_length(a: Value) -> Value {
    u32_result(string_value(a).chars().count() as u32)
}

#[no_mangle]
pub extern "C" fn bend_op_array_new(size: Value, fill: Value) -> Value {
    array_new_value(u32_value(size) as u64, fill)
}
#[no_mangle]
pub extern "C" fn bend_op_array_set(array: Value, index: Value, value: Value) -> Value {
    with_object(array, 0, |object| match object {
        Object::Array(a) => {
            let mut values = locked(&a.values);
            if !values.is_empty() {
                let at = u32_value(index) as usize % values.len();
                values[at] = array_cell(value);
            }
            array
        }
        _ => 0,
    })
}
#[no_mangle]
pub extern "C" fn bend_op_array_get(array: Value, index: Value, tag: Value) -> Value {
    array_get_value(array, index, tag, None)
}
#[no_mangle]
pub extern "C" fn bend_op_array_swap(
    array: Value,
    index: Value,
    value: Value,
    tag: Value,
) -> Value {
    array_get_value(array, index, tag, Some(value))
}
#[no_mangle]
pub extern "C" fn bend_op_array_size(array: Value, tag: Value) -> Value {
    let len = with_object(array, 0, |object| match object {
        Object::Array(a) => locked(&a.values).len() as u32,
        _ => 0,
    });
    array_pair(tag, array, alloc(Object::U32(len)))
}
#[no_mangle]
pub extern "C" fn bend_op_array_clone(array: Value, tag: Value) -> Value {
    array_pair(tag, array, array_clone_value(array))
}

fn atomic_update(
    array: Value,
    index: Value,
    operand: Value,
    expected: Option<Value>,
    tag: Value,
    kind: u8,
) -> Value {
    let old = with_object(array, 0, |object| match object {
        Object::Array(a) => {
            let mut values = locked(&a.values);
            if values.is_empty() {
                return 0;
            }
            let at = u32_value(index) as usize % values.len();
            let old = values[at].value;
            retain_in_current_region(old);
            let ov = u32_value(old);
            let rhs = u32_value(operand);
            let next = match kind {
                0 => ov.wrapping_add(rhs),
                1 => ov.min(rhs),
                2 => ov.max(rhs),
                3 => ov & rhs,
                4 => ov | rhs,
                5 => ov ^ rhs,
                6 => {
                    if expected.is_some_and(|x| u32_value(x) == ov) {
                        rhs
                    } else {
                        ov
                    }
                }
                8 => rhs,
                _ => ov,
            };
            let new = if kind == 7 {
                alloc(Object::F32(f32_value(old) + f32_value(operand)))
            } else if kind == 8 {
                operand
            } else {
                alloc(Object::U32(next))
            };
            values[at] = array_cell(new);
            old
        }
        _ => 0,
    });
    array_pair(tag, array, old)
}

#[no_mangle]
pub extern "C" fn bend_op_array_atomic_add(a: Value, i: Value, v: Value, t: Value) -> Value {
    atomic_update(a, i, v, None, t, 0)
}
#[no_mangle]
pub extern "C" fn bend_op_array_atomic_min(a: Value, i: Value, v: Value, t: Value) -> Value {
    atomic_update(a, i, v, None, t, 1)
}
#[no_mangle]
pub extern "C" fn bend_op_array_atomic_max(a: Value, i: Value, v: Value, t: Value) -> Value {
    atomic_update(a, i, v, None, t, 2)
}
#[no_mangle]
pub extern "C" fn bend_op_array_atomic_exch(a: Value, i: Value, v: Value, t: Value) -> Value {
    atomic_update(a, i, v, None, t, 8)
}
#[no_mangle]
pub extern "C" fn bend_op_array_atomic_and(a: Value, i: Value, v: Value, t: Value) -> Value {
    atomic_update(a, i, v, None, t, 3)
}
#[no_mangle]
pub extern "C" fn bend_op_array_atomic_or(a: Value, i: Value, v: Value, t: Value) -> Value {
    atomic_update(a, i, v, None, t, 4)
}
#[no_mangle]
pub extern "C" fn bend_op_array_atomic_xor(a: Value, i: Value, v: Value, t: Value) -> Value {
    atomic_update(a, i, v, None, t, 5)
}
#[no_mangle]
pub extern "C" fn bend_op_array_atomic_cas(
    a: Value,
    i: Value,
    e: Value,
    v: Value,
    t: Value,
) -> Value {
    atomic_update(a, i, v, Some(e), t, 6)
}
#[no_mangle]
pub extern "C" fn bend_op_array_atomic_fadd(a: Value, i: Value, v: Value, t: Value) -> Value {
    atomic_update(a, i, v, None, t, 7)
}

#[no_mangle]
pub extern "C" fn bend_runtime_init(argc: i32, argv: *const *const c_char) -> i32 {
    let result = catch_unwind(AssertUnwindSafe(|| {
        let mut args = Vec::new();
        if argc > 0 && !argv.is_null() {
            for i in 0..argc as isize {
                let p = unsafe { *argv.offset(i) };
                if !p.is_null() {
                    args.push(unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned());
                }
            }
        }
        let mut filtered = Vec::new();
        let mut requested_threads = None;
        if let Some(program) = args.first() {
            filtered.push(program.clone());
        }
        let mut i = 1;
        while i < args.len() {
            match args[i].as_str() {
                "--threads" if i + 1 < args.len() => {
                    if let Ok(n) = args[i + 1].parse::<usize>() {
                        requested_threads = Some(n.clamp(1, 128));
                    }
                    i += 2;
                }
                "--gpu" if i + 1 < args.len() => {
                    i += 2;
                }
                "--gpu-build" | "--bend-help" => {
                    i += 1;
                }
                "--" => {
                    filtered.extend(args[i + 1..].iter().cloned());
                    break;
                }
                _ => {
                    filtered.push(args[i].clone());
                    i += 1;
                }
            }
        }
        *locked(argv_store()) = filtered;
        let default_threads = thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        THREAD_LIMIT.store(
            requested_threads.unwrap_or(default_threads).clamp(1, 128),
            Ordering::SeqCst,
        );
        ACTIVE_THREADS.store(1, Ordering::SeqCst);
        STARTED.get_or_init(Instant::now);
    }));
    if result.is_err() {
        1
    } else {
        0
    }
}

#[no_mangle]
pub extern "C" fn bend_runtime_shutdown() {
    // Values left in the top-level result and IO argument objects are owned by
    // the root region and can all be released together.
    let objects = std::mem::take(&mut *locked(roots()));
    unsafe { drop_region(region_from_objects(objects)) }
}

#[no_mangle]
pub extern "C" fn bend_closure(fnptr: Value, captures: *const Value, n: Value) -> Value {
    let call: BendFn = unsafe { mem::transmute(fnptr as usize) };
    let count = (n as usize).min(1 << 20);
    let captured = if count == 0 || captures.is_null() {
        Vec::new()
    } else {
        unsafe { std::slice::from_raw_parts(captures, count) }.to_vec()
    };
    alloc(Object::Closure(Closure {
        call,
        captures: captured,
    }))
}

#[no_mangle]
pub extern "C" fn bend_capture(env: Value, index: Value) -> Value {
    let capture = with_object(env, 0, |object| match object {
        Object::Closure(closure) => closure.captures.get(index as usize).copied().unwrap_or(0),
        _ => 0,
    });
    retain_in_current_region(capture);
    capture
}

#[no_mangle]
pub extern "C" fn bend_tail_call(fun: Value, arg: Value) -> Value {
    alloc(Object::TailCall { fun, arg })
}

#[no_mangle]
pub extern "C" fn bend_apply(fun: Value, arg: Value) -> Value {
    catch_unwind(AssertUnwindSafe(|| apply_inner(fun, arg))).unwrap_or_else(|_| report_panic())
}

#[no_mangle]
pub extern "C" fn bend_ctor(tag: Value, arity: Value, fields: *const Value) -> Value {
    let n = (arity as usize).min(1 << 20);
    let values = if n == 0 || fields.is_null() {
        Vec::new()
    } else {
        unsafe { std::slice::from_raw_parts(fields, n) }.to_vec()
    };
    ctor_value(tag, values)
}

#[no_mangle]
pub extern "C" fn bend_plain_ctor(tag: Value, arity: Value, fields: *const Value) -> Value {
    let n = arity as usize;
    let values = if n == 0 {
        Vec::new()
    } else {
        unsafe { std::slice::from_raw_parts(fields, n) }.to_vec()
    };
    alloc_named_ctor(tag, values)
}

#[no_mangle]
pub extern "C" fn bend_register_ctor(tag: Value, name: *const c_char, _arity: Value) {
    if name.is_null() {
        return;
    }
    let name = unsafe { CStr::from_ptr(name) }
        .to_string_lossy()
        .into_owned();
    locked_write(ctor_names()).insert(tag, name);
}

#[no_mangle]
pub extern "C" fn bend_tag(value: Value) -> Value {
    with_object(value, 0, |object| match object {
        Object::Ctor(ctor) => ctor.tag,
        Object::String(s) => {
            ctor_tag_named(if s.is_empty() { "SNil" } else { "SCon" }).unwrap_or(0)
        }
        Object::Array(a) => {
            let len = locked(&a.values).len();
            ctor_tag_named(if len == 1 { "ALeaf" } else { "ANode" }).unwrap_or(0)
        }
        Object::Nat(n) => ctor_tag_named(if *n == 0 { "Zero" } else { "Succ" }).unwrap_or(0),
        Object::U32(_) => ctor_tag_named("U32").unwrap_or(0),
        Object::F32(_) => ctor_tag_named("F32").unwrap_or(0),
        Object::Char(_) => ctor_tag_named("Chr").unwrap_or(0),
        _ => 0,
    })
}

#[no_mangle]
pub extern "C" fn bend_field(value: Value, index: Value) -> Value {
    let field = with_object(value, 0, |object| match object {
        Object::Ctor(ctor) => ctor.fields.get(index as usize).copied().unwrap_or(0),
        Object::Nat(n) if index == 0 && *n > 0 => alloc(Object::Nat(n - 1)),
        Object::U32(n) if index == 0 => word_from_u32(*n),
        Object::F32(n) if index == 0 => word_from_u32(n.to_bits()),
        Object::Char(c) if index == 0 => alloc(Object::U32(*c as u32)),
        Object::String(s) => match index {
            0 => s
                .chars()
                .next()
                .map(|c| alloc(Object::Char(c)))
                .unwrap_or(0),
            1 => s
                .chars()
                .next()
                .map(|c| alloc(Object::String(s[c.len_utf8()..].to_owned())))
                .unwrap_or(0),
            _ => 0,
        },
        Object::Array(a) => {
            let values = locked(&a.values);
            if values.len() == 1 && index == 0 {
                let field = values[0].value;
                retain_in_current_region(field);
                field
            } else {
                let mid = values.len() / 2;
                let part = if index == 0 {
                    &values[..mid]
                } else if index == 1 {
                    &values[mid..]
                } else {
                    return 0;
                };
                alloc(Object::Array(Array {
                    values: Mutex::new(part.to_vec()),
                }))
            }
        }
        _ => 0,
    });
    retain_in_current_region(field);
    field
}

#[no_mangle]
pub extern "C" fn bend_nat(value: Value) -> Value {
    alloc(Object::Nat(value))
}

#[no_mangle]
pub extern "C" fn bend_u32(value: u32) -> Value {
    alloc(Object::U32(value))
}

#[no_mangle]
pub extern "C" fn bend_f32(value: f32) -> Value {
    alloc(Object::F32(value))
}

#[no_mangle]
pub extern "C" fn bend_string(bytes: *const u8, len: Value) -> Value {
    let n = (len as usize).min(1 << 30);
    if n == 0 {
        return alloc(Object::String(String::new()));
    }
    if bytes.is_null() {
        return alloc(Object::String(String::new()));
    }
    let slice = unsafe { std::slice::from_raw_parts(bytes, n) };
    alloc(Object::String(String::from_utf8_lossy(slice).into_owned()))
}

#[no_mangle]
pub extern "C" fn bend_array(len: Value, values: *const Value) -> Value {
    let n = (len as usize).min(1 << 24);
    let values = if n == 0 || values.is_null() {
        Vec::new()
    } else {
        unsafe { std::slice::from_raw_parts(values, n) }.to_vec()
    };
    alloc(Object::Array(Array {
        values: Mutex::new(array_cells(values)),
    }))
}

#[no_mangle]
pub extern "C" fn bend_show(value: Value) -> i32 {
    let text = show(value);
    let _ = writeln!(io::stdout().lock(), "{text}");
    0
}

#[no_mangle]
pub extern "C" fn bend_io_action(opcode: Value, arg0: Value, arg1: Value) -> Value {
    alloc(Object::IoAction { opcode, arg0, arg1 })
}

#[no_mangle]
pub extern "C" fn bend_io_run(value: Value) -> i32 {
    let emit = ctor_tag_named("Emit").unwrap_or(0);
    let captures = [emit];
    let cont = bend_closure(
        io_emit as *const () as usize as Value,
        captures.as_ptr(),
        captures.len() as Value,
    );
    let result = bend_apply(value, cont);
    with_object(result, 0, |object| match object {
        Object::Ctor(ctor) => {
            let name = locked_read(ctor_names())
                .get(&ctor.tag)
                .cloned()
                .unwrap_or_default();
            if name.ends_with("Emit") {
                0
            } else if name.ends_with("Halt") {
                let code = ctor.fields.first().copied().map(u32_value).unwrap_or(1);
                let message = ctor
                    .fields
                    .get(1)
                    .copied()
                    .map(string_value)
                    .unwrap_or_default();
                if !message.is_empty() {
                    let _ = writeln!(io::stderr().lock(), "bend: {message}");
                }
                code as i32
            } else {
                let _ = writeln!(
                    io::stderr().lock(),
                    "bend: main returned an invalid IO result: {}",
                    show(result)
                );
                1
            }
        }
        _ => {
            let _ = writeln!(
                io::stderr().lock(),
                "bend: main returned an invalid IO result: {}",
                show(result)
            );
            1
        }
    })
}

#[no_mangle]
pub extern "C" fn bend_cpu_fork(closure: Value) -> Value {
    let limit = THREAD_LIMIT.load(Ordering::SeqCst).max(1);
    let mut current = ACTIVE_THREADS.load(Ordering::SeqCst);
    while current < limit {
        match ACTIVE_THREADS.compare_exchange_weak(
            current,
            current + 1,
            Ordering::SeqCst,
            Ordering::SeqCst,
        ) {
            Ok(_) => {
                let closure_hold = retained_object(closure);
                let join = thread::spawn(move || {
                    let _closure_hold = closure_hold;
                    let result = take_region_result(closure);
                    ACTIVE_THREADS.fetch_sub(1, Ordering::SeqCst);
                    result
                });
                return alloc(Object::Task(Arc::new(TaskState::pending(join))));
            }
            Err(now) => current = now,
        }
    }
    let result = take_region_result(closure);
    alloc(Object::Task(Arc::new(TaskState::ready(result))))
}

#[no_mangle]
pub extern "C" fn bend_cpu_join(task: Value) -> Value {
    let state = with_object(task, None, |object| match object {
        Object::Task(task) => Some(task.clone()),
        _ => None,
    });
    let Some(state) = state else {
        return 0;
    };
    let Some(result) = state.take_result() else {
        return 0;
    };
    let value = result.value;
    transfer_region(
        value,
        region_from_objects(result.hold.into_iter().collect()),
    );
    value
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    use std::sync::atomic::{AtomicBool, AtomicUsize};
    use std::sync::{Barrier, OnceLock};

    static TEST_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    static PARALLEL_BARRIER: OnceLock<Barrier> = OnceLock::new();
    static WAIT_FOR_PEER: AtomicBool = AtomicBool::new(false);
    static ACTIVE_CALLBACKS: AtomicUsize = AtomicUsize::new(0);
    static PEAK_CALLBACKS: AtomicUsize = AtomicUsize::new(0);
    static NESTED_LEAVES: AtomicUsize = AtomicUsize::new(0);
    static BAD_ARRAY_READ: AtomicBool = AtomicBool::new(false);
    static FIRST_WRITE: AtomicBool = AtomicBool::new(false);
    static READ_SNAPSHOT: AtomicBool = AtomicBool::new(false);
    static OVERWRITE_DONE: AtomicBool = AtomicBool::new(false);
    const ARRAY_STRESS_LENGTH: u64 = 20_000;

    fn guard() -> std::sync::MutexGuard<'static, ()> {
        locked(TEST_LOCK.get_or_init(|| Mutex::new(())))
    }

    fn closure(call: BendFn, captures: &[Value]) -> Value {
        bend_closure(
            call as *const () as usize as Value,
            captures.as_ptr(),
            captures.len() as Value,
        )
    }

    unsafe extern "C" fn record_thread(env: Value, index: Value) -> Value {
        let active = ACTIVE_CALLBACKS.fetch_add(1, Ordering::SeqCst) + 1;
        PEAK_CALLBACKS.fetch_max(active, Ordering::SeqCst);
        let mut hasher = DefaultHasher::new();
        thread::current().id().hash(&mut hasher);
        let thread_id = hasher.finish() as u32;
        let array = bend_capture(env, 0);
        let index = alloc(Object::U32(u32_value(index)));
        let value = alloc(Object::U32(thread_id));
        bend_op_array_set(array, index, value);
        if WAIT_FOR_PEER.load(Ordering::SeqCst) {
            PARALLEL_BARRIER.get_or_init(|| Barrier::new(2)).wait();
        }
        ACTIVE_CALLBACKS.fetch_sub(1, Ordering::SeqCst);
        0
    }

    unsafe extern "C" fn update_shared_array(env: Value, _: Value) -> Value {
        let array = bend_capture(env, 0);
        let item = alloc(Object::String("survives callback".to_owned()));
        let boxed = bend_ctor(77, 1, &item);
        let index = alloc(Object::U32(0));
        bend_op_array_set(array, index, boxed);
        0
    }

    unsafe extern "C" fn store_captured_value(env: Value, _: Value) -> Value {
        let array = bend_capture(env, 0);
        let value = bend_capture(env, 1);
        let index = alloc(Object::U32(0));
        bend_op_array_set(array, index, value);
        0
    }

    unsafe extern "C" fn create_parent_value_then_store(env: Value, _: Value) -> Value {
        let array = bend_capture(env, 0);
        let text = alloc(Object::String("survives nested callback".to_owned()));
        let value = bend_ctor(79, 1, &text);
        let captures = [array, value];
        let setter = closure(store_captured_value, &captures);
        bend_apply(setter, 0);
        0
    }

    unsafe extern "C" fn create_parent_value_then_fork(env: Value, _: Value) -> Value {
        let array = bend_capture(env, 0);
        let text = alloc(Object::String("survives forked callback".to_owned()));
        let value = bend_ctor(80, 1, &text);
        let captures = [array, value];
        let setter = closure(store_captured_value, &captures);
        let task = bend_cpu_fork(setter);
        bend_cpu_join(task);
        0
    }

    unsafe extern "C" fn return_box(_: Value, _: Value) -> Value {
        let text = alloc(Object::String("alive after return".to_owned()));
        bend_ctor(78, 1, &text)
    }

    unsafe extern "C" fn identity(_: Value, value: Value) -> Value {
        value
    }

    unsafe extern "C" fn nested(env: Value, _: Value) -> Value {
        let level = u32_value(bend_capture(env, 0));
        if level == 0 {
            NESTED_LEAVES.fetch_add(1, Ordering::SeqCst);
            return 0;
        }
        let next = alloc(Object::U32(level - 1));
        let captures = [next];
        let left = closure(nested, &captures);
        let right = closure(nested, &captures);
        let a = bend_cpu_fork(left);
        let b = bend_cpu_fork(right);
        bend_cpu_join(a);
        bend_cpu_join(b);
        0
    }

    unsafe extern "C" fn io_program(_: Value, continuation: Value) -> Value {
        let action = bend_io_action(9, 0, 0);
        bend_apply(action, continuation)
    }

    unsafe extern "C" fn overwrite_loop(env: Value, arg: Value) -> Value {
        let length = nat(bend_capture(env, 2));
        let n = if arg == 0 { length } else { nat(arg) };
        if n == 0 {
            return 0;
        }
        if n == length - 1 {
            while !READ_SNAPSHOT.load(Ordering::SeqCst) {
                thread::yield_now();
            }
        }
        let array = bend_capture(env, 0);
        let index = bend_capture(env, 1);
        let text = alloc(Object::String(format!("value-{n}")));
        let boxed = bend_ctor(803, 1, &text);
        bend_op_array_set(array, index, boxed);
        if n == length {
            FIRST_WRITE.store(true, Ordering::SeqCst);
        } else if n == length - 1 {
            OVERWRITE_DONE.store(true, Ordering::SeqCst);
        }
        bend_tail_call(env, nat_result(n - 1))
    }

    unsafe extern "C" fn read_loop(env: Value, arg: Value) -> Value {
        let n = nat(arg);
        if n == 0 {
            return 0;
        }
        if n == ARRAY_STRESS_LENGTH {
            while !FIRST_WRITE.load(Ordering::SeqCst) {
                thread::yield_now();
            }
        }
        let array = bend_capture(env, 0);
        let index = bend_capture(env, 1);
        let pair = bend_op_array_get(array, index, 804);
        if n == ARRAY_STRESS_LENGTH {
            READ_SNAPSHOT.store(true, Ordering::SeqCst);
            while !OVERWRITE_DONE.load(Ordering::SeqCst) {
                thread::yield_now();
            }
        }
        let boxed = bend_field(pair, 1);
        let text = bend_field(boxed, 0);
        if !string_value(text).starts_with("value-") && string_value(text) != "seed" {
            BAD_ARRAY_READ.store(true, Ordering::SeqCst);
        }
        bend_tail_call(env, nat_result(n - 1))
    }

    #[test]
    fn fork_join_is_bounded_parallel_and_keeps_shared_array_writes_alive() {
        let _guard = guard();
        THREAD_LIMIT.store(2, Ordering::SeqCst);
        ACTIVE_THREADS.store(1, Ordering::SeqCst);
        ACTIVE_CALLBACKS.store(0, Ordering::SeqCst);
        PEAK_CALLBACKS.store(0, Ordering::SeqCst);
        WAIT_FOR_PEER.store(true, Ordering::SeqCst);
        let zeros = [alloc(Object::U32(0)), alloc(Object::U32(0))];
        let array = bend_array(2, zeros.as_ptr());
        let captures = [array];
        let worker = closure(record_thread, &captures);
        let task = bend_cpu_fork(worker);
        let main_index = alloc(Object::U32(1));
        bend_apply(worker, main_index);
        bend_cpu_join(task);
        WAIT_FOR_PEER.store(false, Ordering::SeqCst);
        let ids = array_values(array);
        assert_ne!(u32_value(ids[0].value), u32_value(ids[1].value));
        assert_eq!(PEAK_CALLBACKS.load(Ordering::SeqCst), 2);

        let updated = bend_array(1, zeros.as_ptr());
        let capture = [updated];
        let updater = closure(update_shared_array, &capture);
        bend_apply(updater, 0);
        let boxed = array_values(updated)[0].value;
        assert_eq!(string_value(bend_field(boxed, 0)), "survives callback");

        // The stored value belongs to the outer apply region while the write
        // runs in a nested apply region. It must be promoted when the outer
        // callback returns, even though the older array is not its result.
        let older = bend_array(1, zeros.as_ptr());
        let capture = [older];
        bend_apply(closure(create_parent_value_then_store, &capture), 0);
        let boxed = array_values(older)[0].value;
        assert_eq!(
            string_value(bend_field(boxed, 0)),
            "survives nested callback"
        );

        let forked = bend_array(1, zeros.as_ptr());
        let capture = [forked];
        bend_apply(closure(create_parent_value_then_fork, &capture), 0);
        let boxed = array_values(forked)[0].value;
        assert_eq!(
            string_value(bend_field(boxed, 0)),
            "survives forked callback"
        );
    }

    #[test]
    fn one_thread_runs_fork_work_on_the_current_thread() {
        let _guard = guard();
        THREAD_LIMIT.store(1, Ordering::SeqCst);
        ACTIVE_THREADS.store(1, Ordering::SeqCst);
        WAIT_FOR_PEER.store(false, Ordering::SeqCst);
        let zeros = [alloc(Object::U32(0)), alloc(Object::U32(0))];
        let array = bend_array(2, zeros.as_ptr());
        let captures = [array];
        let worker = closure(record_thread, &captures);
        let task = bend_cpu_fork(worker);
        let main_index = alloc(Object::U32(1));
        bend_apply(worker, main_index);
        bend_cpu_join(task);
        let ids = array_values(array);
        assert_eq!(u32_value(ids[0].value), u32_value(ids[1].value));
    }

    #[test]
    fn concurrent_array_reads_keep_replaced_cells_alive() {
        let _guard = guard();
        THREAD_LIMIT.store(2, Ordering::SeqCst);
        ACTIVE_THREADS.store(1, Ordering::SeqCst);
        BAD_ARRAY_READ.store(false, Ordering::SeqCst);
        FIRST_WRITE.store(false, Ordering::SeqCst);
        READ_SNAPSHOT.store(false, Ordering::SeqCst);
        OVERWRITE_DONE.store(false, Ordering::SeqCst);

        let text = alloc(Object::String("seed".to_owned()));
        let initial = bend_ctor(803, 1, &text);
        let values = [initial];
        let array = bend_array(values.len() as Value, values.as_ptr());
        let index = alloc(Object::U32(0));
        let length = alloc(Object::Nat(ARRAY_STRESS_LENGTH));
        let captures = [array, index, length];
        let writer = closure(overwrite_loop, &captures);
        let task = bend_cpu_fork(writer);
        let read_captures = [array, index];
        let reader = closure(read_loop, &read_captures);
        bend_apply(reader, length);
        bend_cpu_join(task);

        assert!(!BAD_ARRAY_READ.load(Ordering::SeqCst));
        let final_value = array_values(array)[0].value;
        let text = bend_field(final_value, 0);
        assert!(string_value(text).starts_with("value-"));
    }

    #[test]
    fn nested_forks_finish_and_returned_objects_and_aliases_survive_scopes() {
        let _guard = guard();
        THREAD_LIMIT.store(2, Ordering::SeqCst);
        ACTIVE_THREADS.store(1, Ordering::SeqCst);
        NESTED_LEAVES.store(0, Ordering::SeqCst);
        let depth = alloc(Object::U32(2));
        let captures = [depth];
        let task = bend_cpu_fork(closure(nested, &captures));
        bend_cpu_join(task);
        assert_eq!(NESTED_LEAVES.load(Ordering::SeqCst), 4);

        let boxed = bend_apply(closure(return_box, &[]), 0);
        assert_eq!(string_value(bend_field(boxed, 0)), "alive after return");
        let borrowed = bend_string(b"borrowed".as_ptr(), 8);
        assert_eq!(bend_apply(closure(identity, &[]), borrowed), borrowed);
        assert_eq!(string_value(borrowed), "borrowed");
    }

    #[test]
    fn io_actions_run_through_the_church_continuation() {
        let _guard = guard();
        THREAD_LIMIT.store(3, Ordering::SeqCst);
        bend_register_ctor(901, c"IO.OP.Emit".as_ptr(), 1);
        bend_register_ctor(902, c"IO.OP.Halt".as_ptr(), 2);
        let code = bend_io_run(closure(io_program, &[]));
        assert_eq!(code, 0);
    }
}
