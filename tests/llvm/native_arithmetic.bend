import Base

# One polymorphic identity must reuse the native layout of each scalar.
def keep(-A: Data, x: A) -> A:
  x

def uadd(x: U32, y: U32) -> U32:
  U32.add(x, y)

def fadd(x: F32, y: F32) -> F32:
  F32.add(x, y)

def nadd(x: Nat, y: Nat) -> Nat:
  Nat.add(x, y)

def finish(ok: Bool, value: U32) -> U32:
  match ok:
    case True{}:
      U32.add(value, 1)
    case False{}:
      0

def main() -> U32:
  u = keep(U32, uadd(21, 20))
  f = keep(F32, fadd(1.25, 1.75))
  n = keep(Nat, nadd(5n, 7n))
  ok = keep(Bool, Bool.and(F32.is_eq(f, 3.0), Nat.is_eq(n, 12n)))
  finish(ok, u)

#|42
