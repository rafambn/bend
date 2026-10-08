import Base

# U32 is Word(32n): construction and nested field matching stay native.
def twice32(+w: Word(32n)) -> Word(32n):
  Word.add(32n, w, w)

def wrap(w: Word(32n)) -> U32:
  U32{w}

def low_two_xor(u: U32) -> Bool:
  match u:
    case U32{WCon{lo, WCon{hi, _}}}:
      Bool.xor(lo, hi)

def finish(ok: Bool, total: Nat) -> U32:
  match ok:
    case True{}:
      U32.from_nat(Nat.add(total, 40n))
    case False{}:
      0

def main() -> U32:
  one32 = Word.inc(32n, Word.zero(32n))
  +two32 = twice32(one32)
  finish(low_two_xor(wrap(two32)), Word.to_nat(32n, two32))

#|42
