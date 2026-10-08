import Base

# Reinterpret an explicit F32{Word(32n)} through U32 without boxing.
def as_u32(x: F32) -> U32:
  match x:
    case F32{bits}:
      U32{bits}

def main() -> U32:
  bits = Word.inc(32n, Word.zero(32n))
  as_u32(F32{bits})

#|1
