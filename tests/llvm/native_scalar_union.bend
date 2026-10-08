import Base

# The ADT's flat field slot must carry either an i32 or an f32.
type ScalarChoice is Data:
  AsU32{value: U32}
  AsF32{value: F32}

def as_u32(x: ScalarChoice) -> U32:
  match x:
    case AsU32{n}:
      n
    case AsF32{f}:
      F32.to_u32(f)

def main() -> U32:
  as_u32(AsF32{41.75})

#|41
