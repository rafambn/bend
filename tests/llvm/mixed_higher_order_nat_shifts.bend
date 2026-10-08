import Base

# Passing shift operations through a generic callback must retain full Nat k.
def apply_shift(f: U32 -> Nat -> U32, x: U32, k: Nat) -> U32:
  f(x, k)

def main() -> IO(Unit):
  left = apply_shift(U32.shln, 7, Nat.add(4294967295n, 1n))
  right = apply_shift(U32.shrn, 7, 32n)
  IO.print(U32.show(U32.add(left, right)))

#|0
