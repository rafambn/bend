import Base

# Counts at or above the word width must zero both directions.
def shifted(+x: U32, +k: Nat) -> U32:
  U32.add(U32.shln(x, k), U32.shrn(x, k))

def main() -> U32:
  a = shifted(7, 32n)
  b = shifted(7, Nat.add(4294967295n, 1n))
  c = shifted(7, 0n)
  U32.add(U32.add(a, b), c)

#|14
