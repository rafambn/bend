import Base

# One dependent Word(n) helper is reused at three concrete widths.
def twice(n: Nat, +w: Word(n)) -> Word(n):
  Word.add(n, w, w)

def main() -> Nat:
  one8 = Word.inc(8n, Word.zero(8n))
  one16 = Word.inc(16n, Word.zero(16n))
  one32 = Word.inc(32n, Word.zero(32n))
  two8 = twice(8n, one8)
  two16 = twice(16n, one16)
  two32 = twice(32n, one32)
  small = Nat.add(Word.to_nat(8n, two8), Word.to_nat(16n, two16))
  Nat.add(small, Word.to_nat(32n, two32))

#|6n
