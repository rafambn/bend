import Base

def main() -> IO(Unit):
  do IO<Unit>:
    IO.print(Nat.show(Nat.add(2n, 3n)))
    IO.print(U32.show(U32.mul(6, 7)))
    IO.print(U32.show(F32.to_u32(F32.mul(1.5, 2.0))))

#|5
#|42
#|3
