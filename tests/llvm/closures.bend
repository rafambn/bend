import Base

def add(k: U32) -> U32 -> U32:
  x => U32.add(x, k)

def main() -> IO(Unit):
  f = add(2)
  do IO<Unit>:
    IO.print(U32.show(f(3)))

#|5
