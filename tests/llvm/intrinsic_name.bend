import Base

def u32_add(a: U32, b: U32) -> U32:
  U32.sub(a, b)

def main() -> IO(Unit):
  IO.print(U32.show(u32_add(42, 5)))

#|37
