import Base

# A general IO caller should cross directly into a native scalar kernel.
def add_words(x: U32, y: U32) -> U32:
  U32.add(x, y)

def main() -> IO(Unit):
  IO.print(U32.show(add_words(40, 2)))

#|42
