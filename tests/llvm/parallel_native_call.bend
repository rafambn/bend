import Base

# Marked calls use CPU workers while their scalar kernel keeps the native ABI.
def add_words(x: U32, y: U32) -> U32:
  U32.add(x, y)

def main() -> IO(Unit):
  a b = add_words!(20, 1) add_words!(20, 1)
  IO.print(U32.show(U32.add(a, b)))

#|42
