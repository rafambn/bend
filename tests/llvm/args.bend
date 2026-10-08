import Base

def count(xs: List<String>) -> Nat:
  match xs:
    case Nil{}:
      0n
    case _ <> tail:
      List.length(&1, String, tail)

def main() -> IO(Unit):
  do IO<Unit>:
    xs : List<String> <- IO.args()
    IO.print(Nat.show(count(xs)))

#|0
