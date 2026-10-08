import Base

type Items is Data:
  Empty{}
  More{head: U32, tail: Items}

def sum(xs: Items) -> U32:
  match xs:
    case Empty{}:
      0
    case More{head, tail}:
      U32.add(head, sum(tail))

def main() -> IO(Unit):
  IO.print(U32.show(sum(More{1, More{2, More{3, Empty{}}}})))

#|6
