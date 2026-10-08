# A one-field constructor named Tuple must keep its ADT identity when shown.
type Flag is Data:
  Leaf{}
  Tuple{value: Flag}

def main() -> Flag:
  Tuple{Leaf{}}

#|Tuple{Leaf{}}
