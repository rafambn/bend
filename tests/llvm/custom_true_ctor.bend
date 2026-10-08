# Keep a recursive user `True` distinct from the builtin Bool constructor.
type Flag is Data:
  Leaf{}
  True{value: Flag}

def main() -> Flag:
  True{Leaf{}}

#|True{Leaf{}}
