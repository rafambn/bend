type Atom is Data:
  One{}

type Counter is Data:
  Zero{value: Atom}

def unwrap(x: Counter) -> Atom:
  match x:
    case Zero{value}:
      value

def main() -> Atom:
  unwrap(Zero{One{}})

#|One{}
