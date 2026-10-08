import Base

def array() -> Array<U32>:
  a = [0 : U32*4n]
  a[1] <- 5
  a[3] <- 7

def main() -> IO(Unit):
  IO.print(List.show(~&1, ~U32, ~(x => U32.show(x)),
    Array.to_list(~U32, array())))

#|[0, 5, 0, 7]
