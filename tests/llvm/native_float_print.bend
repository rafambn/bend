import Base

# Pure native output must match the established F32 printer at edge cases.
def main() -> (F32 & F32 & F32 & F32 & F32):
  (F32.neg(0.0), F32.div(0.0, 0.0), F32.div(1.0, 0.0),
    F32.div(1.0, 10000000.0), F32.mul(1000000000000.0, 1000000000.0))

#|(-0.0, nan, inf, 1.0e-7, 1.0e+21)
