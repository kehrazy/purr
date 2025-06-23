# purr | Pythonic Universal IR Representation

A replacement for Tablegen when using MLIR.

> [!WARNING]
> this is NOT complete in any way shape or form

## Example

```python
from purr import Dialect, Op, asm, op, types

cool_math = Dialect(
    name="cool_math",
    cpp_namespace="cool::math",
    description="A dialect for cool math operations",
)


class AddOp(Op):
    dialect = cool_math
    lhs = op.operand(types.Tensor)
    rhs = op.operand(types.Tensor)
    output = op.result(types.Tensor)

    assembly_format = (
        asm.operand("lhs"),
        asm.keyword("+"),
        asm.operand("rhs"),
        asm.keyword("->"),
        asm.operand("output"),
    )
```
