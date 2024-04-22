abstract type AbstractOperator end

@doc raw"""
    MatrixOperator(A, dim_range, dim_domain) <: AbstractOperator

Flag object for the multiple dipatch of `op!` and `op_adj!`.

`op!(y, x, MatrixOperator(A))` means that `y = A*x`,
- `A <: Matrix` corresponds to ``A \in \mathbb{R}^{m\times n}``
- `dim_range` corresponds to ``m`` (dimension of `y`)
- `dim_domain` corresponds to ``n`` (dimension of `x`)

---
関数`op!`, `op_adj!`の多重ディスパッチのためのフラグオブジェクト．
`op!(y, x, MatrixOperator(A))`は，`y = A*x`の計算をすることを意味する．
- `A <: Matrix`,  ``A \in \mathbb{R}^{m\times n}``
- `dim_range`, ``m`` (`y`の次元)
- `dim_domain`, ``n`` (`x`の次元)
"""
struct MatrixOperator{T1<:Matrix, T2} <: AbstractOperator
    A::T1
    dim_range::T2
    dim_domain::T2
end

function MatrixOperator(A)
    return MatrixOperator(A, size(A,1), size(A,2))
end

function op!(y, x, operator::MatrixOperator)
    A = operator.A
    y .= A * x
end

function op_adj!(x, y, operator::MatrixOperator)
    A = operator.A
    x .= A' * y
end