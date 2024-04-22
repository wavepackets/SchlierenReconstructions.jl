@doc raw"""
    LeastSquaresProblem(operator, bvec)

Storing information on a least squares problem
``\mathrm{argmin}_\boldsymbol{x} \left\Vert \boldsymbol{b} - A\boldsymbol{x} \right\Vert_2^2``
- `operator::T where T<:AbstractOperator` corresponding to ``A \in \mathbb{R}^{m\times n}``
- `bvec::T where T<:Vector` corresponding to ``\boldsymbol{b} \in \mathbb{R}^m``

---
最小二乗問題 ``\mathrm{argmin}_\boldsymbol{x} \left\Vert \boldsymbol{b} - A\boldsymbol{x} \right\Vert_2^2``
についての情報を保存する構造体．
- `operator::T where T<:AbstractOperator` は``A \in \mathbb{R}^{m\times n}``に対応
- `bvec::T where T<:Vector` は``\boldsymbol{b} \in \mathbb{R}^m``に対応
"""
@kwdef struct LeastSquaresProblem{T1<:AbstractOperator, T2<:Vector}
    operator::T1
    bvec::T2
end



abstract type AbstractSolver end

@doc raw"""
    LSQR <: AbstractSolver

Flag object for the multiple dipatch for function `solve`.
`solve(prob, LSQR())` means that `prob` will be solved with the LSQR method
(Paige & Saunders (1982) "LSQR: An Algorithm for Sparse Linear Equations and 
Sparse Least Squares," ACM Transactions on Mathematical Software 8(1) 43-71.
doi: 10.1145/355984.355989)

---
関数`solve`の多重ディスパッチ用のフラグオブジェクト．
`solve(prob, LSQR())`は，`prob`がLSQR法で解かれることを意味する．
"""
struct LSQR <: AbstractSolver end

function __sym_ortho(a, b)
    ### Implemented based on scipy.sparse.linalg.lsqr
    ### https://github.com/scipy/scipy/blob/v1.9.0/scipy/sparse/linalg/_isolve/lsqr.py#L96-L586

    if  b ≈ 0
        return sign(a), zero(a), abs(a)
    elseif a ≈ 0
        return zero(b), sign(b), abs(b)
    elseif abs(b) > abs(a)
        tau = a / b
        s = sign(b) / sqrt(1 + tau^2)
        c = s * tau
        r = b / s
    else
        tau = b / a
        c = sign(a) / sqrt(1 + tau^2)
        s = c * tau
        r = a / c
    end

    return c, s, r
end

@doc raw"""
    solve(prob::LeastSquaresProblem, solver::LSQR; ...)

Solve a least squares problem `prob` with LSQR method.
(Paige & Saunders (1982) "LSQR: An Algorithm for Sparse Linear Equations and 
Sparse Least Squares," ACM Transactions on Mathematical Software 8(1) 43-71.
doi: 10.1145/355984.355989)

Note: Implemented based on scipy.sparse.linalg.lsqr
https://github.com/scipy/scipy/blob/v1.9.0/scipy/sparse/linalg/_isolve/lsqr.py#L96-L586

- `n_step_max`, the maximum number of iteration steps
- `xvec_init`, initial guess of the solution `x` (must be the same size as `x`)
"""
function solve(
        prob::LeastSquaresProblem,
        solver::LSQR;
        n_step_max = 5,
        xvec_init = nothing,
        skip_display_norms = 10,
        )
    (; operator, bvec) = prob
    T = eltype(bvec)
    xvec = zeros(T, operator.dim_domain)
    
    ### allocate intermediate arrays
    u = zeros(T, operator.dim_range)
    v = zeros(T, operator.dim_domain)
    w = zeros(T, operator.dim_domain)
    
    u_buf = zeros(T, operator.dim_range)
    v_buf = zeros(T, operator.dim_domain)
    
    condA_est = 0.0
    ddnorm = 0.0
    anorm = 0.0
    r_norms = zeros(T, n_step_max)
    Aᵀr_norms = zeros(T, n_step_max)
    x_norms = zeros(T, n_step_max)
    
    ### initialize
    if !isnothing(xvec_init)
        @assert length(xvec_init) == operator.dim_domain
        xvec .= xvec_init
        op!(u_buf, xvec, operator)
        u .= bvec .- u_buf
    else
        u .= bvec
    end
    β = norm(u)
    
    if β ≈ 0
        v .= xvec
        α = 0.0
    else
        u ./= β
        op_adj!(v, u, operator)
        α = norm(v)
    end
    
    if !(α ≈ 0)
        v ./= α
    end
    
    w .= v
    ρbar = α
    ϕbar = β
    
    for i_step in 1:n_step_max
        u_buf .= 0
        v_buf .= 0
        
        op!(u_buf, v, operator)
        @. u = u_buf - α*u
        β = norm(u)
        anorm = sqrt(anorm^2 + α^2 + β^2) ### なぜこのタイミング? αは更新しなくてよい?
        
        if !(β ≈ 0)
            u ./= β
            
            op_adj!(v_buf, u, operator)
            @. v = v_buf - β*v
            α = norm(v)
            
            if !(α ≈ 0)
                v ./= α
            end
        end
        
        cs, sn, ρ = __sym_ortho(ρbar, β)
        ## original implementation based on Paige and Saunders 1982 is as follows:
        ## ρ = sqrt(ρbar^2 + β^2)
        ## cs = ρbar / ρ
        ## sn = β / ρ
        
        θ = sn * α
        ρbar = -cs * α
        ϕ = cs * ϕbar
        ϕbar = sn * ϕbar
        
        ddnorm += (norm(w) / ρ)^2   ## なぜこのタイミング？
        
        @. xvec += (ϕ/ρ)*w
        @. w = v - (θ/ρ)*w
        
        ### Calculate norms
        x_norm = norm(xvec)
        r_norm = ϕbar
        Aᵀr_norm = α * abs(sn * ϕ)
        condA_est = anorm * sqrt(ddnorm)
        
        r_norms[i_step] = r_norm
        Aᵀr_norms[i_step] = Aᵀr_norm
        x_norms[i_step] = norm(xvec)
        
        if i_step % skip_display_norms == 0
            display(@sprintf(
                    "Step %3d: ‖x‖=%e, ‖r‖=%e, ‖Aᵀr‖=%e, cond(A)≥%e",
                    i_step, x_norm, r_norm, Aᵀr_norm, condA_est
                    ))
        end
    end
    
    sol = (; xvec, x_norms, r_norms, Aᵀr_norms, condA_est)
    return sol
end