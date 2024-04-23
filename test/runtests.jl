using SchlierenReconstructions
using Test

using Random, LinearAlgebra

@testset "MatrixOperator" begin
    rng = MersenneTwister(123)
    m, n = 5, 3
    A = rand(rng, m, n)
    x = rand(rng, n)
    y = A * x
    z = A' * y
    
    operator = MatrixOperator(A)
    y_new = zeros(operator.dim_range)
    z_new = zeros(operator.dim_domain)
    @time SchlierenReconstructions.op!(y_new, x, operator)
    @time SchlierenReconstructions.op_adj!(z_new, y, operator)
    
    @test all(isapprox.(y_new, y))
    @test all(isapprox.(z_new, z))
end
    

@testset "LSQR" begin
    rng = MersenneTwister(123)
    m, n = 5, 3
    A = rand(rng, m, n)
    x = rand(rng, n)
    b = A * x
    
    x_exact = inv(A' * A) * A' * b  ## least squares solution
    @test all(isapprox.(x, x_exact))
    
    
    operator = MatrixOperator(A)
    prob = LeastSquaresProblem(operator, b)
    @time sol = solve(prob, LSQR(); n_step_max=5)
    x_lsqr = sol.xvec
    @test all(isapprox.(x_lsqr, x_exact))
    
    ## With initial value test
    xvec_init = rand(rng, n)
    operator = MatrixOperator(A)
    prob = LeastSquaresProblem(operator, b)
    @time sol = solve(prob, LSQR(); n_step_max=5, xvec_init=xvec_init)
    x_lsqr = sol.xvec
    @test all(isapprox.(x_lsqr, x_exact))
end