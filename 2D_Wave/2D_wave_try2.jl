"""
Ok we're doing this this time
"""

using LinearAlgebra
using Plots
using SparseArrays
using Pkg
using BenchmarkTools

include("./diagonal_sbp.jl")
include("./methods.jl")
include("./helper_functions.jl")

# Globals
C = pi / 5

function trivial_mu(Ny, Nz)
    # Assume all mus are 1
    m = zeros((Ny+1)*(Nz+1), (Ny+1)*(Nz+1)) # Start big matrix
    row = ones(Nz+1)
    for i in 1:Ny+1
        m[i, 1 + (i - 1)*Nz: 1 + i*Nz] = row
    end
    return sparse(m)
end



function source_term(t, y_mesh, z_mesh)
    res = zeros(length(y_mesh) * length(z_mesh))
    for i in eachindex(y_mesh)
        for j in eachindex(z_mesh)
            res[(i - 1) * length(y_mesh) + j] = -sin(C * (y_mesh[i] + z_mesh[j]) + t) + 2*C^4*sin(C * (y_mesh[i] + z_mesh[j]) + t)
        end
    end
    return  res # ST F = U_tt - c^2Uxx
end

function initialize(y_mesh, z_mesh)
    N = length(y_mesh) * length(z_mesh)
    res = zeros(2*N)
    for i in eachindex(y_mesh)
        for j in eachindex(z_mesh)
            res[(i - 1) * length(y_mesh) + j] = sin(C*(y_mesh[i] + z_mesh[j]))
            res[(i - 1) * length(y_mesh) + j + N] = C*cos(C*(y_mesh[i] + z_mesh[j]))
        end
    end
    return  res # ST F = U_tt - c^2Uxx

end

function d2(N, dx)

    # Make the basic finite difference operator
    m = spzeros(N+1, N+1)
    m[1, 1] = 1
    m[1, 2] = -2
    m[1, 3] = 1

    m[N+1, N-1] = 1
    m[N+1, N] = -2
    m[N+1, N+1] = 1

    for i in 2:N
        m[i, i - 1] = 1
        m[i, i] = -2
        m[i, i + 1] = 1
    end
    return (1 / dx^2) .* m
end

# Boundary Conditions:

function g(offset, mesh, t)
    # Displacement U(0, Z, t)
    res = zeros(length(mesh))
    for i in eachindex(mesh)
        res[i] = sin((C*(mesh[i] + offset)) + t)
    end
    return res
end


## Let's Start with all Dirichlet for now
function g_prime(offset, mesh, t)
     # Velocity V(0, Z, t)
     res = zeros(length(mesh))
     for i in eachindex(mesh)
         res[i] = C*cos((C*(mesh[i] + offset)) + t)
     end
     return res
 end


function sbp_operators(y0::Int, yN::Int, z0::Int, zN::Int, 
                        Ny::Int, Nz::Int, dy::Float64, dz::Float64)
    # Wrapper around Alex's 1D 1st Derivative operators, then makes the correct 2D for z and y in 
    # E + D 2014

    # This is what looks like diagonal_sbp_D1(p, N; xc = (-1, 1))
    # First get the Easy 1D operators
    (Dy, HIy, Hy, ry) = diagonal_sbp_D1(2, Ny; xc = (y0, yN))
    (Dz, HIz, Hz, rz) = diagonal_sbp_D1(2, Nz; xc = (z0, zN))

    # Next we need to build Bs
    By = zeros(Ny+1, Ny+1)
    By[1,1] = -1.0
    By[Ny+1, Ny+1] = 1.0
    By = sparse(By)

    Bz = zeros(Nz+1, Nz+1)
    Bz[1,1] = -1.0
    Bz[Nz+1, Nz+1] = 1.0
    Bz = sparse(Bz)

    # Next Ss
    Sy = Matrix{Float64}(I, Ny+1, Ny+1)
    Sy[1,1] = -1.5
    Sy[1,2] = 2.0
    Sy[1,3] = -0.5
    Sy[Ny+1,Ny-1] = 0.5
    Sy[Ny+1,Ny] = -2.0
    Sy[Ny+1,Ny+1] = 1.5
    Sy = (1/dy) .* Sy# Make the BS term prop to 1/dx
    Sy = sparse(Sy)

    Sz = Matrix{Float64}(I, Nz+1, Nz+1)
    Sz[1,1] = -1.5
    Sz[1,2] = 2.0
    Sz[1,3] = -0.5
    Sz[Nz+1,Nz-1] = 0.5
    Sz[Nz+1,Nz] = -2.0
    Sz[Nz+1,Nz+1] = 1.5
    Sz = (1/dz) .* Sz# Make the BS term prop to 1/dx
    Sz = sparse(Sz)

    #Is
    Iy = sparse(Matrix{Float64}(I, Ny+1, Ny+1))
    Iz = sparse(Matrix{Float64}(I, Nz+1, Nz+1))

    mu = I(((Ny+1) * (Nz+1)))

    C = ones(Nz+1)
    C[1] = 0
    C[Nz+1] = 0
    C = sparse(diagm(C))

    D2y = d2(Ny, dy) 
    D2z = d2(Nz, dz)

    R_mu_y = dy^7 / 4.0 * kron(transpose(D2y), Iz) * kron(C, Iz) * mu * kron(D2y, Iz)
    R_mu_z = dz^7 / 4.0 * kron(Iy, transpose(D2z)) * kron(Iy, C) * mu * kron(Iy, D2z)

    # FINALLY MADE IT TO THE OPERATOR THANK GAWD
    D2_mu_y = kron(HIy, Iz) * (-1 * transpose(kron(Dy, Iz)) * mu * kron(Hy, Iz) * kron(Dy, Iz) - R_mu_y + mu * (kron(By*Sy, Iz))) 
    D2_mu_z = kron(Iy, HIz) * (-1 * transpose(kron(Iy, Dz)) * mu * kron(Iy, Hz) * kron(Iy, Dz) - R_mu_z + mu * (kron(Iy, Bz*Sz)))
    
    # Finalize with the E matrices
    ey0 = zeros(Ny+1)
    ey0[1] = 1

    eyn = zeros(Ny+1)
    eyn[end] = 1

    ez0 = zeros(Nz+1)
    ez0[1] = 1

    ezn = zeros(Nz+1)
    ezn[end] = 1

    Ef = kron(ey0, Iz)
    Er = kron(eyn, Iz)
    Es = kron(Iy, ez0)
    Ed = kron(Iy, ezn)
    
    (D2y_test, S0_y, SN_y, HIy_test, Hy_test, r) = diagonal_sbp_D2(2, Ny; xc = (y0, yN))
    (D2z_test, S0_z, SN_z, HIz_test, Hz_test, r) = diagonal_sbp_D2(2, Ny; xc = (z0, zN))
    return (sparse(kron(D2y_test, Iz)), sparse(kron(Iy, D2z_test)), Iy, Iz, Hy_test, Hz_test, HIy_test, HIz_test, sparse(S0_y+SN_y), sparse(S0_z+SN_z), mu, Ef, Er, Es, Ed)
end

# Setup SAT Penalty Terms
# Dirichlet is working for y=0, y=Ny so apply for z=0, z=Nz as well
function p_f(params, x, t)
    # Dirichlet SAT for y=0
    # Grab parameters from operators to vectors
    (mu, HIy, HIz, Iy, Iz, Ef, Er, Es, Ed, BSy, BSz, dy, dz, Ny, Nz, y_mesh, z_mesh, D2y, D2z) = params

    # Set Constants per Erickson and Dunham 2014
    alpha_f = -13 / dy 
    beta = 1

    # Massage UV vector so this is a bit easier to work with
    u_res = zeros(Ny+1, Nz+1)
    v_res = zeros(Ny+1, Nz+1)
    convert!(x, y_mesh, z_mesh, u_res, v_res)

    # First term in the sum
    t1 = alpha_f * kron(HIy, Iz) * Ef * (u_res[1, :] - g(y_mesh[1], z_mesh, t))
    t2 = beta * kron(HIy, Iz) * transpose(kron(BSy, Iz)) * Ef * (u_res[1, :] - g(y_mesh[1], z_mesh, t))
    #return zeros(length(t1))
    return t1 + t2 .* dy
end

function p_r(params, x, t)
    # Dirichlet SAT for y=Ny
    (mu, HIy, HIz, Iy, Iz, Ef, Er, Es, Ed, BSy, BSz, dy, dz, Ny, Nz, y_mesh, z_mesh, D2y, D2z) = params
    
    alpha_r = -13 / dy # params from E + D 2014
    beta = 1
    
    u_res = zeros(Ny+1, Nz+1)
    v_res = zeros(Ny+1, Nz+1)
    convert!(x, y_mesh, z_mesh, u_res, v_res)

    t1 = alpha_r * kron(HIy, Iz) * Er * (u_res[end, :] - g(y_mesh[end], z_mesh, t))
    t2 = beta * kron(HIy, Iz) * transpose(kron(BSy, Iz)) * Er * (u_res[end, :]- g(y_mesh[end], z_mesh, t))
    # return zeros(length(t1))
    return t1 + t2 .* dy
    #return t1 + t2

end

function p_d(params, x, t)
    # Dirichlet SAT for z=Nz
    # Grab parameters from operators to vectors
    (mu, HIy, HIz, Iy, Iz, Ef, Er, Es, Ed, BSy, BSz, dy, dz, Ny, Nz, y_mesh, z_mesh, D2y, D2z) = params

    # Set Constants per Erickson and Dunham 2014
    alpha_f = -13 / dz
    beta = 1

    # Massage UV vector so this is a bit easier to work with
    u_res = zeros(Ny+1, Nz+1)
    v_res = zeros(Ny+1, Nz+1)
    convert!(x, y_mesh, z_mesh, u_res, v_res)

    # First term in the sum
    t1 = alpha_f * kron(Iy, HIz) * Ef * (u_res[:, end] - g(z_mesh[end], y_mesh, t))
    t2 = beta * kron(Iy, HIz) * transpose(kron(Iy, BSz)) * Es * (u_res[:, end] - g(z_mesh[end], y_mesh, t))
    #return zeros(length(t1))
    return t1 + t2 .* dz
end

"""
function p_s(params, x, t)

    (mu, HIy, HIz, Iy, Iz, Ef, Er, Es, Ed, BSy, BSz, dy, dz, Ny, Nz, y_mesh, z_mesh, D2y, D2z) = params
    
    alpha_s = -1 # params from E + D 2014
    beta = 1
    
    # Make it easier to grab a given u or v
    u_res = zeros(Ny+1, Nz+1)
    v_res = zeros(Ny+1, Nz+1)
    convert!(x, y_mesh, z_mesh, u_res, v_res)

    t1 = alpha_s .* kron(Iy, HIz) * Es # get first term

    temp = zeros(2*(Ny+1) * (Nz+1))
    temp[1: ((Ny+1)*(Nz+1))] = kron(Iy, BSz)*x[1:((Ny+1)*(Nz+1))]

    convert!(temp, y_mesh, z_mesh, u_res, v_res) # The mu kron(Iz BSz * u) term

    t2 = u_res[:, 1] + g_prime(z_mesh[1], y_mesh, t)

    return dz .* t1 * t2
    # return t1 + t2

end
"""

function p_s(params, x, t)

     # Grab parameters from operators to vectors
     (mu, HIy, HIz, Iy, Iz, Ef, Er, Es, Ed, BSy, BSz, dy, dz, Ny, Nz, y_mesh, z_mesh, D2y, D2z) = params

     # Set Constants per Erickson and Dunham 2014
     alpha_f = -13 / dz
     beta = 1
 
     # Massage UV vector so this is a bit easier to work with
     u_res = zeros(Ny+1, Nz+1)
     v_res = zeros(Ny+1, Nz+1)
     convert!(x, y_mesh, z_mesh, u_res, v_res)
 
     # First term in the sum
     t1 = alpha_f * kron(Iy, HIz) * Ef * (u_res[:, 1] - g(z_mesh[1], y_mesh, t))
     t2 = beta *  kron(Iy, HIz) * transpose(kron(Iy, BSz)) * Es * (u_res[:, 1] - g(z_mesh[1], y_mesh, t))
     #return zeros(length(t1))
     return t1 + t2 .* dz

end


function rhs(t, x, params)
    # Total right hand side of our ODEs
    (mu, HIy, HIz, Iy, Iz, Ef, Er, Es, Ed, BSy, BSz, dy, dz, Ny, Nz, y_mesh, z_mesh, D2y, D2z) = params
    N = (Ny + 1) * (Nz + 1)
    u = x[1:N]
    v = x[N+1:2*N]
    b = zeros(2 * N)
    b[1:N] = v # move u = v part
    b[N+1:2*N] = C^2 .* ((D2y + D2z) * u) + p_f(params, x, t) + p_r(params, x, t) + p_s(params, x, t) + p_d(params, x, t) + source_term(t, y_mesh, z_mesh)# update v with sbp
    return b
end

function run()
    
    # MESHING

    # Space
    Y0 = 0
    Z0 = 0

    YN = 5
    ZN = 5

    DY = 0.0625
    DZ = 0.0625

    Y_GRID = 0:DY:YN
    Z_GRID = 0:DZ:ZN

    NY = length(Y_GRID) - 1
    NZ = length(Z_GRID) - 1
    N = (NY + 1) * (NZ + 1)

    # Time
    A = 0
    B = 1
    DT = 0.0001

    T_GRID = A:DT:B
    NT = length(T_GRID)

    # Get SBP Operators
    print("Timing for SBP OP Creation:\n")
    @time (D2y, D2z, Iy, Iz, Hy, Hz, HIy, HIz, BSy, BSz, mu, Ef, Er, Es, Ed) = sbp_operators(Y0, YN, Z0, ZN, NY, NZ, DY, DZ)
    #     (sparse(kron(D2y_test, Iz)), sparse(kron(Iy, D2z_test)), Iy, Iz, Hy_test, Hz_test, HIy_test, HIz_test, sparse(S0_y+SN_y), sparse(S0_z+SN_z), mu, Ef, Er, Es, Ed)
    # Set Up Initials
    result = zeros(2 * (NY+ 1) * (NZ + 1), NT)
    c = initialize(Y_GRID, Z_GRID)
    params = (mu, HIy, HIz, Iy, Iz, Ef, Er, Es, Ed, BSy, BSz, DY, DZ, NY, NZ, Y_GRID, Z_GRID, D2y, D2z)

    #        (mu, HIy, HIz, Iy, Iz, Ef, Er, Es, Ed, BSy, BSz, dy, dz, Ny, Nz, y_mesh, z_mesh,D2y, D2z) = params

    # forward_euler!(f, c, dt, result, time_mesh, params)
    print("\nTiming for RK2:\n")
    @time rk2!(rhs, c, DT, result, T_GRID, params)

    u = zeros(NY+1, NZ+1)
    v = zeros(NY+1, NZ+1)

    convert!(result[:, 1], Y_GRID, Z_GRID, u, v)
    plot(Z_GRID, u[1, :], label="Numerical")
    plot!(Z_GRID, [sin(C*z + T_GRID[1]) for z in Z_GRID], label="Exact" )
    png("2D Test Y0 Z TSTART")

    plot(Y_GRID, u[:, 1], label = "Numerical")
    plot!(Y_GRID, [sin(C*z + T_GRID[1]) for z in Y_GRID], label="Exact" )
    png("2D Z0 Y Plot TSTART")

    plot(Z_GRID, u[end, :], label="Numerical")
    plot!(Z_GRID, [sin(C*(z + Y_GRID[end]) + T_GRID[1]) for z in Z_GRID], label="Exact" )
    png("2D Test YN Z TSTART")

    plot(Y_GRID, v[:, end], label = "Numerical")
    plot!(Y_GRID, [C*cos(C*(z + Z_GRID[end]) + T_GRID[1]) for z in Y_GRID], label="Exact" )
    png("2D ZN Y Plot TSTART")

    convert!(result[:, end], Y_GRID, Z_GRID, u, v)
    
    plot(Z_GRID, u[1, :], label="Numerical")
    plot!(Z_GRID, [sin(C*z + T_GRID[end]) for z in Z_GRID], label="Exact" )
    png("2D Test Y0 Z TEND")

    plot(Y_GRID, u[:, 1], label = "Numerical")
    plot!(Y_GRID, [sin(C*z + T_GRID[end]) for z in Y_GRID], label="Exact" )
    png("2D Z0 Y Plot TEND")

    plot(Z_GRID, u[end, :], label="Numerical")
    plot!(Z_GRID, [sin(C*(z + Y_GRID[end]) + T_GRID[end]) for z in Z_GRID], label="Exact" )
    png("2D Test YN Z TEND")

    plot(Y_GRID, u[:, end], label = "Numerical")
    plot!(Y_GRID, [sin(C*(z + Z_GRID[end]) + T_GRID[end]) for z in Y_GRID], label="Exact" )
    png("2D ZN Y Plot TEND")

    test = zeros(div(length(result[:, end]), 2))
    stack!(test, u)

    print(norm(result[1:(NY+1)*(NZ+1), end] - test))
    
    return nothing
end

run()

