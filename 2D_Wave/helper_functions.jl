function stack!(result_u, u_matrix)
    # Takes in the  y x z u matrix and returns a stacked vector [u1, u2, ... , un]
    y,z = size(u_matrix)
    for i in 1:y
        result_u[(i-1)*z + 1: i*z] = u_matrix[i, :]
    end
    return nothing
end

function get_const_z_vec(u::Matrix{Float64}, Ny::Int, Nz::Int, index::Int)
    # Assumes vector is stacked u = [Uy=1,z^T, Uy=2, z^T, .... , Uy=Ny+1, z^T]
    # Returns the vector len Ny for a given Z index
    result = zeros(Ny + 1)
    for i in 1:Ny+1
        result[i] = u[(i-1)*Nz + index]
    end
    return result
end

function convert!(uv, y, z, u_res::Matrix{Float64}, v_res::Matrix{Float64})
    # Takes in the stacked vector and returns U, V which are displacements and velocities in 2D array, Y rows, Z cols
    NY = length(y)
    NZ = length(z)

    for i in eachindex(y)
        u_res[i, 1:NZ] = uv[(i - 1)*NZ + 1:(i)*NZ]
        v_res[i, 1:NZ] = uv[(i - 1)*NZ + 1 + NY*NZ:(i)*NZ + NY*NZ]

    end
    return nothing
end
