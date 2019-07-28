using Random, Distributions


mutable struct ArmedBanditEnvironment
    k::Int
    seed::Int
    reward_distributions::Array{Normal}
    optimal_action::Int
end

# Initialize reward distributions as normal distributions where μ is centered at samples from Normal(μ=0.0, σ=1.0) σ is 1.0
function ArmedBanditEnvironment(k::Int, seed::Int)
    reward_distributions = []
    μs = []

    for i in 1:k
        rng = MersenneTwister(i * seed)
        μ = randn(rng)
        push!(μs, μ)
        push!(reward_distributions, Normal(μ))
    end

    ArmedBanditEnvironment(k, seed, reward_distributions, getindex(argmax(μs), 1))
end

reward(env::ArmedBanditEnvironment, action::Int) = rand(env.reward_distributions[action])

optimal_action(env::ArmedBanditEnvironment) = env.optimal_action
