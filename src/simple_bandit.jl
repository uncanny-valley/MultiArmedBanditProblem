using Plots
include("./ArmedBanditEnvironment.jl")
include("./action_selection.jl")


function simple_bandit(num_actions::Int, ε::Float64, num_runs::Int, num_pulls::Int)
    action_values = zeros(num_actions, num_runs)
    action_counts = ones(num_actions, num_runs)
    N = num_pulls + 1
    M = num_runs

    envs = [ArmedBanditEnvironment(num_actions, seed) for seed in 1:num_runs]
    optimal_actions = optimal_action.(envs)
    num_optimal_actions = zeros(N, 1)
    average_rewards = zeros(N, 1)
    average_rewards[1] = 0.

    for p in 2:N
        rewards = zeros(num_runs, 1)

        for i in 1:M
            A = ε_greedy(ε, action_values[:, i])

            if A == optimal_actions[i]
                num_optimal_actions[p] += 1
            end

            R = reward(envs[i], A)
            action_counts[A, i] += 1
            action_values[A, i] += (1 / action_counts[A, i]) * (R - action_values[A, i])
            rewards[i] = R
        end

        average_rewards[p] = mean(rewards)
    end

    (100 / M) * num_optimal_actions, average_rewards
end

k = 10
num_runs = 2000
num_pulls = 1000
optimal_action_ratio_1, average_reward_history_1 = simple_bandit(k, 0.1, num_runs, num_pulls)
optimal_action_ratio_2, average_reward_history_2 = simple_bandit(k, 0.01, num_runs, num_pulls)
optimal_action_ratio_3, average_reward_history_3 = simple_bandit(k, 0.0, num_runs, num_pulls)

p = plot(average_reward_history_1, label="e=0.1", legend=:bottomright)
plot!(p, average_reward_history_2, label="e=0.01")
plot!(p, average_reward_history_3, label="e=0.0 (greedy)")

xlabel!("Steps")
ylabel!("Average Reward")
savefig(p, "assets/e_greedy_reward_curves.png")

p2 = plot(optimal_action_ratio_1, label="e=0.1", legend=:bottomright)
plot!(p2, optimal_action_ratio_2, label="e=0.01")
plot!(p2, optimal_action_ratio_3, label="e=0.0 (greedy)")

xlabel!("Steps")
ylabel!("% Optimal Action")
savefig(p2, "assets/e_greedy_optimal_action_curves.png")
