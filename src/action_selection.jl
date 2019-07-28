
"""
Select a random action with probability ε or if all action-value estimates are equal
Greedily select optimal action with probability 1 - ε
"""
function ε_greedy(ε_threshold::Float64, action_values)
    sample = rand(Uniform())
    num_actions = size(action_values, 1)

    if sample < ε_threshold || all(a -> a == getindex(action_values, 1), action_values)
        return rand(1:num_actions)
    else
        return getindex(argmax(action_values), 1)
    end
end
