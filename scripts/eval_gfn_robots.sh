num_timesteps=20_000_000
results_path="./results"
rews_paths=(
    "/scratch/knagiredla/robonet/logs/exp_GSCA_5_flat_base_hopper_50_000_134_1734445021/policies/rew_modified.json"
    "/scratch/knagiredla/robonet/logs/exp_GSCA_5_rugged_base_hopper_50_000_134_1734445249/policies/rew_modified.json"
    "/scratch/knagiredla/robonet/logs/exp_GSCA_5_gap_base_hopper_50_000_134_1734445249/policies/rew_modified.json"
    "/scratch/knagiredla/robonet/logs/exp_GSCA_5_flat_base_swimmer_50_000_134_1734610818/policies/rew_modified.json"
)

script="python tasks/eval_gfn_robots.py"

for rews_path in "${rews_paths[@]}"; do
    # Extract terrain type and robot type from path
    terrain_type=$(echo $rews_path | grep -o -E "flat|rugged|gap" || echo "unknown")
    robot_type=$(echo $rews_path | grep -o -E "hopper|swimmer")
    
    # Set env_id based on robot type
    env_id="${robot_type^}-v5"  # capitalize first letter
    expt_name="${robot_type}_${terrain_type}"
    
    sbatch ./scripts/eval_sbatch.sh ${script} \
        --rews_path "$rews_path" \
        --env_id "$env_id" \
        --num_timesteps "$num_timesteps" \
        --expt_name "$expt_name" \
        --results_path "$results_path"
    
    echo "rews_path: $rews_path"
    echo "env_id: $env_id"
    echo "num_timesteps: $num_timesteps"
    echo "expt_name: $expt_name"
    echo "-------------------"
done