num_timesteps=20_000_000

# Initialize empty array for reward paths
rews_paths=("/home/knagiredla/robonet/logs/exp_hoppersens_1_0.002_6_20_000_22_1745678777/policies/rew_modified.json")

# # Find all rew_modified.json files in the logs directory
# echo "Collecting reward paths from ./logs..."
# while IFS= read -r path; do
#     if [[ -f "$path" ]]; then
#         rews_paths+=("$path")
#         echo "Found: $path"
#     fi
# done < <(find ./logs -path "*/policies/rew_modified.json")

# # Check if any paths were found
# if [ ${#rews_paths[@]} -eq 0 ]; then
#     echo "No reward files found! Exiting."
#     exit 1
# else
#     echo "Found ${#rews_paths[@]} reward files."
# fi

alive_rew=1

script="python tasks/eval_gfn_robots.py"

for rews_path in "${rews_paths[@]}"; do
    # Extract w1 and w2 values from the path
    if [[ $rews_path =~ exp_sensitivity_([0-9.]+)_([0-9.]+)_ ]]; then
        w1="${BASH_REMATCH[1]}"
        w2="${BASH_REMATCH[2]}"
        echo "Extracted w1=$w1, w2=$w2 from path"
    else
        echo "Warning: Could not extract w1 and w2 from path: $rews_path"
        # Default values if extraction fails
        w1=1
        w2=0.002
    fi
    # Extract terrain type and robot type from path
    terrain_type=$(echo "$rews_path" | grep -o -E "flat|rugged|gap" | head -n1 || echo "unknown")
    robot_type=$(echo $rews_path | grep -o -E "hopper|swimmer|ant")
    echo "robot_type: $robot_type"
    
    # Set env_id based on robot type
    env_id="${robot_type^}-v5"  # capitalize first letter
    expt_name="hopper_test_${w1}_${w2}_${robot_type}"
    
    results_path="./sensitivity_results/${expt_name}"

    echo "results_path: $results_path"
    
    # Check if results directory exists, create it if not
    if [ ! -d "$results_path" ]; then
        mkdir -p "$results_path"
        echo "Created directory: $results_path"
    fi

    echo "terrain_type: $terrain_type"
    echo "expt_name_1: $expt_name"
    sbatch ./scripts/eval_sbatch.sh ${script} \
        --rews_path "$rews_path" \
        --env_id "$env_id" \
        --num_timesteps "$num_timesteps" \
        --expt_name "$expt_name" \
        --results_path "$results_path" \
        --alive_rew "$alive_rew"
    
    echo "rews_path: $rews_path"
    echo "env_id: $env_id"
    echo "num_timesteps: $num_timesteps"
    echo "expt_name: $expt_name"
    echo "-------------------"
    echo "alive_rew: $alive_rew"
done