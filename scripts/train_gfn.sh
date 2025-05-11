seed="22 23"
env="hopper"
env_id="Hopper-v5"
# train_steps=500_000
start_point="base"    # "orig" / "base" 
exp_method="GSCA_no_xpos"      # "CA" / "GSCA" / "linearscaling" / "GSCA_no_alive_rew" / "GSCA_no_xpos"
env_terrain="flat"      # "flat" / "incline" / "gap" / "wall" / "rugged"
path="/home/knagiredla/robonet/logs"
max_nodes=6
min_steps=20_000
global_batch_size=32
w1=1
w2=0.002
scripts="python ./tasks/train_gfn.py"
tb_p_b_is_parameterized=False
num_training_steps=50
offline_data_iters=15

# echo "env_terrain is $env_terrain"

if [[ "$env_terrain" == "gap" || "$env_terrain" == "wall" || "$env_terrain" == "rugged" ]]; then
    ext_terrain=1   #True
else
    ext_terrain=0   #False
fi 

# echo "ext_terrain is $ext_terrain

for seed in $seed
do
    experiment_name="hopper_50iter_${seed}_${max_nodes}_${min_steps}"
    for ((i = 0; i < ${#scripts[@]}; i++))
    do  
        sbatch ./scripts/gfn_sbatch.sh ${scripts[$i]} --name $experiment_name \
            --min_steps $min_steps \
            --lastbatch_rl_timesteps 1_000_000 \
            --seed $seed \
            --env_id $env_id \
            --env $env \
            --start_point $start_point \
            --env_terrain $env_terrain \
            --exp_method $exp_method \
            --base_xml_path "./assets/${start_point}_${env}_${env_terrain}.xml" \
            --terrain_from_external_source $ext_terrain \
            --run_path $path \
            --max_gfn_nodes $max_nodes \
            --seed $seed \
            --global_batch_size $global_batch_size \
            --w1 $w1 \
            --w2 $w2 \
            --num_training_steps $num_training_steps \
            --offline_data_iters $offline_data_iters \
            # --tb_p_b_is_parameterized $tb_p_b_is_parameterized
        echo ${scripts[$i]} --name $experiment_name --rl_timesteps $rl_timesteps --terrain_from_external_source $ext_terrain
    done
done
