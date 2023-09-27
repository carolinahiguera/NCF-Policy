#!/bin/bash
GPUS=$1
SEED=$2
CACHE=$3
POLICY=$4

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:4:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

LAUNCHER="basic"
# LAUNCHER="faircluster"

echo extra "${EXTRA_ARGS}"
echo "Testing bowl in dishrack"

if [ "${POLICY}" == "proprio" ]; then
    echo "Running proprio only dishrack"
    CUDA_VISIBLE_DEVICES=${GPUS} \
    python3.8 NCF_policies/isaacgymenvs/train_cluster.py -m \
     task=NCFTaskBowlDishrack \
     test=True \
     headless=False \
     seed=${SEED} \
     launcher=${LAUNCHER} \
     num_envs=20 \
     checkpoint=/media/chiguera/2TB/NCF_v2/outputs_dishrack/proprio_only/v1/stage1_nn/best_reward.pth \
     task.sim.sim_digits=False \
     train.ppo.tactile_info=False \
     train.ppo.ncf_info=False \
     task.rl.orientation_penalty_scale=0.1 \
     train.algo=PPO2 \
     train.ppo.multi_gpu=False \
     train.ppo.minibatch_size=512 \
     train.ppo.output_name=dishrack_proprio_only/"${CACHE}" \
     ${EXTRA_ARGS}

elif [ "${POLICY}" == "tactile" ]; then
    echo "Running proprio + tactile dishrack"
    python3.8 NCF_policies/isaacgymenvs/train_cluster.py -m \
     task=NCFTaskBowlDishrack \
     test=True \
     headless=True \
     seed=${SEED} \
     num_envs=20 \
     pipeline=gpu \
     launcher=${LAUNCHER} \
     checkpoint= \
     task.sim.sim_digits=True \
     task.rl.compute_contact_gt=False \
     task.rl.orientation_penalty_scale=0.1 \
     train.algo=PPO2 \
     train.ppo.tactile_info=True \
     train.ppo.ncf_info=False \
     train.ppo.minibatch_size=512 \
     train.ppo.output_name=dishrack_proprio_tactile/"${CACHE}" \
     ${EXTRA_ARGS}

elif [ "${POLICY}" == "proprio_gt" ]; then
    echo "Running proprio + gt contact dishrack"
    CUDA_VISIBLE_DEVICES=${GPUS} \
    python3.8 NCF_policies/isaacgymenvs/train_cluster.py -m \
     task=NCFTaskBowlDishrack \
     test=True \
     headless=False \
     seed=${SEED} \
     launcher=${LAUNCHER} \
     num_envs=20 \
     checkpoint=/media/chiguera/2TB/NCF_v2/outputs_dishrack/proprio_gt/v1/stage1_nn/best_reward.pth \
     task.rl.orientation_penalty_scale=0.1 \
     task.sim.sim_digits=False \
     task.rl.max_episode_length=250 \
     task.rl.compute_contact_gt=True \
     task.rl.world_ref_pointcloud=True \
     train.algo=PPO2 \
     train.ppo.multi_gpu=False \
     train.ppo.normalize_point_cloud=True \
     train.ppo.ncf_info=True \
     train.ppo.ncf_use_gt=True \
     train.ppo.minibatch_size=512 \
     train.ppo.output_name=dishrack_proprio_gt_norm/"${CACHE}" \
     ${EXTRA_ARGS}

elif [ "${POLICY}" == "proprio_ncf" ]; then
    echo "Running proprio + ncf contact dishrack"
    CUDA_VISIBLE_DEVICES=${GPUS} \
    python3.8 NCF_policies/isaacgymenvs/train_cluster.py -m \
     task=NCFTaskBowlDishrack \
     test=True \
     headless=True \
     seed=${SEED} \
     launcher=${LAUNCHER} \
     num_envs=20 \
     checkpoint=/media/chiguera/2TB/NCF_v2/outputs_dishrack/proprio_ncf/v1/stage1_nn/best_reward.pth \
     task.sim.sim_digits=True \
     task.rl.max_episode_length=250 \
     task.rl.compute_contact_gt=False \
     task.rl.world_ref_pointcloud=True \
     task.ncf.ncf_arch=transformer \
     task.ncf.ncf_epoch=19 \
     train.algo=PPO2 \
     train.ppo.multi_gpu=False \
     train.ppo.normalize_point_cloud=True \
     train.ppo.ncf_info=True \
     train.ppo.ncf_use_gt=False \
     train.ppo.minibatch_size=512 \
     train.ppo.output_name=dishrack_proprio_ncf_norm/"${CACHE}" \
     ${EXTRA_ARGS}

fi