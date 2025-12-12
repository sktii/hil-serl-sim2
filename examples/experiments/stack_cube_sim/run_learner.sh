export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
python ../../train_rlpd.py "$@" \
    --exp_name=stack_cube_sim \
    --checkpoint_path=first_run \
    --demo_path=/home/wayne/hil-serl-sim2/examples/experiments/stack_cube_sim/demo_data/stack_cube_sim_1_demos_2025-12-12_11-05-37.pkl \
    --learner \
