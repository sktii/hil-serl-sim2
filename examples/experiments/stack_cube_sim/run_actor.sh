export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1 && \
python ../../train_rlpd.py "$@" \
    --exp_name=stack_cube_sim \
    --checkpoint_path=/home/wayne/hil-serl-sim2/examples/experiments/stack_cube_sim/run  \
    --actor \
