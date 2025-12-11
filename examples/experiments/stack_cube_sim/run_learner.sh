export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
python ../../train_rlpd.py "$@" \
    --exp_name=stack_cube_sim \
    --checkpoint_path=first_run \
    --demo_path=/path/to/your/demo_data/stack_cube_sim_20_demos_DATE.pkl \
    --learner \
