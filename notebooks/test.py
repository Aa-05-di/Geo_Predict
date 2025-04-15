llm = Llama(
    model_path=model_path,
    n_gpu_layers=5,  # Increased from 1
    n_threads=8,     # M2 has 8 CPU cores
    main_gpu=0,      # Explicitly select GPU
    verbose=True     # Show detailed logs
)