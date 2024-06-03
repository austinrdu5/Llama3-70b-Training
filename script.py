import os
import yaml  # Import the yaml library

os.environ['ACCELERATE_USE_FSDP'] = '1'
os.environ['FSDP_CPU_RAM_EFFICIENT_LOADING'] = '1'

if __name__ == "__main__":
    # Load YAML Configuration
    with open("llama_3_70b_fsdp_qlora.yaml", "r") as f:  # Replace with the actual path to your YAML file
        yaml_config = yaml.safe_load(f)

    # Argument Parsing with TrlParser and YAML Input
    parser = TrlParser((ScriptArguments, TrainingArguments))
    script_args, training_args = parser.parse_dict(yaml_config)  # Pass the YAML content

    # Set gradient checkpointing (if needed)
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}

    # Set seed (if needed)
    set_seed(training_args.seed)

    # Launch Training
    training_function(script_args, training_args)
