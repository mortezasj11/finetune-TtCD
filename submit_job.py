#!/usr/bin/env python3
import os
import submitit
import yaml
import argparse

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def submit_job(config):
    # Create executor
    executor = submitit.AutoExecutor(folder=config['log']['folder'])
    
    # Set slurm parameters
    executor.update_parameters(
        name=config['name'],
        partition=config['partition'],
        time=config['time'],
        nodes=config['nodes'],
        gpus_per_node=config['gpus_per_node'],
        cpus_per_task=config['cpus_per_task'],
        mem_gb=config['mem_gb'],
        slurm_additional_parameters={
            'mail-user': config.get('email', ''),
            'mail-type': config.get('email_type', 'ALL'),
        }
    )
    
    # Define job function
    def job():
        # Set up environment
        for setup_cmd in config['setup']:
            os.system(setup_cmd)
        
        # Run command
        cmd = ' '.join(config['command'])
        print(f"Running command: {cmd}")
        return os.system(cmd)
    
    # Submit job
    job = executor.submit(job)
    print(f"Submitted job: {job.job_id}")
    return job

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Submit multi-GPU training job")
    parser.add_argument("--config", type=str, default="multiGPU_submit.yaml", 
                        help="Path to the YAML configuration file")
    args = parser.parse_args()
    
    config = load_config(args.config)
    job = submit_job(config) 