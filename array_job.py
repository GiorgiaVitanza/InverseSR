import os
import subprocess

# Griglia di valori da testare
learning_rates = [1e-2]
lambdas = [1000]
steps = [500]

with open("templateNone.slurm", "r") as f:
    template = f.read()

os.makedirs("generated_jobs", exist_ok=True)
os.makedirs("logs", exist_ok=True)

for lr in learning_rates:
    for lmb in lambdas:
        for st in steps:
            exp_name = f"opt_lr{lr}_lam{lmb}_steps{st}_None_augment_Inversion_global_robust_5"
            
            # Sostituzione variabili
            job_script = (
                template
                .replace("__LR__", str(lr))
                .replace("__LAMBDA__", str(lmb))
                .replace("__STEPS__", str(st))
                .replace("__EXP_NAME__", exp_name)
            )
            
            file_path = f"generated_jobs/job_{exp_name}.slurm"
            with open(file_path, "w") as f:
                f.write(job_script)
            
            # Lancio del job SLURM
            subprocess.run(["sbatch", file_path])
            print(f"Lanciato Job: {exp_name}")