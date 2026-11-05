import os
import subprocess

# Griglia di valori da testare
learning_rates = [1e-2, 5e-2, 1e-3, 5e-3]
lambdas = [1000, 500, 100, 0]
steps = [500, 100, 1000]
prior = [0, 0.1, 1, 10]
prior_every = [1, 10, 20, 30]
with open("templateNone.slurm", "r") as f:
    template = f.read()

os.makedirs("generated_jobs", exist_ok=True)
os.makedirs("logs", exist_ok=True)

for lr in learning_rates:
    for lmb in lambdas:
        for st in steps:
            for pr in prior:
                for pe in prior_every:
                    exp_name = f"lr{lr}_lam{lmb}_steps{st}_None_local_prior{pr}_every{pe}"
                    
                    # Sostituzione variabili
                    job_script = (
                        template
                        .replace("__LR__", str(lr))
                        .replace("__LAMBDA__", str(lmb))
                        .replace("__STEPS__", str(st))
                        .replace("__EXP_NAME__", exp_name)
                        .replace("__PRIOR__", str(pr))
                        .replace("__PRIOR_EVERY__", str(pe))
                    )
                    
                    file_path = f"generated_jobs_paper/job_{exp_name}.slurm"
                    with open(file_path, "w") as f:
                        f.write(job_script)
                    
                    # Lancio del job SLURM
                    subprocess.run(["sbatch", file_path])
                    print(f"Lanciato Job: {exp_name}")