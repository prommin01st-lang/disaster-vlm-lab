import os, subprocess
job = os.environ["JOB"]
os.makedirs("/content/dvl/out", exist_ok=True)
cmd = f"cd /content/dvl && (bash {job}; echo EXIT_CODE=$? >> out/job.log) >> out/job.log 2>&1 &"
subprocess.Popen(["bash", "-c", cmd], env=dict(os.environ, PYTHONPATH="/content/dvl"))
print("launched", job)
