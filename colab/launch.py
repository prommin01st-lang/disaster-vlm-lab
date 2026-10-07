import os, subprocess
job = os.environ["JOB"]
os.makedirs("/content/dvl/out", exist_ok=True)
env = dict(os.environ, PYTHONPATH="/content/dvl")
tok = "/content/.hf_token"  # อัปโหลดแยกต่างหาก ไม่ส่งผ่าน argv
if os.path.exists(tok):
    env["HF_TOKEN"] = open(tok).read().strip()
cmd = f"cd /content/dvl && (bash {job}; echo EXIT_CODE=$? >> out/job.log) >> out/job.log 2>&1 &"
subprocess.Popen(["bash", "-c", cmd], env=env)
print("launched", job)
