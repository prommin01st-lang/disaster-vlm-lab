import subprocess, tarfile
tarfile.open("/content/bundle.tar.gz").extractall("/content/dvl", filter="data")
r = subprocess.run(["pip", "install", "-q", "-r", "/content/dvl/requirements-colab.txt"],
                   capture_output=True, text=True)
print("pip rc", r.returncode, r.stderr[-2000:])
# torchao 0.10 ที่ติดมากับ Colab ทำให้ peft 0.19 (LoRA dispatch) โยน ImportError — ไม่ได้ใช้ จึงถอดออก
u = subprocess.run(["pip", "uninstall", "-y", "-q", "torchao"], capture_output=True, text=True)
print("uninstall torchao rc", u.returncode)
if r.returncode != 0:
    print("__BOOTSTRAP_FAILED__")
    raise SystemExit(1)
