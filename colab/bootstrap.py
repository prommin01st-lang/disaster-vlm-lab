import subprocess, tarfile
tarfile.open("/content/bundle.tar.gz").extractall("/content/dvl", filter="data")
r = subprocess.run(["pip", "install", "-q", "-r", "/content/dvl/requirements-colab.txt"],
                   capture_output=True, text=True)
print("pip rc", r.returncode, r.stderr[-2000:])
if r.returncode != 0:
    print("__BOOTSTRAP_FAILED__")
    raise SystemExit(1)
