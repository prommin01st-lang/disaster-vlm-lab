import os
p = "/content/dvl/out/job.log"
lines = open(p, encoding="utf-8", errors="replace").read().splitlines() if os.path.exists(p) else []
print("\n".join(lines[-15:]))
print("__DONE__" if any(l.startswith("EXIT_CODE=") for l in lines) else "__RUNNING__")
