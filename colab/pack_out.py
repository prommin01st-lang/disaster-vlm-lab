import tarfile
with tarfile.open("/content/out.tar.gz", "w:gz") as t:  # ต้องปิดไฟล์ให้ flush ก่อน download
    t.add("/content/dvl/out", arcname="out")
