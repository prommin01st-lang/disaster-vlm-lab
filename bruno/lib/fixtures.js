// ภาพ fixture จาก dataset (CrisisMMD/Flickr — ลิขสิทธิ์ของเจ้าของภาพ) ไม่แจกใน repo → request ที่ใช้ภาพพวกนี้ skip ถ้ายังไม่ได้ดึง
// ดึงด้วย: python bruno/fetch_fixtures.py  (เขียน bruno/.env → DVL_FIXTURES=<รายชื่อไฟล์ที่มี>)
// ตรวจไฟล์จริงด้วย fs เมื่อรัน --sandbox developer · sandbox safe (ค่าเริ่มต้น) ไม่มี fs จึงอ่านรายชื่อจาก DVL_FIXTURES แทน
function fixtureReady(bru, name) {
  try {
    const fs = require("fs");
    const path = require("path");
    return fs.existsSync(path.join(bru.cwd(), "fixtures", name));
  } catch (e) {
    return (bru.getProcessEnv("DVL_FIXTURES") || "").split(",").map((s) => s.trim()).includes(name);
  }
}

function skipUnlessFixture(bru, name) {
  if (fixtureReady(bru, name)) return true;
  console.log(`fixtures/${name} not fetched — skipping (run: python bruno/fetch_fixtures.py)`);
  bru.runner.skipRequest();
  return false;
}

module.exports = { fixtureReady, skipUnlessFixture };
