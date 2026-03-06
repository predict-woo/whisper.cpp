const { getGpuDevices } = require("./dist/index.js");

console.log("=== getGpuDevices() test ===\n");

const devices = getGpuDevices();

console.log(`Returned type: ${typeof devices} (isArray: ${Array.isArray(devices)})`);
console.log(`Device count: ${devices.length}\n`);

for (const dev of devices) {
  console.log(`[${dev.index}] ${dev.name}`);
  console.log(`    description : ${dev.description}`);
  console.log(`    type        : ${dev.type}`);
  console.log(`    memory_free : ${(dev.memory_free / 1e9).toFixed(2)} GB`);
  console.log(`    memory_total: ${(dev.memory_total / 1e9).toFixed(2)} GB`);
  console.log();

  // Validate field types
  const checks = [
    ["index", "number"],
    ["name", "string"],
    ["description", "string"],
    ["type", "string"],
    ["memory_free", "number"],
    ["memory_total", "number"],
  ];
  for (const [field, expected] of checks) {
    const actual = typeof dev[field];
    if (actual !== expected) {
      console.error(`  FAIL: ${field} is ${actual}, expected ${expected}`);
      process.exit(1);
    }
  }

  if (dev.type !== "gpu" && dev.type !== "igpu") {
    console.error(`  FAIL: type is "${dev.type}", expected "gpu" or "igpu"`);
    process.exit(1);
  }
}

console.log("ALL CHECKS PASSED");
