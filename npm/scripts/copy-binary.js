const fs = require("fs");
const path = require("path");
const os = require("os");

const scriptDir = __dirname;
const npmDir = path.dirname(scriptDir);
const whisperRoot = path.dirname(npmDir);

const platformKey = `${os.platform()}-${os.arch()}`;
const packageDir = path.join(npmDir, "packages", platformKey);

if (!fs.existsSync(packageDir)) {
  console.error(`Error: No package found for ${platformKey}`);
  console.error(`Expected: ${packageDir}`);
  process.exit(1);
}

const possibleSources = [
  path.join(
    whisperRoot,
    "examples",
    "addon.node",
    "build",
    "Release",
    "addon.node.node"
  ),
  path.join(
    whisperRoot,
    "build",
    "examples",
    "addon.node",
    "addon.node.node"
  ),
];

const source = possibleSources.find((candidate) => fs.existsSync(candidate));

if (!source) {
  console.error("Error: Binary not found.");
  console.error("Tried:");
  for (const candidate of possibleSources) {
    console.error(`  - ${candidate}`);
  }
  console.error("");
  console.error("Please build the addon first:");
  console.error("  cd examples/addon.node");
  console.error("  npm install");
  console.error("  npx cmake-js compile");
  process.exit(1);
}

const dest = path.join(packageDir, "whisper.node");

fs.copyFileSync(source, dest);

const stats = fs.statSync(dest);
const sizeMb = (stats.size / (1024 * 1024)).toFixed(2);

console.log(`Copied binary to ${dest}`);
console.log(`Binary size: ${sizeMb} MB`);
