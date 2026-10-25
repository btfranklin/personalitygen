import { rmSync } from "node:fs";

const allowedDirectories = new Set([
  "dist",
  ".test-dist",
  "../../examples/.example-dist",
]);
const directory = process.argv[2];

if (directory === undefined || !allowedDirectories.has(directory)) {
  throw new Error(`Refusing to clean unsupported output directory: ${directory}`);
}

rmSync(directory, { recursive: true, force: true });
