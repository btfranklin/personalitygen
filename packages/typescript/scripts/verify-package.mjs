import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { isAbsolute, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { arrayValue, objectValue, parseJson, stringValue } from "./json-data.mjs";

const packageRoot = fileURLToPath(new URL("..", import.meta.url));
const packageMetadata = objectValue(
  parseJson(readFileSync(new URL("../package.json", import.meta.url), "utf8")),
);
const temporaryDirectory = mkdtempSync(join(tmpdir(), "personalitygen-npm-"));

/** @param {string} name */
function argumentValue(name) {
  const index = process.argv.indexOf(name);
  if (index === -1) {
    return undefined;
  }
  const value = process.argv[index + 1];
  assert.ok(value, `${name} requires a value`);
  return isAbsolute(value) ? value : resolve(process.cwd(), value);
}

try {
  const requestedDestination = argumentValue("--pack-destination");
  const packDestination = requestedDestination ?? temporaryDirectory;
  mkdirSync(packDestination, { recursive: true });
  const packOutput = execFileSync(
    "npm",
    ["pack", "--json", "--pack-destination", packDestination],
    { cwd: packageRoot, encoding: "utf8" },
  );
  const packMetadata = parseJson(packOutput);
  const entries = Array.isArray(packMetadata)
    ? arrayValue(packMetadata)
    : Object.values(objectValue(packMetadata));
  const packed = objectValue(entries[0]);
  const version = stringValue(packed["version"]);
  assert.equal(stringValue(packed["name"]), "personalitygen");
  assert.equal(version, stringValue(packageMetadata["version"]));

  const paths = new Set(
    arrayValue(packed["files"]).map((file) => stringValue(objectValue(file)["path"])),
  );
  for (const required of [
    "LICENSE",
    "README.md",
    "dist/index.js",
    "dist/index.js.map",
    "dist/index.d.ts",
    "package.json",
  ]) {
    assert.ok(paths.has(required), `package is missing ${required}`);
  }
  assert.ok(
    [...paths].every(
      (path) =>
        path === "LICENSE" ||
        path === "README.md" ||
        path === "package.json" ||
        path.startsWith("dist/"),
    ),
    "package contains an unexpected file",
  );
  assert.ok(
    [...paths].every((path) => !path.endsWith(".d.ts.map")),
    "package contains an unusable declaration map",
  );

  const packagePath = join(packDestination, stringValue(packed["filename"]));
  const consumerDirectory = join(temporaryDirectory, "consumer");
  mkdirSync(consumerDirectory);
  writeFileSync(
    join(consumerDirectory, "package.json"),
    JSON.stringify({ private: true, type: "module" }),
  );
  execFileSync("npm", ["install", "--ignore-scripts", packagePath], {
    cwd: consumerDirectory,
    stdio: "pipe",
  });
  writeFileSync(
    join(consumerDirectory, "check.ts"),
    readFileSync(new URL("../tests/compat/consumer.ts", import.meta.url), "utf8"),
  );
  writeFileSync(
    join(consumerDirectory, "tsconfig.json"),
    JSON.stringify({
      extends: join(packageRoot, "tsconfig.json"),
      compilerOptions: {
        rootDir: ".",
        outDir: "build",
        noEmit: false,
        declaration: false,
        declarationMap: false,
        sourceMap: false,
        inlineSources: false,
      },
      include: ["check.ts"],
    }),
  );
  execFileSync(
    join(packageRoot, "node_modules", ".bin", "tsc6"),
    ["-p", "tsconfig.json"],
    {
      cwd: consumerDirectory,
      stdio: "pipe",
    },
  );

  execFileSync("node", ["build/check.js"], {
    cwd: consumerDirectory,
    stdio: "pipe",
  });

  const browserBundle = join(consumerDirectory, "browser-bundle.mjs");
  execFileSync(
    join(packageRoot, "node_modules", ".bin", "esbuild"),
    [
      "build/check.js",
      "--bundle",
      "--platform=browser",
      "--format=esm",
      `--outfile=${browserBundle}`,
    ],
    {
      cwd: consumerDirectory,
      stdio: "pipe",
    },
  );
  execFileSync("node", [browserBundle], { stdio: "pipe" });

  const installedPackageRoot = join(consumerDirectory, "node_modules/personalitygen");
  const installed = objectValue(
    parseJson(readFileSync(join(installedPackageRoot, "package.json"), "utf8")),
  );
  assert.equal(installed["dependencies"], undefined);
  const sourceMap = objectValue(
    parseJson(readFileSync(join(installedPackageRoot, "dist/index.js.map"), "utf8")),
  );
  const sources = arrayValue(sourceMap["sources"]);
  const sourcesContent = arrayValue(sourceMap["sourcesContent"]);
  assert.equal(
    sourcesContent.length,
    sources.length,
    "JavaScript source map does not embed its TypeScript sources",
  );
  for (const source of sourcesContent) {
    stringValue(source);
  }

  process.stdout.write(
    `${JSON.stringify({
      package: packagePath,
      version,
    })}\n`,
  );
} finally {
  rmSync(temporaryDirectory, { recursive: true, force: true });
}
