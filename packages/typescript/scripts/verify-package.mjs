import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { isAbsolute, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const packageRoot = fileURLToPath(new URL("..", import.meta.url));
const packageMetadata = JSON.parse(
  readFileSync(new URL("../package.json", import.meta.url), "utf8"),
);
const temporaryDirectory = mkdtempSync(join(tmpdir(), "personalitygen-npm-"));

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
  const packMetadata = JSON.parse(packOutput);
  const packed = Array.isArray(packMetadata)
    ? packMetadata[0]
    : Object.values(packMetadata)[0];
  assert.ok(packed, "npm pack did not return package metadata");
  assert.equal(packed.name, "personalitygen");
  assert.equal(packed.version, packageMetadata.version);

  const paths = new Set(packed.files.map(({ path }) => path));
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

  const packagePath = join(packDestination, packed.filename);
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
    join(consumerDirectory, "check.mjs"),
    `
      import { AdaptiveBifurcatedProfile, BigFivePersonality, LifeStage } from "personalitygen";
      const personality = BigFivePersonality.random(LifeStage.Adult);
      const profile = AdaptiveBifurcatedProfile.fromBigFive(personality.traits);
      if (profile.vector.length !== 5) process.exit(1);
    `,
  );
  execFileSync("node", ["check.mjs"], {
    cwd: consumerDirectory,
    stdio: "pipe",
  });
  writeFileSync(
    join(consumerDirectory, "check.ts"),
    `
      import { AdaptiveBifurcatedProfile, BigFivePersonality, LifeStage } from "personalitygen";
      const personality = BigFivePersonality.random(LifeStage.Adult);
      AdaptiveBifurcatedProfile.fromBigFive(personality.traits);
    `,
  );
  writeFileSync(
    join(consumerDirectory, "tsconfig.json"),
    JSON.stringify({
      compilerOptions: {
        target: "ES2022",
        module: "NodeNext",
        moduleResolution: "NodeNext",
        lib: ["ES2022"],
        types: [],
        strict: true,
        noEmit: true,
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

  writeFileSync(
    join(consumerDirectory, "browser-entry.js"),
    `
      import { AdaptiveBifurcatedProfile } from "personalitygen";
      const profile = AdaptiveBifurcatedProfile.random({
        rng: { uniform: (minimum, maximum) => minimum + (maximum - minimum) / 2 },
      });
      if (profile.vector.length !== 5) throw new Error("unexpected vector length");
    `,
  );
  const browserBundle = join(consumerDirectory, "browser-bundle.mjs");
  execFileSync(
    join(packageRoot, "node_modules", ".bin", "esbuild"),
    [
      "browser-entry.js",
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
  const installed = JSON.parse(
    readFileSync(join(installedPackageRoot, "package.json"), "utf8"),
  );
  assert.equal(installed.dependencies, undefined);
  const sourceMap = JSON.parse(
    readFileSync(join(installedPackageRoot, "dist/index.js.map"), "utf8"),
  );
  assert.ok(
    Array.isArray(sourceMap.sourcesContent) &&
      sourceMap.sourcesContent.length === sourceMap.sources.length &&
      sourceMap.sourcesContent.every((source) => typeof source === "string"),
    "JavaScript source map does not embed its TypeScript sources",
  );

  process.stdout.write(
    `${JSON.stringify({
      package: packagePath,
      version: packed.version,
    })}\n`,
  );
} finally {
  rmSync(temporaryDirectory, { recursive: true, force: true });
}
