import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const temporaryDirectory = mkdtempSync(join(tmpdir(), "personalitygen-npm-"));

try {
  const packOutput = execFileSync(
    "npm",
    ["pack", "--json", "--pack-destination", temporaryDirectory],
    { encoding: "utf8" },
  );
  const packMetadata = JSON.parse(packOutput);
  const packed = Array.isArray(packMetadata)
    ? packMetadata[0]
    : Object.values(packMetadata)[0];
  assert.ok(packed, "npm pack did not return package metadata");
  assert.equal(packed.name, "personalitygen");
  assert.equal(packed.version, "0.3.0");

  const paths = new Set(packed.files.map(({ path }) => path));
  for (const required of [
    "LICENSE",
    "README.md",
    "dist/index.js",
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

  const packagePath = join(temporaryDirectory, packed.filename);
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
      const profile = AdaptiveBifurcatedProfile.fromBigFive(personality.traitConfiguration);
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
      AdaptiveBifurcatedProfile.fromBigFive(personality.traitConfiguration);
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
    join(process.cwd(), "node_modules", ".bin", "tsc6"),
    ["-p", "tsconfig.json"],
    {
      cwd: consumerDirectory,
      stdio: "pipe",
    },
  );

  const installed = JSON.parse(
    readFileSync(
      join(consumerDirectory, "node_modules/personalitygen/package.json"),
      "utf8",
    ),
  );
  assert.equal(installed.dependencies, undefined);
} finally {
  rmSync(temporaryDirectory, { recursive: true, force: true });
}
