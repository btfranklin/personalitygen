import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

const packageMetadata = JSON.parse(
  readFileSync(new URL("../package.json", import.meta.url), "utf8"),
);
const releaseTag = process.env.RELEASE_TAG;

assert.ok(releaseTag, "RELEASE_TAG must be set");
assert.equal(
  releaseTag,
  `v${packageMetadata.version}`,
  "release tag must match the package version",
);
