import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

import { objectValue, parseJson, stringValue } from "./json-data.mjs";

const packageMetadata = objectValue(
  parseJson(readFileSync(new URL("../package.json", import.meta.url), "utf8")),
);
const releaseTag = process.env["RELEASE_TAG"];

assert.ok(releaseTag, "RELEASE_TAG must be set");
assert.equal(
  releaseTag,
  `v${stringValue(packageMetadata["version"])}`,
  "release tag must match the package version",
);
