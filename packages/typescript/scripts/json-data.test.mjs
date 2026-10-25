import assert from "node:assert/strict";
import { test } from "node:test";
import { arrayValue, objectValue, parseJson, stringValue } from "./json-data.mjs";

test("metadata fields need type checks before use", () => {
  const metadata = objectValue(parseJson('{"version":"1.0.0","files":["index.js"]}'));
  assert.equal(stringValue(metadata["version"]), "1.0.0");
  assert.deepEqual(arrayValue(metadata["files"]).map(stringValue), ["index.js"]);
  assert.throws(() => stringValue(metadata["missing"]), TypeError);
});

test("metadata readers reject incorrect shapes", () => {
  for (const value of [null, [], "text", 1, true]) {
    assert.throws(() => objectValue(value), TypeError);
  }
  for (const value of [null, {}, "text", 1, true]) {
    assert.throws(() => arrayValue(value), TypeError);
  }
  for (const value of [null, {}, [], 1, true]) {
    assert.throws(() => stringValue(value), TypeError);
  }
  assert.throws(() => parseJson("{"), SyntaxError);
});
