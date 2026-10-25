import assert from "node:assert/strict";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { pathToFileURL } from "node:url";
import { Fixture, loadFixture } from "./fixture-data.js";

test("fixture reader checks nested values before use", () => {
  const fixture = new Fixture({
    count: 2,
    name: "sample",
    scores: [0.25, 0.75],
    labels: ["low", "high"],
    nested: { first: 0.5, second: 1 },
    rows: [{ name: "first" }, { name: "second" }],
  });
  assert.equal(fixture.number("count"), 2);
  assert.equal(fixture.string("name"), "sample");
  assert.deepEqual(fixture.numbers("scores"), [0.25, 0.75]);
  assert.deepEqual(fixture.strings("labels"), ["low", "high"]);
  assert.deepEqual(fixture.mapping("nested").numberValues(), {
    first: 0.5,
    second: 1,
  });
  assert.deepEqual(
    fixture.objects("rows").map((row) => row.stringValues()),
    [{ name: "first" }, { name: "second" }],
  );
  assert.deepEqual(fixture.mapping("nested").keys(), ["first", "second"]);
});

test("fixture reader rejects missing keys and incorrect scalar types", () => {
  for (const value of [null, [], "text", 2, true]) {
    assert.throws(() => new Fixture(value), TypeError);
  }
  const fixture = new Fixture({
    text: "0.5",
    bool: true,
    number: 0.5,
    null: null,
    infinity: Infinity,
  });
  assert.throws(() => fixture.value("missing"), /Missing fixture key/);
  assert.throws(() => fixture.value("toString"), /Missing fixture key/);
  for (const key of ["text", "bool", "null", "infinity"]) {
    assert.throws(() => fixture.number(key), TypeError);
  }
  assert.throws(() => fixture.string("number"), TypeError);
  assert.throws(() => fixture.mapping("null"), TypeError);
  assert.throws(() => fixture.numberValues(), TypeError);
  assert.throws(() => fixture.stringValues(), TypeError);
});

test("fixture reader checks every array item", () => {
  const fixture = new Fixture({
    scalar: 0.5,
    numbers: [0.5, "0.75"],
    strings: ["low", 0.75],
    objects: [{ score: 0.5 }, null],
  });
  assert.throws(() => fixture.numbers("scalar"), TypeError);
  assert.throws(() => fixture.numbers("numbers"), TypeError);
  assert.throws(() => fixture.strings("strings"), TypeError);
  assert.throws(() => fixture.objects("objects"), TypeError);
});

test("fixture loader checks the parsed JSON root", () => {
  const directory = mkdtempSync(join(tmpdir(), "personalitygen-fixture-"));
  const path = join(directory, "fixture.json");
  try {
    writeFileSync(path, '{"score":0.5}');
    assert.equal(loadFixture(pathToFileURL(path)).number("score"), 0.5);
    writeFileSync(path, "null");
    assert.throws(() => loadFixture(pathToFileURL(path)), TypeError);
    writeFileSync(path, "{");
    assert.throws(() => loadFixture(pathToFileURL(path)), SyntaxError);
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
});
