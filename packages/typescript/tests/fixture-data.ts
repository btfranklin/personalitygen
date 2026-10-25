import { readFileSync } from "node:fs";

function isMapping(value: unknown): value is Readonly<Record<string, unknown>> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function items(value: unknown): readonly unknown[] {
  if (!Array.isArray(value)) {
    throw new TypeError("Expected a JSON array");
  }
  return value;
}

function number(value: unknown): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new TypeError("Expected a finite JSON number");
  }
  return value;
}

function string(value: unknown): string {
  if (typeof value !== "string") {
    throw new TypeError("Expected a JSON string");
  }
  return value;
}

export class Fixture {
  readonly #data: Readonly<Record<string, unknown>>;

  constructor(value: unknown) {
    if (!isMapping(value)) {
      throw new TypeError("Expected a JSON object");
    }
    this.#data = value;
  }

  value(key: string): unknown {
    if (!Object.hasOwn(this.#data, key)) {
      throw new TypeError(`Missing fixture key: ${key}`);
    }
    return this.#data[key];
  }

  mapping(key: string): Fixture {
    return new Fixture(this.value(key));
  }

  objects(key: string): readonly Fixture[] {
    return items(this.value(key)).map((value) => new Fixture(value));
  }

  number(key: string): number {
    return number(this.value(key));
  }

  numbers(key: string): readonly number[] {
    return items(this.value(key)).map(number);
  }

  string(key: string): string {
    return string(this.value(key));
  }

  strings(key: string): readonly string[] {
    return items(this.value(key)).map(string);
  }

  numberValues(): Readonly<Record<string, number>> {
    return Object.fromEntries(
      Object.entries(this.#data).map(([key, value]): [string, number] => [
        key,
        number(value),
      ]),
    );
  }

  stringValues(): Readonly<Record<string, string>> {
    return Object.fromEntries(
      Object.entries(this.#data).map(([key, value]): [string, string] => [
        key,
        string(value),
      ]),
    );
  }

  keys(): readonly string[] {
    return Object.keys(this.#data);
  }
}

export function loadFixture(path: URL): Fixture {
  const value: unknown = JSON.parse(readFileSync(path, "utf8"));
  return new Fixture(value);
}
