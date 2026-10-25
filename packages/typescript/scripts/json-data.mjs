/** @param {unknown} value
 * @returns {value is Record<string, unknown>}
 */
function isObject(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

/** @param {unknown} value
 * @returns {Record<string, unknown>}
 */
export function objectValue(value) {
  if (!isObject(value)) {
    throw new TypeError("Expected an object.");
  }
  return value;
}

/** @param {unknown} value
 * @returns {readonly unknown[]}
 */
export function arrayValue(value) {
  if (!Array.isArray(value)) {
    throw new TypeError("Expected an array.");
  }
  return value;
}

/** @param {unknown} value
 * @returns {string}
 */
export function stringValue(value) {
  if (typeof value !== "string") {
    throw new TypeError("Expected a string.");
  }
  return value;
}

/** @param {string} text
 * @returns {unknown}
 */
export function parseJson(text) {
  return JSON.parse(text);
}
