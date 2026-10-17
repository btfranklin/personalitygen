import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";

for (const example of [
  "generate-npc.js",
  "project-big-five-to-abbf.js",
  "compare-characters.js",
  "select-npcs-by-pole.js",
  "rpg-party.js",
  "mystery-casting.js",
  "test-population.js",
]) {
  const path = fileURLToPath(
    new URL(`../../../examples/.example-dist/${example}`, import.meta.url),
  );
  execFileSync("node", [path], { stdio: "pipe" });
}

execFileSync(
  "node",
  [
    "--test",
    fileURLToPath(
      new URL("../../../examples/.example-dist/rpg-rules.test.js", import.meta.url),
    ),
    fileURLToPath(
      new URL("../../../examples/.example-dist/mystery-rules.test.js", import.meta.url),
    ),
    fileURLToPath(
      new URL(
        "../../../examples/.example-dist/population-rules.test.js",
        import.meta.url,
      ),
    ),
  ],
  { stdio: "inherit" },
);
