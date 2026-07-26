import { execFileSync } from "node:child_process";

for (const example of [
  "generate-npc.js",
  "project-big-five-to-abbf.js",
  "compare-characters.js",
  "select-npcs-by-pole.js",
]) {
  execFileSync("node", [`.example-dist/${example}`], { stdio: "pipe" });
}
