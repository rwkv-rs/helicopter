import { expect, test } from "@playwright/test";
import { existsSync, readdirSync, statSync } from "node:fs";
import path from "node:path";

const CLIENT_ROOT = process.cwd();
const REPO_SRC = path.resolve(CLIENT_ROOT, "..");
const BACKEND_API_ROUTES = path.join(REPO_SRC, "scoreboard-server", "scoreboard_server", "routes", "api");
const CLIENT_API = path.join(CLIENT_ROOT, "lib", "api");
const CLIENT_DTOS_API = path.join(CLIENT_ROOT, "lib", "dtos", "api");

function leafFiles(root: string, extension: string, ignoreRootIndex = false): string[] {
  const leaves: string[] = [];
  const walk = (directory: string) => {
    for (const name of readdirSync(directory)) {
      const full = path.join(directory, name);
      if (statSync(full).isDirectory()) {
        walk(full);
        continue;
      }
      if (!name.endsWith(extension)) continue;
      const relative = path.relative(root, full).split(path.sep).join("/");
      if (ignoreRootIndex && relative === "index.ts") continue;
      leaves.push(relative.slice(0, -extension.length));
    }
  };
  walk(root);
  return leaves.sort();
}

test("client API and DTO trees mirror backend API routes", () => {
  const backendLeaves = leafFiles(BACKEND_API_ROUTES, ".py")
    .filter((leaf) => !leaf.endsWith("/__init__") && leaf !== "__init__")
    .map((leaf) => leaf.replaceAll("__init__", "index"))
    .sort();
  const apiLeaves = leafFiles(CLIENT_API, ".ts", true);
  const dtoLeaves = leafFiles(CLIENT_DTOS_API, ".ts");

  expect(existsSync(path.join(CLIENT_ROOT, "lib", "api.ts"))).toBe(false);
  expect(existsSync(path.join(CLIENT_ROOT, "lib", "types.ts"))).toBe(false);
  expect(apiLeaves).toEqual(backendLeaves);
  expect(dtoLeaves).toEqual(backendLeaves);
});
