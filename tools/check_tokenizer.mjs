// Hold the browser tokenizer to the Python one, id for id.
//
//   python tools/dump_tokenizer_cases.py && node tools/check_tokenizer.mjs
//
// Fails loudly on the first disagreement: an encoding that is merely close is
// an encoding that prompts the model with a string it never trained on.

import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { Tokenizer } from "../web/js/tokenizer.js";

const MODELS = join(dirname(fileURLToPath(import.meta.url)), "..", "web", "models");
const fixtures = JSON.parse(readFileSync(join(MODELS, "tokenizer-cases.json"), "utf8"));

let checked = 0;
let failed = 0;
for (const [name, cases] of Object.entries(fixtures)) {
  const tok = Tokenizer.fromJSON(readFileSync(join(MODELS, name), "utf8"));
  for (const { text, ids, decoded } of cases) {
    const got = tok.encode(text);
    const label = JSON.stringify(text.length > 48 ? text.slice(0, 48) + "..." : text);
    if (got.length !== ids.length || got.some((id, i) => id !== ids[i])) {
      console.error(`FAIL encode ${name} ${label}\n  python ${ids}\n  js     ${got}`);
      failed++;
    } else if (tok.decode(got) !== decoded) {
      console.error(
        `FAIL decode ${name} ${label}\n  python ${JSON.stringify(decoded)}` +
          `\n  js     ${JSON.stringify(tok.decode(got))}`,
      );
      failed++;
    }
    checked++;
  }
}

console.log(`${checked - failed}/${checked} cases match`);
process.exit(failed ? 1 : 0);
