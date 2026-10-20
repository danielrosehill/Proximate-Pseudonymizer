# Idea

Daniel, 2026-10-21 (verbatim):

> is there a model for obfuscation on HF? text to text. input: text with identifiable details
> (say: "I was born in Dublin"). output: text with obfuscated details: "I was born in London".

> Can we create a simpler fine tune ? The goal would be to create obfuscated text that is
> proximate. If I share a story as Daniel in Jerusalem it would change Daniel to David and
> Jerusalem to Tel Aviv. Text in -> text out, direct replacement.

> the only rule besides similarity is that the obfuscations should stay consistent within the
> same body of text.

## Contract

- **Input:** any free text (story, email, prompt to an AI, voice-note transcript).
- **Output:** the same text, character for character, except that each identifying span is
  replaced by a *proximate* substitute: same kind, same gender/culture for names, a
  similar-sized city in the same country for places, a nearby date in the same format, a
  same-format number.
- **Consistency:** one original → one substitute everywhere in the text, including partial
  forms (`Daniel Rosen` → `David Cohen` implies `Daniel` → `David`).
- **Not a redactor:** no `[NAME]` tags, no mapping output. Plain text in, plain text out.
