# Cat Talk To Me — Build Plan

## Goal
A static web app. The user uploads a cat photo. The app reads the photo and
reports a plausible feeling. The app draws that feeling on the photo as a
sticker. The user shares the result through the native sharesheet.

## Constraints
- GitHub Pages hosts the app. No server exists. All work runs in the browser.
- No photo leaves the device.
- The design must feel cozy, warm, and human.

## Honesty rule
No browser model reads cat emotion reliably. The app measures real image
features. The app maps those features onto documented feline cue families
(ears, pupils, whiskers, posture). The app labels the result a considered
guess. The UI states this plainly.

## Architecture
| Module | Responsibility | Pure? |
|---|---|---|
| `src/lib/features.ts` | Turn `ImageData` into numeric features. | Yes |
| `src/lib/feelings.ts` | Turn features into a ranked feeling. | Yes |
| `src/lib/sticker.ts` | Compute sticker geometry, then draw it. | Layout is pure |
| `src/lib/share.ts` | Share a blob, or fall back to download. | Injected navigator |
| `src/lib/cat-check.ts` | Optional MobileNet cat verification. | No, lazy |
| `src/main.ts` | Wire the UI. | No |

## TDD steps
1. RED: write `features.test.ts` against synthetic `ImageData`. GREEN: build `features.ts`.
2. RED: write `feelings.test.ts` for scoring, determinism, and coverage. GREEN: build `feelings.ts`.
3. RED: write `sticker.test.ts` for layout maths. GREEN: build `sticker.ts`.
4. RED: write `share.test.ts` with a fake navigator. GREEN: build `share.ts`.
5. RED: write `integration.test.ts` that decodes `test/fixtures/cat.jpg`. GREEN: build the pipeline.
6. Build the UI and the cozy design.
7. Verify in a real browser with the real cat photo.
8. Add the GitHub Pages workflow.

## Definition of done
- All tests pass.
- The real cat photo produces a feeling, a sticker, and a share action.
- The browser check shows the finished sticker.
