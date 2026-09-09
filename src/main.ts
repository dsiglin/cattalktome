import './style.css';
import { layoutSticker, layoutImageSticker, drawSticker, drawImageSticker } from './lib/sticker';
import { pickStickerFor } from './lib/genz-stickers';
import { shareSticker, supportsFileShare } from './lib/share';
import { requestDeeperRead, type DeeperReading } from './lib/deeper-read';

/** The largest photo the app draws. This keeps the shared file small. */
const MAX_RENDER_SIDE = 1400;

/** The deeper-read server. See server/README.md for what it does and why. */
const DEEPER_READ_API = 'https://cat-talk-to-me-api-982825418658.us-central1.run.app';

const THINKING_LINES = [
  'Sitting with your cat for a moment…',
  'Sending it over for a closer look…',
  'Reading ears, eyes, and muzzle shape…',
  'Choosing my words carefully…',
];

const $ = <T extends HTMLElement>(id: string) => {
  const el = document.getElementById(id);
  if (!el) throw new Error(`missing element #${id}`);
  return el as T;
};

const panels = {
  pick: $('panel-pick'),
  thinking: $('panel-thinking'),
  result: $('panel-result'),
  error: $('panel-error'),
};

const fileInput = $<HTMLInputElement>('file');
const dropzone = $<HTMLLabelElement>('dropzone');
const canvas = $<HTMLCanvasElement>('result-canvas');
const shareButton = $<HTMLButtonElement>('share');
const toast = $('toast');

let currentLabel = 'Cat';
let currentBlurb = '';
let currentFeelingId = '';

function show(which: keyof typeof panels) {
  for (const [name, panel] of Object.entries(panels)) panel.hidden = name !== which;
}

function say(message: string) {
  toast.textContent = message;
  if (message) window.setTimeout(() => { if (toast.textContent === message) toast.textContent = ''; }, 4500);
}

/** Draw the photo onto a canvas, capped in size, and hand back the pixels. */
function drawPhoto(bitmap: ImageBitmap): CanvasRenderingContext2D {
  const scale = Math.min(1, MAX_RENDER_SIDE / Math.max(bitmap.width, bitmap.height));
  canvas.width = Math.round(bitmap.width * scale);
  canvas.height = Math.round(bitmap.height * scale);
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) throw new Error('this browser cannot draw on a canvas');
  ctx.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
  return ctx;
}

function renderReading(reading: DeeperReading) {
  const { feeling, runnerUpLabel, confidence, evidence, reliable } = reading;

  $('verdict-emoji').textContent = feeling.emoji;
  $('verdict-label').textContent = feeling.label;
  $('verdict-blurb').textContent = feeling.blurb;
  $('cue').textContent = feeling.cue;

  const list = $('evidence');
  const lines = reliable
    ? evidence
    : [...evidence, 'I am not sure I found the face - it may be at an odd angle, or I may have picked a patch of fur. Hold this one loosely.'];
  list.replaceChildren(...lines.map((line) => {
    const li = document.createElement('li');
    li.textContent = line;
    return li;
  }));

  const percent = Math.round(confidence * 100);
  $('meter-fill').style.width = `${percent}%`;
  $('meter').setAttribute('aria-label', `confidence ${percent} percent`);
  $('meter-caption').textContent =
    `${percent}% sure. My second guess was “${runnerUpLabel}”.`;

  currentLabel = `${feeling.label} ${feeling.emoji}`;
  currentBlurb = `My cat is ${feeling.label.toLowerCase()}. ${feeling.blurb}`;
  currentFeelingId = feeling.id;
}

/** Load an image, resolving null instead of rejecting - a missing bonus
 * sticker should never break the actual reading. */
function loadImage(src: string): Promise<HTMLImageElement | null> {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => resolve(null);
    img.src = src;
  });
}

/** Put the feeling sticker on the photo. */
function stampLabel(ctx: CanvasRenderingContext2D, label: string) {
  const measure = (text: string, fontSize: number) => {
    ctx.save();
    ctx.font = `700 ${fontSize}px "Nunito", ui-rounded, system-ui, sans-serif`;
    const width = ctx.measureText(text).width;
    ctx.restore();
    return width;
  };
  drawSticker(ctx, layoutSticker(canvas.width, canvas.height, label, measure), label);
}

/** Fetch a bonus sticker that suits the feeling and put it on the photo,
 * opposite the feeling label. Decorative only - failures are silent. */
async function stampBonusSticker(ctx: CanvasRenderingContext2D, feelingId: string) {
  const sticker = pickStickerFor(feelingId);
  const src = `${import.meta.env.BASE_URL}stickers/${sticker.file}`;
  const image = await loadImage(src);
  if (!image) return;

  const aspect = image.naturalWidth / image.naturalHeight;
  drawImageSticker(ctx, image, layoutImageSticker(canvas.width, canvas.height, aspect, sticker.file));
}

const ERROR_COPY: Record<string, string> = {
  'no-cat': "I could not find a cat's face in that photo. I need the face turned mostly toward the camera - a side profile hides the eyes and ears I read. Try one where the cat is looking at you?",
  'rate-limited': 'This little server has a visitor limit and it has been reached for now - try again a bit later.',
  'too-large': 'That photo is too large to send. Try a smaller one?',
  'bad-image': 'That file does not look like a picture the reader can open.',
  network: 'Could not reach the reader - check your connection and try again.',
  unknown: 'That hit a snag. Try again in a moment?',
};

async function handleFile(file: File) {
  if (!file.type.startsWith('image/')) {
    $('error-text').textContent = 'That file is not a picture. Try a photo instead?';
    show('error');
    return;
  }

  $('thinking-line').textContent = THINKING_LINES[0];
  show('thinking');

  let step = 1;
  const ticker = window.setInterval(() => {
    $('thinking-line').textContent = THINKING_LINES[step % THINKING_LINES.length];
    step++;
  }, 900);

  try {
    const bitmap = await createImageBitmap(file);
    const ctx = drawPhoto(bitmap);
    bitmap.close?.();

    // A clean copy of the photo, before any sticker lands on it - what gets
    // uploaded, and what the sticker is later drawn on top of.
    const clean = document.createElement('canvas');
    clean.width = canvas.width;
    clean.height = canvas.height;
    clean.getContext('2d')?.drawImage(canvas, 0, 0);

    const blob = await new Promise<Blob | null>((resolve) => clean.toBlob(resolve, 'image/png'));
    if (!blob) throw new Error('could not make an image to send');

    const result = await requestDeeperRead(blob, DEEPER_READ_API);
    window.clearInterval(ticker);

    if (!result.ok) {
      $('error-text').textContent = ERROR_COPY[result.reason] ?? result.message;
      show('error');
      return;
    }

    renderReading(result.reading);
    stampLabel(ctx, currentLabel);
    void stampBonusSticker(ctx, currentFeelingId);
    show('result');
  } catch (error) {
    window.clearInterval(ticker);
    console.error(error);
    $('error-text').textContent = 'That photo would not open. Try another one?';
    show('error');
  }
}

/* ------------------------------- wiring -------------------------------- */

fileInput.addEventListener('change', () => {
  const file = fileInput.files?.[0];
  if (file) void handleFile(file);
});

for (const event of ['dragenter', 'dragover'] as const) {
  dropzone.addEventListener(event, (e) => {
    e.preventDefault();
    dropzone.classList.add('is-over');
  });
}
for (const event of ['dragleave', 'drop'] as const) {
  dropzone.addEventListener(event, () => dropzone.classList.remove('is-over'));
}
dropzone.addEventListener('drop', (e) => {
  e.preventDefault();
  const file = e.dataTransfer?.files?.[0];
  if (file) void handleFile(file);
});

shareButton.addEventListener('click', async () => {
  shareButton.disabled = true;
  try {
    const blob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, 'image/png'));
    if (!blob) throw new Error('could not make an image');

    const result = await shareSticker({
      blob,
      filename: `cat-${currentLabel.split(' ')[0].toLowerCase()}.png`,
      title: 'Cat, Talk To Me',
      text: currentBlurb,
    });

    if (result === 'saved') say('Saved to your device. Send it to whoever needs to know.');
    if (result === 'cancelled') say('No rush. It is still here.');
  } catch (error) {
    console.error(error);
    say('That did not work. Try again?');
  } finally {
    shareButton.disabled = false;
  }
});

const restart = () => {
  fileInput.value = '';
  toast.textContent = '';
  show('pick');
};
$('again').addEventListener('click', restart);
$('retry').addEventListener('click', restart);

// Tell the truth about what the button will do on this browser.
if (!supportsFileShare()) $('share-label').textContent = 'Save this';

show('pick');
