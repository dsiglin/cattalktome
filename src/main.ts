import './style.css';
import { analysePhoto, type Analysis } from './lib/analyse';
import { layoutSticker, drawSticker } from './lib/sticker';
import { shareSticker, supportsFileShare } from './lib/share';
import { verifyCat } from './lib/cat-check';

/** The largest photo the app draws. This keeps the shared file small. */
const MAX_RENDER_SIDE = 1400;

const THINKING_LINES = [
  'Sitting with your cat for a moment…',
  'Reading the light and the shadows…',
  'Looking at how still they are…',
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

function renderReading(analysis: Analysis) {
  const { feeling, runnerUp, confidence, evidence } = analysis;

  $('verdict-emoji').textContent = feeling.emoji;
  $('verdict-label').textContent = feeling.label;
  $('verdict-blurb').textContent = feeling.blurb;
  $('cue').textContent = feeling.cue;

  const list = $('evidence');
  list.replaceChildren(...evidence.map((line) => {
    const li = document.createElement('li');
    li.textContent = line;
    return li;
  }));

  const percent = Math.round(confidence * 100);
  $('meter-fill').style.width = `${percent}%`;
  $('meter').setAttribute('aria-label', `confidence ${percent} percent`);
  $('meter-caption').textContent =
    `${percent}% sure. My second guess was “${runnerUp.label}”.`;

  currentLabel = `${feeling.label} ${feeling.emoji}`;
  currentBlurb = `My cat is ${feeling.label.toLowerCase()}. ${feeling.blurb}`;
}

/** Put the sticker on the photo, measuring the text with the real font. */
function stampSticker(ctx: CanvasRenderingContext2D, label: string) {
  const measure = (text: string, fontSize: number) => {
    ctx.save();
    ctx.font = `700 ${fontSize}px "Nunito", ui-rounded, system-ui, sans-serif`;
    const width = ctx.measureText(text).width;
    ctx.restore();
    return width;
  };
  drawSticker(ctx, layoutSticker(canvas.width, canvas.height, label, measure), label);
}

async function runCatCheck(photoOnly: HTMLCanvasElement) {
  const note = $('catcheck');
  const verdict = await verifyCat(photoOnly);
  if (!verdict) return;

  const icon = document.createElement('span');
  icon.className = 'ico';
  icon.setAttribute('aria-hidden', 'true');
  const words = document.createElement('span');

  if (verdict.isBigCat) {
    icon.textContent = '\u{1F981}';
    words.textContent = `That reads as a ${verdict.label} to me. Bold choice. The feeling still stands.`;
  } else if (verdict.isCat) {
    icon.textContent = '\u2705';
    words.textContent = `Confirmed cat (${verdict.label}).`;
  } else {
    icon.textContent = '\u{1F914}';
    words.textContent = `I am not certain that is a cat \u2014 it looks more like a ${verdict.label}. I read it anyway.`;
  }
  note.replaceChildren(icon, words);
  note.hidden = false;
}

async function handleFile(file: File) {
  if (!file.type.startsWith('image/')) {
    $('error-text').textContent = 'That file is not a picture. Try a photo instead?';
    show('error');
    return;
  }

  $('catcheck').hidden = true;
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

    // Keep a clean copy of the photo for the classifier, before the sticker lands.
    const clean = document.createElement('canvas');
    clean.width = canvas.width;
    clean.height = canvas.height;
    clean.getContext('2d')?.drawImage(canvas, 0, 0);

    const analysis = analysePhoto(ctx.getImageData(0, 0, canvas.width, canvas.height));
    renderReading(analysis);
    stampSticker(ctx, currentLabel);

    window.clearInterval(ticker);
    show('result');
    void runCatCheck(clean);
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
