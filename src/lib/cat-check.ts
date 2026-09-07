/**
 * Optional confirmation that the photo holds a cat.
 *
 * This uses MobileNet, a real image classifier, loaded only after a photo
 * arrives. The app works fully without it. If the model or the network fails,
 * this returns null and the app says nothing about it.
 */

/** ImageNet labels that mean a domestic cat. */
const HOUSE_CATS = ['tabby', 'tiger cat', 'persian cat', 'siamese cat', 'egyptian cat'];

/** ImageNet labels that mean a big cat. Worth a friendly note. */
const BIG_CATS = ['cougar', 'lynx', 'leopard', 'snow leopard', 'jaguar', 'lion', 'tiger', 'cheetah'];

export interface CatVerdict {
  isCat: boolean;
  isBigCat: boolean;
  label: string;
  score: number;
}

let modelPromise: Promise<{ classify: (i: HTMLImageElement | HTMLCanvasElement) => Promise<Array<{ className: string; probability: number }>> }> | null = null;

async function loadModel() {
  if (!modelPromise) {
    modelPromise = (async () => {
      await import('@tensorflow/tfjs');
      const mobilenet = await import('@tensorflow-models/mobilenet');
      return mobilenet.load({ version: 2, alpha: 0.5 });
    })();
  }
  return modelPromise;
}

export async function verifyCat(source: HTMLCanvasElement): Promise<CatVerdict | null> {
  try {
    const model = await loadModel();
    const predictions = await model.classify(source);
    if (predictions.length === 0) return null;

    let best: CatVerdict | null = null;
    for (const p of predictions) {
      // A prediction can name several synonyms, so test each one.
      for (const name of p.className.toLowerCase().split(',').map((s) => s.trim())) {
        const house = HOUSE_CATS.includes(name);
        const big = BIG_CATS.includes(name);
        if (house || big) {
          best = { isCat: true, isBigCat: big, label: name, score: p.probability };
          break;
        }
      }
      if (best) break;
    }

    return best ?? {
      isCat: false,
      isBigCat: false,
      label: predictions[0].className.split(',')[0],
      score: predictions[0].probability,
    };
  } catch {
    // The model is a bonus, never a requirement.
    return null;
  }
}
