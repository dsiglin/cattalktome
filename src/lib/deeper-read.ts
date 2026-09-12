/**
 * The optional "deeper read" - uploads the photo to a small server that
 * measures real cat facial geometry (ear-base angle, head tilt, muzzle
 * ratio) instead of whole-photo light and shadow. Opt-in only: this is
 * the one place in the app where a photo leaves the device.
 */

export interface DeeperReading {
  feeling: { id: string; label: string; emoji: string; blurb: string; cue: string };
  runnerUpLabel: string;
  confidence: number;
  reliable: boolean;
  evidence: string[];
  eyes: EyeReading;
}

/** The cat's own pupils, 0..1 dark fraction. See server/app/eyes.py. `usable`
 * is false when the pupils could not be read - the gauge must show that
 * honestly rather than guess a position. */
export interface EyeReading {
  pupilDilation: number;
  usable: boolean;
}

export type DeeperReadFailureReason =
  | 'no-cat' | 'rate-limited' | 'too-large' | 'bad-image' | 'network' | 'unknown';

export type DeeperReadResult =
  | { ok: true; reading: DeeperReading }
  | { ok: false; reason: DeeperReadFailureReason; message: string };

const REASON_BY_STATUS: Record<number, DeeperReadFailureReason> = {
  400: 'bad-image',
  413: 'too-large',
  422: 'no-cat',
  429: 'rate-limited',
};

const FRIENDLY_MESSAGE: Record<DeeperReadFailureReason, string> = {
  'no-cat': "I could not find a cat's face in this photo, so the deeper read has nothing to measure.",
  'rate-limited': 'This little server has a visitor limit and it has been reached - try again a bit later.',
  'too-large': 'That photo is too large for the deeper read to accept.',
  'bad-image': 'That file does not look like an image the deeper read can open.',
  network: 'Could not reach the deeper-read server - check your connection and try again.',
  unknown: 'The deeper read hit a snag on its end. Try again in a moment.',
};

export async function requestDeeperRead(
  photo: Blob,
  apiBaseUrl: string,
  fetchImpl: typeof fetch = fetch,
): Promise<DeeperReadResult> {
  const form = new FormData();
  form.append('photo', photo, 'photo.png');

  let response: Response;
  try {
    response = await fetchImpl(`${apiBaseUrl}/analyze`, { method: 'POST', body: form });
  } catch {
    return { ok: false, reason: 'network', message: FRIENDLY_MESSAGE.network };
  }

  if (!response.ok) {
    const reason = REASON_BY_STATUS[response.status] ?? 'unknown';
    let serverDetail = '';
    try {
      const body = await response.json();
      if (typeof body?.detail === 'string') serverDetail = body.detail;
    } catch {
      // Body wasn't JSON - fall back to the friendly message below.
    }
    return { ok: false, reason, message: serverDetail || FRIENDLY_MESSAGE[reason] };
  }

  const body = await response.json();
  return {
    ok: true,
    reading: {
      feeling: body.feeling,
      runnerUpLabel: body.runner_up.label,
      confidence: body.confidence,
      reliable: body.reliable,
      evidence: body.evidence,
      eyes: {
        pupilDilation: body.eyes?.pupil_dilation ?? 0.5,
        usable: body.eyes?.usable ?? false,
      },
    },
  };
}
