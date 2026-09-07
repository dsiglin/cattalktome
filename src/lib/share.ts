export type ShareResult = 'shared' | 'saved' | 'cancelled';

export interface ShareRequest {
  blob: Blob;
  filename: string;
  title: string;
  text: string;
  navigator?: Navigator;
  saveFallback?: (blob: Blob, filename: string) => void;
}

/** Test whether this browser can put a file into the native sharesheet. */
export function supportsFileShare(nav: Navigator = globalThis.navigator): boolean {
  if (!nav || typeof nav.share !== 'function' || typeof nav.canShare !== 'function') return false;
  try {
    const probe = new File([new Uint8Array([0])], 'probe.png', { type: 'image/png' });
    return nav.canShare({ files: [probe] });
  } catch {
    return false;
  }
}

/** Save the image through a temporary link. This is the fallback on desktop browsers. */
export function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  // Release the object URL once the browser starts the save.
  setTimeout(() => URL.revokeObjectURL(url), 10_000);
}

export async function shareSticker(request: ShareRequest): Promise<ShareResult> {
  const nav = request.navigator ?? globalThis.navigator;
  const save = request.saveFallback ?? downloadBlob;
  const file = new File([request.blob], request.filename, { type: request.blob.type || 'image/png' });

  if (!supportsFileShare(nav)) {
    save(request.blob, request.filename);
    return 'saved';
  }

  try {
    await nav.share({ files: [file], title: request.title, text: request.text });
    return 'shared';
  } catch (error) {
    // The user closing the sharesheet is not a failure.
    if (error instanceof Error && error.name === 'AbortError') return 'cancelled';
    save(request.blob, request.filename);
    return 'saved';
  }
}
