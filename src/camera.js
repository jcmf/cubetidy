// Camera access via getUserMedia. Requires https or localhost.

export async function startCamera(videoEl) {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error('Camera API not available. Use a modern browser over https or localhost.');
  }

  const tryConstraints = [
    { video: { facingMode: { ideal: 'environment' }, width: { ideal: 1280 }, height: { ideal: 720 } } },
    { video: { facingMode: 'environment' } },
    { video: true },
  ];

  let lastErr;
  for (const constraints of tryConstraints) {
    try {
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      videoEl.srcObject = stream;
      await videoEl.play();
      // Wait until dimensions are known.
      if (!videoEl.videoWidth) {
        await new Promise((res) => videoEl.addEventListener('loadedmetadata', res, { once: true }));
      }
      return stream;
    } catch (err) {
      lastErr = err;
    }
  }
  throw lastErr ?? new Error('Could not start camera');
}

export function stopCamera(videoEl) {
  const stream = videoEl.srcObject;
  if (stream) {
    for (const track of stream.getTracks()) track.stop();
    videoEl.srcObject = null;
  }
}
