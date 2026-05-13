const PI_URL = process.env.PI_CAMERA_URL ?? 'http://localhost:8765'

export async function GET() {
  try {
    // 1. Use a standard AbortController for the fetch phase
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 5000);

    const upstream = await fetch(`${PI_URL}/camera/stream`, {
      cache: 'no-store',
      signal: controller.signal,
    });

    // 2. Clear the timeout as soon as headers are received
    clearTimeout(timeoutId);

    if (!upstream.ok || !upstream.body) {
      return new Response('Pi camera unavailable (Status Error)', { status: 503 });
    }

    // 3. Return the stream. Next.js/Node will pipe the body.
    // The connection will stay open until the user closes the tab or the Pi stops sending.
    return new Response(upstream.body, {
      headers: {
        'Content-Type': upstream.headers.get('Content-Type') ?? 'multipart/x-mixed-replace; boundary=frame',
        'Cache-Control': 'no-cache, no-store, must-revalidate',
        'Pragma': 'no-cache',
        'X-Accel-Buffering': 'no', // Critical for Nginx/Vercel proxies
        'Connection': 'keep-alive',
      },
    });
  } catch (err: any) {
    if (err.name === 'AbortError') {
      return new Response('Pi camera connection timed out', { status: 504 });
    }
    return new Response('Pi camera unavailable', { status: 503 });
  }
}