const PI_URL = process.env.PI_CAMERA_URL ?? 'http://localhost:8765'

export async function GET() {
  try {
    const upstream = await fetch(`${PI_URL}/camera/stream`, {
      cache: 'no-store',
      signal: AbortSignal.timeout(5000),
    })
    if (!upstream.ok || !upstream.body) {
      return new Response('Pi camera unavailable', { status: 503 })
    }
    return new Response(upstream.body, {
      headers: {
        'Content-Type': upstream.headers.get('Content-Type') ?? 'multipart/x-mixed-replace; boundary=frame',
        'Cache-Control': 'no-cache, no-store',
        'X-Accel-Buffering': 'no',
      },
    })
  } catch {
    return new Response('Pi camera unavailable', { status: 503 })
  }
}
