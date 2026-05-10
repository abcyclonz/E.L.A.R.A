const PI_URL = process.env.PI_CAMERA_URL ?? 'http://localhost:8765'

export async function GET() {
  try {
    const upstream = await fetch(`${PI_URL}/camera/frame`, {
      cache: 'no-store',
      signal: AbortSignal.timeout(5000),
    })
    if (!upstream.ok) return new Response('Pi camera unavailable', { status: 503 })
    const buf = await upstream.arrayBuffer()
    return new Response(buf, {
      headers: { 'Content-Type': 'image/jpeg', 'Cache-Control': 'no-store' },
    })
  } catch {
    return new Response('Pi camera unavailable', { status: 503 })
  }
}
