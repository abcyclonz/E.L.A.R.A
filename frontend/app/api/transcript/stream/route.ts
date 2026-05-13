import { NextRequest } from 'next/server'

const BACKEND = process.env.BACKEND_URL ?? 'http://localhost:8001'

export const dynamic = 'force-dynamic'

export async function GET(req: NextRequest) {
  const token = req.nextUrl.searchParams.get('token')
  if (!token) return new Response('Unauthorized', { status: 401 })

  try {
    const controller = new AbortController()

    // Abort upstream when client disconnects
    req.signal.addEventListener('abort', () => controller.abort())

    const upstream = await fetch(
      `${BACKEND}/transcript/stream?user_token=${encodeURIComponent(token)}`,
      {
        headers: { Accept: 'text/event-stream' },
        signal: controller.signal,
      }
    )

    if (!upstream.ok || !upstream.body) {
      return new Response('Transcript stream unavailable', { status: 503 })
    }

    return new Response(upstream.body, {
      headers: {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache, no-store',
        'X-Accel-Buffering': 'no',
        'Connection': 'keep-alive',
      },
    })
  } catch (err: any) {
    if (err?.name === 'AbortError') return new Response('', { status: 499 })
    return new Response('Transcript stream unavailable', { status: 503 })
  }
}
