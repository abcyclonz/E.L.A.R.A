import { NextResponse } from 'next/server'

const PI_URL = process.env.PI_CAMERA_URL ?? 'http://localhost:8765'

export async function GET() {
  try {
    const upstream = await fetch(`${PI_URL}/frames/event`, {
      cache: 'no-store',
      signal: AbortSignal.timeout(5000),
    })
    if (!upstream.ok) return NextResponse.json({ error: 'pi unavailable' }, { status: 503 })
    const data = await upstream.json()
    return NextResponse.json(data)
  } catch {
    return NextResponse.json({ error: 'pi unavailable' }, { status: 503 })
  }
}
