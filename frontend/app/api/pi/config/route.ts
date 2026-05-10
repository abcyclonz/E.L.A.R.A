import { NextResponse } from 'next/server'

const PI_URL = process.env.PI_CAMERA_URL ?? 'http://localhost:8765'

export async function GET() {
  try {
    const r = await fetch(`${PI_URL}/config`, {
      cache: 'no-store',
      signal: AbortSignal.timeout(3000),
    })
    const data = await r.json()
    return NextResponse.json(data)
  } catch {
    return NextResponse.json({ is_pi: false, mode: 'laptop', source: 'browser' })
  }
}
