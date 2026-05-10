import { NextRequest, NextResponse } from 'next/server'

const PI_URL = process.env.PI_CAMERA_URL ?? 'http://localhost:8765'

export async function POST(req: NextRequest) {
  try {
    const body = await req.json()
    const upstream = await fetch(`${PI_URL}/register`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(30000),
    })
    const data = await upstream.json()
    return NextResponse.json(data, { status: upstream.status })
  } catch (e) {
    return NextResponse.json({ detail: `Pi unreachable: ${e}` }, { status: 503 })
  }
}
