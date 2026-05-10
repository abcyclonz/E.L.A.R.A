import { NextRequest, NextResponse } from 'next/server'

const BACKEND = process.env.BACKEND_URL ?? 'http://localhost:8001'
const DISABLE_OTP = process.env.NEXT_PUBLIC_DISABLE_OTP === 'true'

export async function POST(request: NextRequest) {
  try {
    if (DISABLE_OTP) {
      return NextResponse.json({ status: 'verified', bypassed: true })
    }

    const body = await request.json()

    const response = await fetch(`${BACKEND}/verify`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })

    const data = await response.json()

    if (!response.ok) {
      return NextResponse.json(data, { status: response.status })
    }

    return NextResponse.json(data)
  } catch (error) {
    console.error('Verify OTP proxy error:', error)
    return NextResponse.json(
      { detail: 'Internal server error' },
      { status: 500 }
    )
  }
}
