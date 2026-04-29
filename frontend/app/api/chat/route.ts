import { NextResponse } from 'next/server'

const BACKEND = process.env.BACKEND_URL ?? 'http://localhost:8001'

export const maxDuration = 60

export async function POST(req: Request) {
  try {
    const { messages, agentState } = await req.json()

    if (!messages || messages.length === 0) {
      return NextResponse.json({ error: 'No messages provided' }, { status: 400 })
    }

    const userToken = agentState?.userToken || null
    const sessionId = agentState?.sessionId || 'frontend-session'
    const location  = agentState?.location  || null

    if (!userToken) {
      return NextResponse.json({ error: 'Not authenticated' }, { status: 401 })
    }

    const payload = {
      session_id: sessionId,
      user_input: messages[messages.length - 1]?.content || '',
      user_token: userToken,
      location,
    }

    const response = await fetch(`${BACKEND}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
      body: JSON.stringify(payload),
    })

    if (!response.ok) {
      const errorText = await response.text()
      return NextResponse.json(
        { error: `Backend error: ${response.status}`, details: errorText },
        { status: response.status }
      )
    }

    const data = await response.json()

    if (!data.ai_response) {
      return NextResponse.json(
        { error: 'Invalid backend response: missing ai_response' },
        { status: 500 }
      )
    }

    return NextResponse.json({
      message: { role: 'assistant', content: data.ai_response },
      agentState: {
        ...agentState,
        sessionId: data.session_id || sessionId,
        turnCount: data.turn_count || 0,
        routerDecision: data.current_router_decision || '',
      },
    })
  } catch (error) {
    console.error('Chat API route error:', error)
    return NextResponse.json(
      { error: `Internal Server Error: ${error instanceof Error ? error.message : 'Unknown error'}` },
      { status: 500 }
    )
  }
}
