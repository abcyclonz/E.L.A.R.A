import { NextRequest, NextResponse } from 'next/server'

// 1. Hard-validate the environment variable at the top level
const PI_URL = process.env.PI_CAMERA_URL ?? 'http://localhost:8765'

// Helper to ensure the URL is what we expect
function isValidPiOrigin(url: string) {
  try {
    const parsed = new URL(url);
    // Allow only specific hostnames or IP ranges if possible
    // Example: only allow localhost or a specific private IP range
    const allowedHosts = ['localhost', '127.0.0.1', 'raspberrypi.local'];
    return allowedHosts.includes(parsed.hostname) || parsed.hostname.startsWith('192.168.');
  } catch {
    return false;
  }
}

export async function POST(req: NextRequest) {
  // 2. Immediate Guard
  if (!isValidPiOrigin(PI_URL)) {
    console.error(`SSRF Guard: Blocked attempt to fetch invalid origin: ${PI_URL}`);
    return NextResponse.json({ detail: "System configuration error: Invalid Upstream" }, { status: 500 });
  }

  try {
    const body = await req.json();
    
    const upstream = await fetch(`${PI_URL}/register`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      // Use a shorter timeout for registration to prevent resource exhaustion
      signal: AbortSignal.timeout(10000), 
    });

    // 3. Check for non-JSON responses before parsing
    const contentType = upstream.headers.get("content-type");
    if (!contentType || !contentType.includes("application/json")) {
       return NextResponse.json({ detail: "Pi returned an invalid response format" }, { status: 502 });
    }

    const data = await upstream.json();
    return NextResponse.json(data, { status: upstream.status });

  } catch (e) {
    // 4. Sanitize error messages - do NOT return the raw error object 'e'
    console.error("Pi Registration Error:", e);
    return NextResponse.json(
      { detail: "The camera service is currently unreachable." }, 
      { status: 503 }
    );
  }
}