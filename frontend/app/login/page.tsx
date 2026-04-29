"use client"

import { useState, useRef, useEffect } from "react"
import { useRouter } from "next/navigation"
import { useAuth } from "@/components/auth-context"
import { toast } from "sonner"

type Mode = 'signup' | 'login'

// ── Ping-pong Video ────────────────────────────────────────────────────────
function PingPongVideo() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const rafRef = useRef<number>(0)

  useEffect(() => {
    const v = videoRef.current
    if (!v) return
    const stepBack = () => {
      v.currentTime = Math.max(0, v.currentTime - 0.035)
      if (v.currentTime <= 0.05) { v.play() }
      else { rafRef.current = requestAnimationFrame(stepBack) }
    }
    const onEnded = () => {
      cancelAnimationFrame(rafRef.current)
      rafRef.current = requestAnimationFrame(stepBack)
    }
    v.addEventListener('ended', onEnded)
    return () => { v.removeEventListener('ended', onEnded); cancelAnimationFrame(rafRef.current) }
  }, [])

  return (
    <video ref={videoRef} src="/Cute_Robot.mp4" autoPlay muted playsInline
      style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', objectFit: 'cover', objectPosition: 'center' }} />
  )
}

// ── Stepper ────────────────────────────────────────────────────────────────
function Stepper({ step }: { step: number }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', marginTop: 24 }}>
      {['Details', 'Face Scan'].map((label, i) => (
        <div key={i} style={{ display: 'flex', alignItems: 'center', ...(i === 0 && { flex: 1 }) }}>
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 7 }}>
            <div style={{
              width: 30, height: 30, borderRadius: '50%',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              fontSize: 12, fontWeight: 700, transition: 'all .3s',
              background: step > i ? 'oklch(0.35 0.10 145)' : step === i ? 'white' : 'rgba(255,255,255,0.2)',
              color: step > i ? 'white' : step === i ? 'oklch(0.22 0.08 145)' : 'rgba(255,255,255,0.6)',
              border: step === i ? '2px solid white' : 'none',
            }}>{step > i ? '✓' : i + 1}</div>
            <span style={{ fontSize: 11, fontWeight: 600, letterSpacing: '0.04em', textTransform: 'uppercase', color: step === i ? 'white' : 'rgba(255,255,255,0.5)' }}>{label}</span>
          </div>
          {i === 0 && (
            <div style={{ flex: 1, height: 1.5, margin: '0 10px', marginBottom: 22, transition: 'background .4s', background: step > 0 ? 'white' : 'rgba(255,255,255,0.25)' }} />
          )}
        </div>
      ))}
    </div>
  )
}

// ── Face Reticle ───────────────────────────────────────────────────────────
function FaceReticle() {
  return (
    <svg viewBox="0 0 320 240" fill="none" xmlns="http://www.w3.org/2000/svg"
      style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', pointerEvents: 'none' }}>
      <ellipse className="e-reticle-ring" cx="160" cy="120" rx="78" ry="98"
        stroke="oklch(0.35 0.10 145)" strokeWidth="2" strokeDasharray="8 5" opacity="0.9" />
      {[[-1, -1], [1, -1], [-1, 1], [1, 1]].map(([sx, sy], i) => (
        <g key={i} transform={`translate(${160 + sx * 78} ${120 + sy * 98}) scale(${sx} ${sy})`}>
          <path d="M0 -13 L0 0 L13 0" stroke="white" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" opacity="0.75" />
        </g>
      ))}
      <line x1="157" y1="120" x2="163" y2="120" stroke="oklch(0.35 0.10 145)" strokeWidth="1.5" opacity="0.7" />
      <line x1="160" y1="117" x2="160" y2="123" stroke="oklch(0.35 0.10 145)" strokeWidth="1.5" opacity="0.7" />
    </svg>
  )
}

// ── Angle Checklist ────────────────────────────────────────────────────────
const ANGLES = ['Center', 'Left', 'Right', 'Up', 'Tilt']

function AngleChecklist({ captured }: { captured: number[] }) {
  return (
    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginTop: 14 }}>
      {ANGLES.map((a, i) => {
        const active = captured.includes(i)
        return (
          <div key={a} style={{
            display: 'flex', alignItems: 'center', gap: 6, padding: '6px 12px',
            borderRadius: 99, fontSize: 13, fontWeight: 500, transition: 'all .22s',
            color: active ? 'oklch(0.26 0.09 145)' : 'oklch(0.58 0.02 200)',
            border: `1.5px solid ${active ? 'oklch(0.35 0.10 145 / 0.45)' : 'rgba(255,255,255,0.65)'}`,
            background: active ? 'oklch(0.35 0.10 145 / 0.15)' : 'rgba(255,255,255,0.45)',
            backdropFilter: 'blur(8px)',
          }}>
            <span style={{ fontSize: 10 }}>{active ? '✓' : '○'}</span>{a}
          </div>
        )
      })}
    </div>
  )
}

// ── Helpers ────────────────────────────────────────────────────────────────
function Spinner() { return <span className="e-spinner" /> }

function FieldGroup({ label, style, children }: { label: string; style?: React.CSSProperties; children: React.ReactNode }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 6, ...style }}>
      <label style={{ fontSize: 13, fontWeight: 500, color: 'oklch(0.36 0.02 200)', letterSpacing: '0.01em' }}>{label}</label>
      {children}
    </div>
  )
}

// ── Mode Toggle ────────────────────────────────────────────────────────────
function ModeToggle({ mode, onSwitch }: { mode: Mode; onSwitch: (m: Mode) => void }) {
  return (
    <div style={{
      background: 'rgba(255,255,255,0.38)', backdropFilter: 'blur(16px)',
      border: '1px solid rgba(255,255,255,0.65)', borderRadius: 99,
      padding: 4, display: 'flex', marginBottom: 32,
      boxShadow: '0 2px 16px rgba(0,0,0,0.07)'
    }}>
      {(['signup', 'login'] as Mode[]).map(m => (
        <button key={m} onClick={() => onSwitch(m)} style={{
          flex: 1, padding: '10px 28px', borderRadius: 99, border: 'none', cursor: 'pointer',
          background: mode === m ? 'white' : 'transparent',
          color: mode === m ? 'oklch(0.13 0.01 145)' : 'oklch(0.58 0.02 200)',
          fontFamily: 'inherit', fontSize: 14, fontWeight: 600,
          boxShadow: mode === m ? '0 2px 10px rgba(0,0,0,0.10)' : 'none',
          transition: 'all .22s'
        }}>{m === 'signup' ? 'Sign Up' : 'Log In'}</button>
      ))}
    </div>
  )
}

// ── Step 1 Form ────────────────────────────────────────────────────────────
function Step1Form({ onNext }: { onNext: (data: Record<string, string>) => void }) {
  const [form, setForm] = useState({ firstName: '', lastName: '', email: '', password: '', language: 'en' })
  const [loading, setLoading] = useState(false)
  const set = (k: string) => (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) =>
    setForm(f => ({ ...f, [k]: e.target.value }))
  const valid = form.firstName && form.lastName && form.email.includes('@') && form.password.length >= 6

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!valid) return
    setLoading(true)
    await new Promise(r => setTimeout(r, 400))
    setLoading(false)
    onNext(form)
  }

  return (
    <form onSubmit={handleSubmit} className="e-anim-in" style={{ display: 'flex', flexDirection: 'column', gap: 18 }}>
      <div style={{ display: 'flex', gap: 14 }}>
        <FieldGroup label="First Name" style={{ flex: 1 }}>
          <input className="e-field-input" placeholder="Jane" value={form.firstName} onChange={set('firstName')} required />
        </FieldGroup>
        <FieldGroup label="Last Name" style={{ flex: 1 }}>
          <input className="e-field-input" placeholder="Smith" value={form.lastName} onChange={set('lastName')} required />
        </FieldGroup>
      </div>
      <FieldGroup label="Email Address">
        <input className="e-field-input" type="email" placeholder="jane@example.com" value={form.email} onChange={set('email')} required />
      </FieldGroup>
      <FieldGroup label="Password">
        <input className="e-field-input" type="password" placeholder="At least 6 characters" value={form.password} onChange={set('password')} required />
      </FieldGroup>
      <FieldGroup label="Preferred Language">
        <select className="e-field-input e-select" value={form.language} onChange={set('language')}>
          <option value="en">English</option>
          <option value="es">Español</option>
          <option value="fr">Français</option>
          <option value="de">Deutsch</option>
          <option value="hi">हिन्दी</option>
          <option value="ml">Malayalam</option>
          <option value="ta">Tamil</option>
        </select>
      </FieldGroup>
      <div style={{ height: 4 }} />
      <button className="e-btn-primary" type="submit" disabled={!valid || loading}>
        {loading ? <Spinner /> : 'Continue to Face Setup →'}
      </button>
    </form>
  )
}

// ── Step 2 Face Scan ───────────────────────────────────────────────────────
function Step2Face({ signupData, onDone }: { signupData: Record<string, string>; onDone: () => void }) {
  const { signup } = useAuth()
  const [captured, setCaptured] = useState<number[]>([])
  const [loading, setLoading] = useState(false)
  const nextAngle = ANGLES.findIndex((_, i) => !captured.includes(i))
  const done = captured.length === ANGLES.length

  const handleCapture = async () => {
    if (done) {
      setLoading(true)
      try {
        await signup({
          email: signupData.email,
          password: signupData.password,
          fullName: `${signupData.firstName} ${signupData.lastName}`.trim(),
          age: '',
          preferredLanguage: signupData.language,
          background: '',
          interests: [],
          conversationPreferences: [],
          technologyUsage: '',
          conversationGoals: [],
          additionalInfo: ''
        })
        onDone()
      } catch (err) {
        toast.error(err instanceof Error ? err.message : 'Signup failed')
      } finally {
        setLoading(false)
      }
      return
    }
    setLoading(true)
    await new Promise(r => setTimeout(r, 900))
    setCaptured(c => [...c, nextAngle])
    setLoading(false)
  }

  return (
    <div className="e-anim-in" style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
      <p style={{ fontSize: 14, color: 'oklch(0.36 0.02 200)', lineHeight: 1.6 }}>
        Position your face in the frame and capture each angle. Good lighting helps.
      </p>
      <div style={{ position: 'relative', aspectRatio: '4/3', background: 'oklch(0.12 0.01 145)', borderRadius: 12, overflow: 'hidden', border: '1.5px solid rgba(255,255,255,0.20)' }}>
        <svg width="100%" height="100%" style={{ position: 'absolute', inset: 0 }}>
          <defs>
            <pattern id="fdiag" width="24" height="24" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
              <rect width="24" height="24" fill="oklch(0.12 0.01 145)" />
              <rect width="12" height="24" fill="oklch(0.15 0.01 145)" />
            </pattern>
          </defs>
          <rect width="100%" height="100%" fill="url(#fdiag)" />
        </svg>
        <div style={{ position: 'absolute', inset: 0, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 10 }}>
          <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="oklch(0.45 0.03 145)" strokeWidth="1.5" strokeLinecap="round">
            <path d="M23 19a2 2 0 01-2 2H3a2 2 0 01-2-2V8a2 2 0 012-2h4l2-3h6l2 3h4a2 2 0 012 2z" />
            <circle cx="12" cy="13" r="4" />
          </svg>
          <span style={{ fontSize: 13, color: 'oklch(0.50 0.04 145)', fontWeight: 500 }}>
            {done ? 'All angles captured ✓' : `Look: ${ANGLES[nextAngle]}`}
          </span>
        </div>
        <FaceReticle />
      </div>
      <AngleChecklist captured={captured} />
      <div style={{ height: 2 }} />
      <button className="e-btn-primary" onClick={handleCapture} disabled={loading} type="button">
        {loading
          ? <><Spinner /> {done ? 'Creating account…' : 'Capturing…'}</>
          : done ? 'Complete Registration →' : `Capture Frame (${captured.length}/${ANGLES.length})`}
      </button>
    </div>
  )
}

// ── Login Form ─────────────────────────────────────────────────────────────
function LoginForm({ onDone }: { onDone: () => void }) {
  const { login } = useAuth()
  const [form, setForm] = useState({ email: '', password: '' })
  const [loading, setLoading] = useState(false)
  const set = (k: string) => (e: React.ChangeEvent<HTMLInputElement>) =>
    setForm(f => ({ ...f, [k]: e.target.value }))
  const valid = form.email.includes('@') && form.password.length >= 1

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!valid || loading) return
    setLoading(true)
    try {
      await login(form.email, form.password)
      onDone()
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Login failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <form onSubmit={handleSubmit} className="e-anim-in" style={{ display: 'flex', flexDirection: 'column', gap: 18 }}>
      <FieldGroup label="Email Address">
        <input className="e-field-input" type="email" placeholder="jane@example.com" value={form.email} onChange={set('email')} required />
      </FieldGroup>
      <FieldGroup label="Password">
        <input className="e-field-input" type="password" placeholder="Your password" value={form.password} onChange={set('password')} required />
      </FieldGroup>
      <div style={{ height: 4 }} />
      <button className="e-btn-primary" type="submit" disabled={!valid || loading}>
        {loading ? <><Spinner /> Signing in…</> : 'Sign In →'}
      </button>
    </form>
  )
}

// ── Main Auth Page ─────────────────────────────────────────────────────────
export default function AuthPage() {
  const router = useRouter()
  const { isAuthenticated, authLoading } = useAuth()
  const [mode, setMode] = useState<Mode>('login')
  const [step, setStep] = useState(0)
  const [signupData, setSignupData] = useState<Record<string, string> | null>(null)

  useEffect(() => {
    if (!authLoading && isAuthenticated) router.replace('/')
  }, [isAuthenticated, authLoading, router])

  const switchMode = (m: Mode) => { setMode(m); setStep(0) }

  const ghostBtn: React.CSSProperties = {
    background: 'none', border: 'none', cursor: 'pointer',
    fontSize: 14, color: 'oklch(0.35 0.10 145)', fontWeight: 500,
    padding: '2px 0', textDecoration: 'underline', textUnderlineOffset: 3,
    fontFamily: 'inherit'
  }

  return (
    <div className="auth-layout">
      {/* LEFT — Video panel */}
      <div className="auth-panel-left">
        <PingPongVideo />
        <div className="auth-overlay">
          <div style={{ flexShrink: 0 }}>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src="/elara_white.png" alt="Elara" style={{ height: 58, width: 'auto', objectFit: 'contain', objectPosition: 'left', filter: 'drop-shadow(0 1px 6px rgba(0,0,0,0.3))' }} />
          </div>
          <div style={{ flex: 1 }} />
          <div style={{ flexShrink: 0 }}>
            <h1 style={{ fontSize: 'clamp(24px, 2.8vw, 40px)', fontWeight: 700, color: 'white', lineHeight: 1.18, letterSpacing: '-0.03em', marginBottom: 14, textShadow: '0 2px 20px rgba(0,0,0,0.4)' }}>
              Welcome to your<br />new companion.
            </h1>
            <p style={{ fontSize: 15, color: 'rgba(255,255,255,0.75)', lineHeight: 1.65, maxWidth: 340, textShadow: '0 1px 8px rgba(0,0,0,0.4)', marginBottom: 8 }}>
              Elara remembers the things that matter. Your routines, your stories, your people — always at your side.
            </p>
            {mode === 'signup' && <Stepper step={step} />}
            <p style={{ fontSize: 11, color: 'rgba(255,255,255,0.35)', marginTop: 24, lineHeight: 1.6 }}>
              Facial data never leaves your device. End-to-end encrypted.
            </p>
          </div>
        </div>
      </div>

      {/* RIGHT — Glass panel */}
      <div className="auth-panel-right">
        {/* Animated blobs */}
        <div style={{ position: 'absolute', width: 480, height: 480, top: -120, right: -100, background: 'oklch(0.78 0.10 145 / 0.55)', borderRadius: '50%', filter: 'blur(65px)', zIndex: 0, animation: 'elaraBlob1 20s ease-in-out infinite alternate', pointerEvents: 'none' }} />
        <div style={{ position: 'absolute', width: 380, height: 380, bottom: -80, left: -80, background: 'oklch(0.72 0.08 175 / 0.45)', borderRadius: '50%', filter: 'blur(65px)', zIndex: 0, animation: 'elaraBlob2 17s ease-in-out infinite alternate', pointerEvents: 'none' }} />
        <div style={{ position: 'absolute', width: 260, height: 260, top: '45%', left: '55%', background: 'oklch(0.85 0.07 120 / 0.38)', borderRadius: '50%', filter: 'blur(65px)', zIndex: 0, animation: 'elaraBlob3 14s ease-in-out infinite alternate', pointerEvents: 'none' }} />

        <div className="auth-scroll-area">
          <div style={{ width: '100%' }}>
            <div className="auth-glass-card">
              <div style={{ marginBottom: 24 }}>
                <p style={{ fontSize: 13, color: 'oklch(0.58 0.02 200)', marginBottom: 6 }}>Step into something new</p>
                <h2 style={{ fontSize: 'clamp(20px, 2vw, 28px)', fontWeight: 700, letterSpacing: '-0.025em', color: 'oklch(0.13 0.01 145)' }}>
                  {mode === 'signup' ? (step === 0 ? 'Create your account' : 'Set up face recognition') : 'Welcome back'}
                </h2>
              </div>

              <ModeToggle mode={mode} onSwitch={switchMode} />

              {mode === 'signup'
                ? (step === 0
                    ? <Step1Form onNext={(data) => { setSignupData(data); setStep(1) }} />
                    : <Step2Face signupData={signupData!} onDone={() => router.replace('/')} />)
                : <LoginForm onDone={() => router.replace('/')} />}

              <div style={{ marginTop: 28, textAlign: 'center', fontSize: 13, color: 'oklch(0.58 0.02 200)' }}>
                {mode === 'signup'
                  ? <span>Already have an account? <button style={ghostBtn} onClick={() => switchMode('login')}>Log in</button></span>
                  : <span>New here? <button style={ghostBtn} onClick={() => switchMode('signup')}>Create an account</button></span>}
              </div>
            </div>

            <div style={{ marginTop: 20, display: 'flex', gap: 20, justifyContent: 'center' }}>
              {['Privacy Policy', 'Terms of Use', 'Accessibility'].map(t => (
                <a key={t} href="#" style={{ fontSize: 12, color: 'oklch(0.36 0.02 200)', textDecoration: 'none', opacity: 0.6 }}>{t}</a>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
