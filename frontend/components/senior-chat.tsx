"use client"

import { useState, useRef, useEffect, useCallback } from "react"
import { useAuth } from "@/components/auth-context"

// ── Types ──────────────────────────────────────────────────────────────────
interface Message { role: 'user' | 'assistant'; text: string; ts: Date }
interface WatchData { hr: number; battery: number; steps: number; temp: number }
interface MemoryItem { id: number; date: string; text: string }

// ── Icons ──────────────────────────────────────────────────────────────────
const Ic = ({ d, size = 18, sw = 1.8 }: { d: React.ReactNode; size?: number; sw?: number }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={sw} strokeLinecap="round" strokeLinejoin="round">{d}</svg>
)
const IcBrain = ({ size = 18 }) => <Ic size={size} d={<path d="M9.5 2A2.5 2.5 0 007 4.5v0A2.5 2.5 0 004.5 7H4a2 2 0 00-2 2v2a2 2 0 002 2h.5A2.5 2.5 0 007 15.5v0A2.5 2.5 0 009.5 18h5A2.5 2.5 0 0017 15.5v0A2.5 2.5 0 0019.5 13H20a2 2 0 002-2V9a2 2 0 00-2-2h-.5A2.5 2.5 0 0017 4.5v0A2.5 2.5 0 0014.5 2h-5z" />} />
const IcCamera = ({ size = 18 }) => <Ic size={size} d={<><path d="M23 19a2 2 0 01-2 2H3a2 2 0 01-2-2V8a2 2 0 012-2h4l2-3h6l2 3h4a2 2 0 012 2z" /><circle cx="12" cy="13" r="4" /></>} />
const IcMic = ({ size = 18, off = false }) => <Ic size={size} d={off
  ? <><line x1="1" y1="1" x2="23" y2="23" /><path d="M9 9v3a3 3 0 005.12 2.12M15 9.34V4a3 3 0 00-5.94-.6" /><path d="M17 16.95A7 7 0 015 12v-2m14 0v2a7 7 0 01-.11 1.23" /><line x1="12" y1="19" x2="12" y2="23" /><line x1="8" y1="23" x2="16" y2="23" /></>
  : <><path d="M12 1a3 3 0 00-3 3v8a3 3 0 006 0V4a3 3 0 00-3-3z" /><path d="M19 10v2a7 7 0 01-14 0v-2" /><line x1="12" y1="19" x2="12" y2="23" /><line x1="8" y1="23" x2="16" y2="23" /></>} />
const IcMoon = ({ size = 18 }) => <Ic size={size} d={<path d="M21 12.79A9 9 0 1111.21 3 7 7 0 0021 12.79z" />} />
const IcSend = ({ size = 18 }) => <Ic size={size} d={<><line x1="22" y1="2" x2="11" y2="13" /><polygon points="22 2 15 22 11 13 2 9 22 2" /></>} />
const IcX = ({ size = 18 }) => <Ic size={size} d={<><line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" /></>} />
const IcWave = ({ size = 18 }) => <Ic size={size} d={<polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />} />
const IcUser = ({ size = 18 }) => <Ic size={size} d={<><circle cx="12" cy="8" r="4" /><path d="M4 20c0-4 3.6-7 8-7s8 3 8 7" /></>} />
const IcLock = ({ size = 18 }) => <Ic size={size} d={<><rect x="3" y="11" width="18" height="11" rx="2" /><path d="M7 11V7a5 5 0 0110 0v4" /></>} />
const IcPhone = ({ size = 18 }) => <Ic size={size} d={<><rect x="5" y="2" width="14" height="20" rx="2" /><line x1="12" y1="18" x2="12.01" y2="18" strokeWidth={3} /></>} />
const IcHelp = ({ size = 18 }) => <Ic size={size} d={<><circle cx="12" cy="12" r="10" /><path d="M9.09 9a3 3 0 015.83 1c0 2-3 3-3 3" /><line x1="12" y1="17" x2="12.01" y2="17" strokeWidth={3} /></>} />
const IcChevron = ({ size = 16 }) => <Ic size={size} sw={2} d={<polyline points="9 18 15 12 9 6" />} />

// ── Mock transcript data ───────────────────────────────────────────────────
const TRANSCRIPT = [
  { who: 'robot', text: "Good morning! How did you sleep?" },
  { who: 'user', text: "Not too bad. A little stiff." },
  { who: 'robot', text: "Want me to walk you through your morning stretches?" },
  { who: 'user', text: "Yes, that would be lovely." },
  { who: 'robot', text: "Let's start with shoulder rolls — ten times, nice and slow." },
  { who: 'user', text: "Okay, doing it now." },
  { who: 'robot', text: "Excellent. And backward now, same count." },
  { who: 'user', text: "Done. That feels better already." },
  { who: 'robot', text: "Great! Neck stretches next?" },
]

// ── Ping-pong Watch Video ──────────────────────────────────────────────────
function WatchVideo({ data }: { data: WatchData }) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const rafRef = useRef<number>(0)

  useEffect(() => {
    const v = videoRef.current
    if (!v) return
    const stepBack = () => {
      v.currentTime = Math.max(0.2, v.currentTime - 0.033)
      if (v.currentTime <= 0.22) { v.currentTime = 0.2; v.play() }
      else { rafRef.current = requestAnimationFrame(stepBack) }
    }
    const onTimeUpdate = () => { if (v.duration && v.currentTime >= v.duration - 0.2) v.dispatchEvent(new Event('ended')) }
    const onEnded = () => { cancelAnimationFrame(rafRef.current); rafRef.current = requestAnimationFrame(stepBack) }
    const onMeta = () => { v.currentTime = 0.2 }
    v.addEventListener('ended', onEnded)
    v.addEventListener('timeupdate', onTimeUpdate)
    v.addEventListener('loadedmetadata', onMeta)
    return () => {
      v.removeEventListener('ended', onEnded)
      v.removeEventListener('timeupdate', onTimeUpdate)
      v.removeEventListener('loadedmetadata', onMeta)
      cancelAnimationFrame(rafRef.current)
    }
  }, [])

  const battColor = data.battery > 50 ? 'oklch(0.72 0.15 145)' : data.battery > 20 ? '#f59e0b' : '#ef4444'
  const metricBox: React.CSSProperties = { background: 'rgba(0,0,0,0.45)', backdropFilter: 'blur(12px)', border: '1px solid rgba(255,255,255,0.12)', borderRadius: 12, padding: '10px 14px', minWidth: 100 }

  return (
    <div style={{ position: 'relative', width: '100%', aspectRatio: '4/3', background: '#000', overflow: 'hidden', borderRadius: 0 }}>
      <video ref={videoRef} src="/watch_merge.mp4" autoPlay muted playsInline
        style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', objectFit: 'cover' }} />
      <div style={{ position: 'absolute', inset: 0, pointerEvents: 'none', display: 'flex', flexDirection: 'column', justifyContent: 'space-between', padding: '16px 18px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
          <div style={metricBox}>
            <div style={{ fontSize: 11, color: 'rgba(255,255,255,0.55)', fontWeight: 500, marginBottom: 3 }}>❤️ Heart Rate</div>
            <div style={{ fontSize: 22, fontWeight: 700, color: (data.hr > 100 || data.hr < 55) ? '#ff6b6b' : 'white', lineHeight: 1, letterSpacing: '-0.02em' }}>{data.hr}</div>
            <div style={{ fontSize: 11, color: 'rgba(255,255,255,0.40)', marginTop: 2 }}>BPM</div>
          </div>
          <div style={{ ...metricBox, textAlign: 'right' }}>
            <div style={{ fontSize: 11, color: 'rgba(255,255,255,0.55)', fontWeight: 500, marginBottom: 3 }}>🌡️ Temp</div>
            <div style={{ fontSize: 22, fontWeight: 700, color: 'white', lineHeight: 1, letterSpacing: '-0.02em' }}>{data.temp}</div>
            <div style={{ fontSize: 11, color: 'rgba(255,255,255,0.40)', marginTop: 2 }}>°C</div>
          </div>
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end' }}>
          <div style={metricBox}>
            <div style={{ fontSize: 11, color: 'rgba(255,255,255,0.55)', fontWeight: 500, marginBottom: 3 }}>👟 Steps</div>
            <div style={{ fontSize: 22, fontWeight: 700, color: 'white', lineHeight: 1, letterSpacing: '-0.02em' }}>{data.steps.toLocaleString()}</div>
            <div style={{ fontSize: 11, color: 'rgba(255,255,255,0.40)', marginTop: 2 }}>today</div>
          </div>
          <div style={{ ...metricBox, textAlign: 'right', minWidth: 110 }}>
            <div style={{ fontSize: 11, color: 'rgba(255,255,255,0.55)', fontWeight: 500, marginBottom: 6 }}>🔋 Battery</div>
            <div style={{ height: 7, background: 'rgba(255,255,255,0.15)', borderRadius: 99, overflow: 'hidden', marginBottom: 4 }}>
              <div style={{ height: '100%', width: `${data.battery}%`, background: battColor, borderRadius: 99, transition: 'width .5s ease' }} />
            </div>
            <div style={{ fontSize: 18, fontWeight: 700, color: battColor, letterSpacing: '-0.02em' }}>{data.battery}%</div>
          </div>
        </div>
      </div>
    </div>
  )
}

// ── Override Modal ─────────────────────────────────────────────────────────
function OverrideModal({ data, onApply, onClose }: { data: WatchData; onApply: (d: WatchData) => void; onClose: () => void }) {
  const [local, setLocal] = useState<WatchData>({ ...data })
  const set = (k: keyof WatchData) => (v: number) => setLocal(d => ({ ...d, [k]: v }))
  const params: { key: keyof WatchData; label: string; unit: string; min: number; max: number; step: number; icon: string }[] = [
    { key: 'hr', label: 'Heart Rate', unit: 'BPM', min: 40, max: 180, step: 1, icon: '❤️' },
    { key: 'battery', label: 'Battery', unit: '%', min: 0, max: 100, step: 1, icon: '🔋' },
    { key: 'steps', label: 'Step Count', unit: 'steps', min: 0, max: 20000, step: 100, icon: '👟' },
    { key: 'temp', label: 'Temperature', unit: '°C', min: 34, max: 42, step: 0.1, icon: '🌡️' },
  ]

  return (
    <div className="e-modal-backdrop" onClick={e => e.target === e.currentTarget && onClose()}>
      <div className="e-anim-in" style={{ width: 440, background: 'rgba(255,255,255,0.88)', backdropFilter: 'blur(40px)', border: '1px solid rgba(255,255,255,0.8)', borderRadius: 24, boxShadow: '0 24px 80px rgba(0,0,0,0.18)', overflow: 'hidden' }}>
        <div style={{ padding: '22px 28px 16px', borderBottom: '1px solid rgba(0,0,0,0.07)', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <div>
            <div style={{ fontSize: 17, fontWeight: 700, letterSpacing: '-0.02em' }}>Override Simulator</div>
            <div style={{ fontSize: 13, color: 'oklch(0.58 0.02 200)', marginTop: 2 }}>Manually set watch parameters</div>
          </div>
          <button onClick={onClose} style={{ background: 'rgba(0,0,0,0.06)', border: 'none', cursor: 'pointer', color: 'oklch(0.58 0.02 200)', padding: 8, borderRadius: 10, display: 'flex' }}>
            <IcX size={18} />
          </button>
        </div>
        <div style={{ padding: '20px 28px', display: 'flex', flexDirection: 'column', gap: 20 }}>
          {params.map(p => (
            <div key={p.key}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
                <span style={{ fontSize: 14, fontWeight: 600, color: 'oklch(0.13 0.01 145)' }}>{p.icon} {p.label}</span>
                <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  <input type="number" min={p.min} max={p.max} step={p.step} value={local[p.key]}
                    onChange={e => set(p.key)(parseFloat(e.target.value) || p.min)}
                    style={{ width: 72, padding: '5px 8px', border: '1.5px solid rgba(0,0,0,0.12)', borderRadius: 8, fontFamily: 'inherit', fontSize: 14, fontWeight: 600, textAlign: 'center', outline: 'none', color: 'oklch(0.13 0.01 145)', background: 'rgba(255,255,255,0.7)' }} />
                  <span style={{ fontSize: 12, color: 'oklch(0.58 0.02 200)', fontWeight: 500 }}>{p.unit}</span>
                </div>
              </div>
              <input type="range" min={p.min} max={p.max} step={p.step} value={local[p.key]}
                onChange={e => set(p.key)(parseFloat(e.target.value))}
                style={{ width: '100%', accentColor: 'oklch(0.35 0.10 145)', height: 5, cursor: 'pointer' }} />
              <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 4, fontSize: 11, color: 'oklch(0.58 0.02 200)' }}>
                <span>{p.min}</span><span>{p.max}</span>
              </div>
            </div>
          ))}
        </div>
        <div style={{ padding: '0 28px 24px', display: 'flex', gap: 10 }}>
          <button onClick={onClose} style={{ flex: 1, padding: 12, borderRadius: 12, border: '1.5px solid rgba(0,0,0,0.12)', background: 'rgba(255,255,255,0.6)', color: 'oklch(0.36 0.02 200)', fontFamily: 'inherit', fontSize: 14, fontWeight: 600, cursor: 'pointer' }}>Cancel</button>
          <button onClick={() => { onApply(local); onClose() }} style={{ flex: 2, padding: 12, borderRadius: 12, border: 'none', background: 'oklch(0.35 0.10 145)', color: 'white', fontFamily: 'inherit', fontSize: 14, fontWeight: 600, cursor: 'pointer', boxShadow: '0 3px 14px oklch(0.35 0.10 145 / 0.35)' }}>Save & Apply</button>
        </div>
      </div>
    </div>
  )
}

// ── Live Transcript ────────────────────────────────────────────────────────
function LiveTranscript({ muted, userName }: { muted: boolean; userName: string }) {
  const [lines, setLines] = useState(TRANSCRIPT.slice(0, 3))
  const scrollRef = useRef<HTMLDivElement>(null)
  const idxRef = useRef(3)

  useEffect(() => {
    if (muted) return
    const t = setInterval(() => {
      if (idxRef.current < TRANSCRIPT.length) setLines(l => [...l, TRANSCRIPT[idxRef.current++]])
      else { idxRef.current = 0; setLines([TRANSCRIPT[0]]) }
    }, 2800)
    return () => clearInterval(t)
  }, [muted])

  useEffect(() => { if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight }, [lines])

  return (
    <div ref={scrollRef} style={{ height: 200, overflowY: 'auto', padding: '14px 18px', display: 'flex', flexDirection: 'column', gap: 10 }}>
      {lines.map((l, i) => (
        <div key={i} style={{ display: 'flex', gap: 9, alignItems: 'flex-start', opacity: 0.65 + 0.35 * (i / lines.length) }}>
          <div style={{ flexShrink: 0, width: 24, height: 24, borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center', background: l.who === 'robot' ? 'oklch(0.35 0.10 145 / 0.15)' : 'rgba(0,0,0,0.07)', fontSize: 9, fontWeight: 700, color: l.who === 'robot' ? 'oklch(0.35 0.10 145)' : 'oklch(0.58 0.02 200)' }}>
            {l.who === 'robot' ? 'E' : userName[0]?.toUpperCase() || 'U'}
          </div>
          <div style={{ fontSize: 13.5, color: 'oklch(0.13 0.01 145)', lineHeight: 1.55, paddingTop: 3 }}>{l.text}</div>
          {i === lines.length - 1 && !muted && (
            <div style={{ display: 'flex', gap: 3, alignItems: 'center', paddingTop: 10, marginLeft: 'auto', flexShrink: 0 }}>
              {[0, 1, 2].map(j => <div key={j} className="e-blink" style={{ width: 3, height: 3, borderRadius: '50%', background: 'oklch(0.35 0.10 145)', animationDelay: `${j * 0.2}s` }} />)}
            </div>
          )}
        </div>
      ))}
      {muted && <div style={{ textAlign: 'center', fontSize: 13, color: 'oklch(0.58 0.02 200)', padding: '16px 0' }}>Microphone muted</div>}
    </div>
  )
}

// ── Hardware Mode ──────────────────────────────────────────────────────────
function HardwareMode({ userName }: { userName: string }) {
  const [muted, setMuted] = useState(false)
  const [sleeping, setSleeping] = useState(false)
  const [view, setView] = useState<'watch' | 'camera'>('watch')
  const [showOverride, setShowOverride] = useState(false)
  const [watchData, setWatchData] = useState<WatchData>({ hr: 72, battery: 82, steps: 4280, temp: 36.6 })
  const handleSleep = () => { const s = !sleeping; setSleeping(s); setMuted(s) }

  const cardRow: React.CSSProperties = { marginBottom: 12 }
  const barBg: React.CSSProperties = { height: 5, background: 'rgba(0,0,0,0.07)', borderRadius: 99, overflow: 'hidden' }

  return (
    <div className="e-anim-in" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 18, alignItems: 'start' }}>
      {/* Watch / Camera panel */}
      <div className="e-glass-card" style={{ overflow: 'hidden' }}>
        <div style={{ padding: '14px 18px 12px', borderBottom: '1px solid rgba(0,0,0,0.06)', display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 10 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <span style={{ fontSize: 15 }}>{view === 'watch' ? '⌚' : '📷'}</span>
            <span style={{ fontSize: 14, fontWeight: 600 }}>{view === 'watch' ? 'Companion Watch' : 'OAK-D Vision'}</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            {view === 'watch' && (
              <button onClick={() => setShowOverride(true)} style={{ padding: '5px 12px', borderRadius: 99, border: '1.5px solid oklch(0.35 0.10 145 / 0.35)', background: 'oklch(0.35 0.10 145 / 0.08)', color: 'oklch(0.35 0.10 145)', fontFamily: 'inherit', fontSize: 12, fontWeight: 600, cursor: 'pointer' }}>
                ⚙ Override
              </button>
            )}
            <div style={{ display: 'flex', background: 'rgba(0,0,0,0.06)', borderRadius: 99, padding: 3, gap: 2 }}>
              {(['watch', 'camera'] as const).map(v => (
                <button key={v} onClick={() => setView(v)} style={{ padding: '4px 10px', borderRadius: 99, border: 'none', cursor: 'pointer', fontFamily: 'inherit', fontSize: 12, fontWeight: 600, background: view === v ? 'white' : 'transparent', boxShadow: view === v ? '0 1px 4px rgba(0,0,0,0.10)' : 'none', transition: 'all .18s' }}>
                  {v === 'watch' ? '⌚' : '📷'}
                </button>
              ))}
            </div>
          </div>
        </div>
        {view === 'watch' ? <WatchVideo data={watchData} /> : (
          <div style={{ position: 'relative', aspectRatio: '4/3', background: 'oklch(0.12 0.01 145)', overflow: 'hidden' }}>
            {sleeping ? (
              <div style={{ position: 'absolute', inset: 0, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 12, color: 'oklch(0.40 0.04 145)' }}>
                <IcMoon size={36} /><span style={{ fontSize: 13, fontWeight: 500 }}>Robot sleeping</span>
              </div>
            ) : (
              <>
                <svg width="100%" height="100%" style={{ position: 'absolute', inset: 0 }}>
                  <defs><pattern id="camGrid" width="32" height="32" patternUnits="userSpaceOnUse"><rect width="32" height="32" fill="oklch(0.10 0.01 145)" /><rect width="16" height="32" fill="oklch(0.13 0.01 145)" /></pattern></defs>
                  <rect width="100%" height="100%" fill="url(#camGrid)" />
                </svg>
                <div style={{ position: 'absolute', inset: 0, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 8, color: 'oklch(0.28 0.03 145)' }}>
                  <IcCamera size={32} /><span style={{ fontSize: 11, fontWeight: 500, fontFamily: 'monospace' }}>oak-d camera feed</span>
                </div>
                {[0, 1, 2, 3].map(i => (
                  <div key={i} style={{ position: 'absolute', top: i < 2 ? 14 : 'auto', bottom: i >= 2 ? 14 : 'auto', left: i % 2 === 0 ? 14 : 'auto', right: i % 2 !== 0 ? 14 : 'auto', width: 18, height: 18, borderTop: i < 2 ? '2px solid oklch(0.35 0.10 145)' : 'none', borderBottom: i >= 2 ? '2px solid oklch(0.35 0.10 145)' : 'none', borderLeft: i % 2 === 0 ? '2px solid oklch(0.35 0.10 145)' : 'none', borderRight: i % 2 !== 0 ? '2px solid oklch(0.35 0.10 145)' : 'none', opacity: 0.6 }} />
                ))}
                <div style={{ position: 'absolute', top: 12, right: 12, display: 'flex', alignItems: 'center', gap: 5, padding: '3px 8px', borderRadius: 99, background: 'rgba(0,0,0,0.55)' }}>
                  <div style={{ width: 6, height: 6, borderRadius: '50%', background: '#ef4444', animation: 'elaraPulse 1.5s ease infinite' }} />
                  <span style={{ fontSize: 9, fontWeight: 700, color: 'white', letterSpacing: '0.06em' }}>REC</span>
                </div>
              </>
            )}
          </div>
        )}
      </div>

      {showOverride && <OverrideModal data={watchData} onApply={setWatchData} onClose={() => setShowOverride(false)} />}

      {/* Right column */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
        {/* Transcript */}
        <div className="e-glass-card">
          <div style={{ padding: '16px 20px 12px', borderBottom: '1px solid rgba(0,0,0,0.06)', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, color: 'oklch(0.35 0.10 145)' }}>
              <IcWave size={16} /><span style={{ fontSize: 14, fontWeight: 600, color: 'oklch(0.13 0.01 145)' }}>Live Transcript</span>
            </div>
            {muted && <span style={{ fontSize: 12, fontWeight: 600, padding: '3px 9px', borderRadius: 99, background: 'rgba(0,0,0,0.07)', color: 'oklch(0.58 0.02 200)' }}>Muted</span>}
          </div>
          <LiveTranscript muted={muted} userName={userName} />
        </div>

        {/* Controls */}
        <div className="e-glass-card" style={{ padding: '18px 20px' }}>
          <div style={{ fontSize: 11, fontWeight: 700, color: 'oklch(0.58 0.02 200)', marginBottom: 12, textTransform: 'uppercase', letterSpacing: '0.07em' }}>Robot Controls</div>
          <div style={{ display: 'flex', gap: 10 }}>
            {[
              { label: muted ? 'Unmute Mic' : 'Mute Mic', icon: <IcMic size={14} off={muted} />, active: muted, color: 'oklch(0.35 0.10 145)', onClick: () => setMuted(m => !m) },
              { label: sleeping ? 'Wake Up' : 'Sleep Mode', icon: <IcMoon size={14} />, active: sleeping, color: 'oklch(0.38 0.10 55)', onClick: handleSleep },
            ].map(b => (
              <button key={b.label} onClick={b.onClick} style={{ flex: 1, padding: 11, border: `1.5px solid ${b.active ? `${b.color} / 0.4` : 'rgba(0,0,0,0.10)'}`, borderColor: b.active ? 'rgba(74,144,80,0.4)' : 'rgba(0,0,0,0.10)', background: b.active ? 'oklch(0.35 0.10 145 / 0.10)' : 'rgba(255,255,255,0.5)', color: b.active ? b.color : 'oklch(0.36 0.02 200)', cursor: 'pointer', fontFamily: 'inherit', fontSize: 13, fontWeight: 600, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 7, backdropFilter: 'blur(8px)', transition: 'all .18s', borderRadius: 10 }}>
                {b.icon}{b.label}
              </button>
            ))}
          </div>
        </div>

        {/* System status */}
        <div className="e-glass-card" style={{ padding: '18px 20px' }}>
          <div style={{ fontSize: 11, fontWeight: 700, color: 'oklch(0.58 0.02 200)', marginBottom: 14, textTransform: 'uppercase', letterSpacing: '0.07em' }}>System Status</div>
          {[{ label: 'Battery', val: watchData.battery, c: 'oklch(0.50 0.15 145)' }, { label: 'WiFi Signal', val: 95, c: 'oklch(0.35 0.10 145)' }, { label: 'Camera', val: 100, c: 'oklch(0.35 0.10 145)' }].map(s => (
            <div key={s.label} style={cardRow}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 5, fontSize: 13 }}>
                <span style={{ color: 'oklch(0.36 0.02 200)', fontWeight: 500 }}>{s.label}</span>
                <span style={{ color: 'oklch(0.58 0.02 200)', fontWeight: 600 }}>{s.val}%</span>
              </div>
              <div style={barBg}><div style={{ height: '100%', width: `${s.val}%`, background: s.c, borderRadius: 99, transition: 'width .6s ease' }} /></div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

// ── Browser Chat Mode ──────────────────────────────────────────────────────
function BrowserChatMode({ userName, sessionId }: { userName: string; sessionId: string | null }) {
  const [messages, setMessages] = useState<Message[]>([
    { role: 'assistant', text: `Hi ${userName}! I'm Elara. How can I help you today?`, ts: new Date() }
  ])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [userCity, setUserCity] = useState<string>('')
  const scrollRef = useRef<HTMLDivElement>(null)

  useEffect(() => { if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight }, [messages])

  useEffect(() => {
    if (!navigator.geolocation) return
    navigator.geolocation.getCurrentPosition(async (pos) => {
      try {
        const { latitude, longitude } = pos.coords
        const res = await fetch(
          `https://nominatim.openstreetmap.org/reverse?lat=${latitude}&lon=${longitude}&format=json`,
          { headers: { 'User-Agent': 'ELARA-Companion/1.0' } }
        )
        const data = await res.json()
        const addr = data.address || {}
        const city = addr.city || addr.town || addr.village || addr.suburb || addr.county || ''
        const state = addr.state || ''
        setUserCity(city ? `${city}${state ? ', ' + state : ''}` : '')
      } catch { /* location optional — silently skip */ }
    }, () => { /* permission denied — silently skip */ })
  }, [])

  const send = useCallback(async () => {
    const txt = input.trim()
    if (!txt || loading) return
    setInput('')
    const userMsg: Message = { role: 'user', text: txt, ts: new Date() }
    setMessages(m => [...m, userMsg])
    setLoading(true)
    try {
      const allMsgs = [...messages, userMsg].map(m => ({ role: m.role, content: m.text }))
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: allMsgs,
          agentState: {
            sessionId,
            userToken: typeof window !== 'undefined' ? localStorage.getItem('memora_token') : null,
            location: userCity || null
          }
        })
      }).catch(() => null)
      let reply: string | undefined
      if (res?.ok) { const d = await res.json().catch(() => null); reply = d?.message?.content }
      if (!reply) {
        await new Promise(r => setTimeout(r, 800 + Math.random() * 600))
        reply = ["That sounds lovely. Tell me more.", "I'll make a note of that.", "Of course! Happy to help.", "Shall we plan it together?", "I hear you. Want me to remind you later?"][Math.floor(Math.random() * 5)]
      }
      setMessages(m => [...m, { role: 'assistant', text: reply!, ts: new Date() }])
    } finally { setLoading(false) }
  }, [input, loading, messages, sessionId])

  const fmt = (ts: Date) => ts.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })

  return (
    <div className="e-anim-in" style={{ display: 'flex', flexDirection: 'column', gap: 12, height: 'calc(100vh - 70px - 130px)', minHeight: 420 }}>
      <div style={{ background: 'oklch(0.96 0.05 55 / 0.65)', border: '1px solid oklch(0.88 0.07 55 / 0.6)', borderRadius: 12, padding: '10px 16px', backdropFilter: 'blur(12px)', display: 'flex', alignItems: 'center', gap: 10, fontSize: 13, fontWeight: 500, color: 'oklch(0.38 0.10 55)', flexShrink: 0 }}>
        <IcMic size={14} off={true} /> Hardware microphone muted. Chatting via browser.
      </div>

      <div className="e-glass-card-strong" style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        <div ref={scrollRef} style={{ flex: 1, overflowY: 'auto', padding: '22px 24px', display: 'flex', flexDirection: 'column', gap: 14 }}>
          {messages.map((m, i) => (
            <div key={i} style={{ display: 'flex', flexDirection: m.role === 'user' ? 'row-reverse' : 'row', gap: 9, alignItems: 'flex-end' }}>
              {m.role === 'assistant' && (
                <div style={{ width: 30, height: 30, borderRadius: '50%', background: 'oklch(0.35 0.10 145)', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0, marginBottom: 2, boxShadow: '0 2px 8px oklch(0.35 0.10 145 / 0.30)' }}>
                  <span style={{ fontSize: 10, fontWeight: 700, color: 'white' }}>E</span>
                </div>
              )}
              <div style={{ maxWidth: '66%', display: 'flex', flexDirection: 'column', gap: 3, alignItems: m.role === 'user' ? 'flex-end' : 'flex-start' }}>
                <div style={{ padding: '12px 15px', borderRadius: m.role === 'user' ? '18px 18px 4px 18px' : '18px 18px 18px 4px', background: m.role === 'user' ? 'oklch(0.35 0.10 145)' : 'rgba(255,255,255,0.70)', backdropFilter: m.role === 'user' ? 'none' : 'blur(8px)', border: m.role === 'user' ? 'none' : '1px solid rgba(255,255,255,0.6)', color: m.role === 'user' ? 'white' : 'oklch(0.13 0.01 145)', fontSize: 14.5, lineHeight: 1.55, boxShadow: m.role === 'user' ? '0 2px 12px oklch(0.35 0.10 145 / 0.25)' : '0 1px 8px rgba(0,0,0,0.06)' }}>
                  {m.text}
                </div>
                <div style={{ fontSize: 11, color: 'oklch(0.58 0.02 200)', padding: '0 4px' }}>{fmt(m.ts)}</div>
              </div>
            </div>
          ))}
          {loading && (
            <div style={{ display: 'flex', gap: 9, alignItems: 'flex-end' }}>
              <div style={{ width: 30, height: 30, borderRadius: '50%', background: 'oklch(0.35 0.10 145)', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
                <span style={{ fontSize: 10, fontWeight: 700, color: 'white' }}>E</span>
              </div>
              <div style={{ padding: '12px 18px', borderRadius: '18px 18px 18px 4px', background: 'rgba(255,255,255,0.70)', backdropFilter: 'blur(8px)', border: '1px solid rgba(255,255,255,0.6)', display: 'flex', gap: 4, alignItems: 'center' }}>
                {[0, 1, 2].map(j => <div key={j} className="e-blink" style={{ width: 6, height: 6, borderRadius: '50%', background: 'oklch(0.58 0.02 200)', animationDelay: `${j * 0.2}s` }} />)}
              </div>
            </div>
          )}
        </div>
        <div style={{ borderTop: '1px solid rgba(0,0,0,0.07)', padding: '14px 18px', display: 'flex', gap: 10, alignItems: 'flex-end', background: 'rgba(255,255,255,0.30)', backdropFilter: 'blur(8px)' }}>
          <textarea
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send() } }}
            placeholder="Message Elara…"
            rows={1}
            className="dash-field-input"
            style={{ flex: 1, resize: 'none', maxHeight: 110, overflowY: 'auto', lineHeight: 1.5 }}
          />
          <button onClick={send} disabled={!input.trim() || loading} style={{ width: 44, height: 44, padding: 0, borderRadius: '50%', background: 'oklch(0.35 0.10 145)', border: 'none', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0, color: 'white', boxShadow: '0 2px 12px oklch(0.35 0.10 145 / 0.30)', opacity: !input.trim() || loading ? 0.5 : 1, transition: 'opacity .15s' }}>
            <IcSend size={16} />
          </button>
        </div>
      </div>
    </div>
  )
}

// ── Memories Panel ─────────────────────────────────────────────────────────
function MemoriesPanel({ onClose, sessionId }: { onClose: () => void; sessionId: string | null }) {
  const [mems, setMems] = useState<MemoryItem[] | null>(null)

  useEffect(() => {
    if (!sessionId) { setTimeout(() => setMems([]), 600); return }
    fetch(`/api/memories/${sessionId}`)
      .then(r => r.json())
      .then(d => {
        const items: MemoryItem[] = []
        const facts: any[] = d.facts || []
        facts.slice(0, 10).forEach((f: any, i: number) => items.push({ id: i, date: 'Recent', text: f.document || String(f) }))
        setMems(items.length ? items : [])
      })
      .catch(() => setMems([]))
  }, [sessionId])

  return (
    <div className="e-side-panel" onClick={e => e.target === e.currentTarget && onClose()}>
      <div className="e-side-inner e-slide-in-end" style={{ width: 400 }}>
        <div style={{ padding: '24px 28px', borderBottom: '1px solid rgba(0,0,0,0.07)', display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexShrink: 0 }}>
          <div>
            <div style={{ fontSize: 18, fontWeight: 700, letterSpacing: '-0.02em' }}>Memories</div>
            <div style={{ fontSize: 13, color: 'oklch(0.58 0.02 200)', marginTop: 2 }}>What Elara remembers</div>
          </div>
          <button onClick={onClose} style={{ background: 'rgba(0,0,0,0.06)', border: 'none', cursor: 'pointer', color: 'oklch(0.58 0.02 200)', padding: 8, borderRadius: 10, display: 'flex' }}>
            <IcX size={18} />
          </button>
        </div>
        <div style={{ flex: 1, overflowY: 'auto', padding: '16px 28px', display: 'flex', flexDirection: 'column', gap: 10 }}>
          {!mems ? (
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, color: 'oklch(0.58 0.02 200)', fontSize: 14, marginTop: 24 }}>
              <div style={{ width: 16, height: 16, border: '2px solid rgba(0,0,0,0.1)', borderTopColor: 'oklch(0.35 0.10 145)', borderRadius: '50%', animation: 'elaraSpin .7s linear infinite' }} />
              Loading memories…
            </div>
          ) : mems.length === 0 ? (
            <div style={{ fontSize: 14, color: 'oklch(0.58 0.02 200)', marginTop: 24 }}>No memories stored yet.</div>
          ) : mems.map(m => (
            <div key={m.id} style={{ padding: '14px 16px', background: 'rgba(255,255,255,0.6)', borderRadius: 12, border: '1px solid rgba(0,0,0,0.07)', backdropFilter: 'blur(8px)' }}>
              <div style={{ fontSize: 11, color: 'oklch(0.35 0.10 145)', fontWeight: 700, marginBottom: 5, textTransform: 'uppercase', letterSpacing: '0.06em' }}>{m.date}</div>
              <div style={{ fontSize: 14, color: 'oklch(0.13 0.01 145)', lineHeight: 1.55 }}>{m.text}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

// ── Profile Panel ──────────────────────────────────────────────────────────
function ProfilePanel({ onClose, user, onLogout }: { onClose: () => void; user: any; onLogout: () => void }) {
  const [activeSection, setActiveSection] = useState('profile')

  const sections = [
    { id: 'profile', label: 'Profile', icon: <IcUser size={18} /> },
    { id: 'account', label: 'Account', icon: <IcLock size={18} /> },
    { id: 'device', label: 'Device', icon: <IcPhone size={18} /> },
    { id: 'support', label: 'Support', icon: <IcHelp size={18} /> },
  ]

  const renderContent = () => {
    switch (activeSection) {
      case 'profile': return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 16, padding: 16, background: 'rgba(255,255,255,0.5)', borderRadius: 14, border: '1px solid rgba(0,0,0,0.07)' }}>
            <div style={{ width: 52, height: 52, borderRadius: '50%', background: 'oklch(0.35 0.10 145)', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0, boxShadow: '0 3px 12px oklch(0.35 0.10 145 / 0.35)' }}>
              <span style={{ fontSize: 18, fontWeight: 700, color: 'white' }}>{user?.fullName?.split(' ').map((n: string) => n[0]).join('').slice(0, 2).toUpperCase() || 'U'}</span>
            </div>
            <div>
              <div style={{ fontWeight: 700, fontSize: 16 }}>{user?.fullName || 'User'}</div>
              <div style={{ fontSize: 13, color: 'oklch(0.58 0.02 200)', marginTop: 2 }}>{user?.email || ''}</div>
            </div>
          </div>
          {[['Language', user?.preferredLanguage || '—'], ['Age', user?.age || '—']].map(([k, v]) => (
            <div key={k} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '12px 0', borderBottom: '1px solid rgba(0,0,0,0.06)' }}>
              <span style={{ fontSize: 13, color: 'oklch(0.58 0.02 200)', fontWeight: 500 }}>{k}</span>
              <span style={{ fontSize: 14, color: 'oklch(0.13 0.01 145)', fontWeight: 500 }}>{String(v)}</span>
            </div>
          ))}
        </div>
      )
      case 'account': return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
          {[{ label: 'Subscription', val: 'Elara Premium', badge: 'Active' }, { label: 'Member since', val: '2025' }].map(r => (
            <div key={r.label} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '13px 14px', background: 'rgba(255,255,255,0.5)', borderRadius: 12, border: '1px solid rgba(0,0,0,0.07)' }}>
              <span style={{ fontSize: 13, color: 'oklch(0.36 0.02 200)', fontWeight: 500 }}>{r.label}</span>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <span style={{ fontSize: 13, color: 'oklch(0.13 0.01 145)', fontWeight: 600 }}>{r.val}</span>
                {'badge' in r && r.badge && <span style={{ fontSize: 11, fontWeight: 700, padding: '2px 8px', borderRadius: 99, background: 'oklch(0.88 0.08 145 / 0.4)', color: 'oklch(0.30 0.10 145)' }}>{r.badge}</span>}
              </div>
            </div>
          ))}
          <div style={{ height: 4 }} />
          <button onClick={onLogout} style={{ padding: 10, borderRadius: 10, border: '1.5px solid rgba(239,68,68,0.3)', background: 'rgba(239,68,68,0.06)', color: '#dc2626', fontFamily: 'inherit', fontSize: 13, fontWeight: 600, cursor: 'pointer' }}>Sign Out</button>
        </div>
      )
      case 'device': return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
          {[['Robot Model', 'Elara Gen 2'], ['Firmware', 'v2.4.1 — Up to date'], ['Serial No.', 'ELR-20240114-002'], ['Last sync', 'Today, 9:42 AM']].map(([k, v]) => (
            <div key={k} style={{ display: 'flex', justifyContent: 'space-between', padding: '12px 0', borderBottom: '1px solid rgba(0,0,0,0.06)' }}>
              <span style={{ fontSize: 13, color: 'oklch(0.58 0.02 200)', fontWeight: 500 }}>{k}</span>
              <span style={{ fontSize: 13, color: 'oklch(0.13 0.01 145)', fontWeight: 600 }}>{v}</span>
            </div>
          ))}
        </div>
      )
      case 'support': return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
          {[{ icon: '📖', title: 'User Guide', sub: 'Setup, tips & daily use' }, { icon: '💬', title: 'Live Chat Support', sub: 'Mon–Fri, 9am–6pm' }, { icon: '🔒', title: 'Privacy & Data', sub: 'Manage your data' }].map(r => (
            <button key={r.title} style={{ display: 'flex', alignItems: 'center', gap: 14, padding: 14, borderRadius: 12, border: '1px solid rgba(0,0,0,0.07)', background: 'rgba(255,255,255,0.5)', cursor: 'pointer', textAlign: 'left', width: '100%', fontFamily: 'inherit' }}>
              <span style={{ fontSize: 22, flexShrink: 0 }}>{r.icon}</span>
              <div>
                <div style={{ fontSize: 14, fontWeight: 600, color: 'oklch(0.13 0.01 145)' }}>{r.title}</div>
                <div style={{ fontSize: 12, color: 'oklch(0.58 0.02 200)', marginTop: 2 }}>{r.sub}</div>
              </div>
              <IcChevron />
            </button>
          ))}
        </div>
      )
      default: return null
    }
  }

  return (
    <div className="e-side-panel" onClick={e => e.target === e.currentTarget && onClose()}>
      <div className="e-side-inner e-slide-in-end" style={{ width: 420 }}>
        <div style={{ padding: '22px 24px 16px', borderBottom: '1px solid rgba(0,0,0,0.07)', display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexShrink: 0 }}>
          <div>
            <div style={{ fontSize: 18, fontWeight: 700, letterSpacing: '-0.02em' }}>Settings</div>
            <div style={{ fontSize: 13, color: 'oklch(0.58 0.02 200)', marginTop: 2 }}>Manage your profile & device</div>
          </div>
          <button onClick={onClose} style={{ background: 'rgba(0,0,0,0.06)', border: 'none', cursor: 'pointer', color: 'oklch(0.58 0.02 200)', padding: 8, borderRadius: 10, display: 'flex' }}>
            <IcX size={18} />
          </button>
        </div>
        <div style={{ display: 'flex', gap: 2, padding: '12px 16px', borderBottom: '1px solid rgba(0,0,0,0.07)', flexShrink: 0 }}>
          {sections.map(s => (
            <button key={s.id} onClick={() => setActiveSection(s.id)} style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 5, padding: '10px 6px', borderRadius: 12, border: 'none', cursor: 'pointer', fontFamily: 'inherit', background: activeSection === s.id ? 'oklch(0.35 0.10 145 / 0.10)' : 'transparent', color: activeSection === s.id ? 'oklch(0.35 0.10 145)' : 'oklch(0.58 0.02 200)', transition: 'all .18s' }}>
              {s.icon}
              <span style={{ fontSize: 11, fontWeight: 600, letterSpacing: '0.02em' }}>{s.label}</span>
            </button>
          ))}
        </div>
        <div key={activeSection} className="e-anim-in" style={{ flex: 1, overflowY: 'auto', padding: '20px 24px' }}>
          <div style={{ fontSize: 16, fontWeight: 700, letterSpacing: '-0.015em', marginBottom: 16 }}>{sections.find(s => s.id === activeSection)?.label}</div>
          {renderContent()}
        </div>
      </div>
    </div>
  )
}

// ── Dashboard (main export) ────────────────────────────────────────────────
export function Dashboard() {
  const { user, logout, sessionId } = useAuth()
  const [mode, setMode] = useState<'hardware' | 'chat'>('hardware')
  const [showMemories, setShowMemories] = useState(false)
  const [showProfile, setShowProfile] = useState(false)

  const firstName = user?.fullName?.split(' ')[0] || 'there'
  const initials = user?.fullName?.split(' ').map((n: string) => n[0]).join('').slice(0, 2).toUpperCase() || 'U'

  const today = new Date().toLocaleDateString('en-US', { weekday: 'long', month: 'long', day: 'numeric' })

  return (
    <div className="dash-root">
      {/* Animated blob background */}
      <div className="dash-bg" aria-hidden>
        <div style={{ position: 'absolute', width: 600, height: 600, top: -100, left: -100, background: 'oklch(0.80 0.10 145 / 0.55)', borderRadius: '50%', filter: 'blur(70px)', animation: 'elaraBlob1 18s ease-in-out infinite alternate', pointerEvents: 'none' }} />
        <div style={{ position: 'absolute', width: 500, height: 500, bottom: -80, right: -80, background: 'oklch(0.72 0.08 180 / 0.45)', borderRadius: '50%', filter: 'blur(70px)', animation: 'elaraBlob2 22s ease-in-out infinite alternate', pointerEvents: 'none' }} />
        <div style={{ position: 'absolute', width: 380, height: 380, top: '50%', left: '55%', background: 'oklch(0.85 0.07 115 / 0.40)', borderRadius: '50%', filter: 'blur(70px)', animation: 'elaraBlob3 16s ease-in-out infinite alternate', pointerEvents: 'none' }} />
        <div style={{ position: 'absolute', width: 280, height: 280, top: '20%', right: '20%', background: 'oklch(0.90 0.05 160 / 0.35)', borderRadius: '50%', filter: 'blur(70px)', animation: 'elaraBlob4 20s ease-in-out infinite alternate', pointerEvents: 'none' }} />
      </div>

      <div className="dash-shell">
        {/* Header */}
        <header className="dash-header">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src="/elara_green.png" alt="Elara" style={{ height: 50, width: 'auto', objectFit: 'contain', objectPosition: 'left', marginRight: 'auto' }} />

          <div style={{ display: 'flex', alignItems: 'center', gap: 7, padding: '6px 14px', borderRadius: 99, background: 'rgba(34,197,94,0.12)', border: '1px solid rgba(34,197,94,0.25)' }}>
            <div className="e-pulse-dot" style={{ width: 7, height: 7, borderRadius: '50%', background: 'oklch(0.55 0.17 145)' }} />
            <span style={{ fontSize: 13, fontWeight: 600, color: 'oklch(0.32 0.12 145)' }}>Robot Connected</span>
          </div>

          <button onClick={() => setShowMemories(s => !s)} style={{ display: 'flex', alignItems: 'center', gap: 7, padding: '8px 16px', borderRadius: 99, border: `1.5px solid ${showMemories ? 'oklch(0.35 0.10 145 / 0.4)' : 'rgba(0,0,0,0.12)'}`, background: showMemories ? 'oklch(0.35 0.10 145 / 0.10)' : 'rgba(255,255,255,0.5)', color: showMemories ? 'oklch(0.35 0.10 145)' : 'oklch(0.36 0.02 200)', cursor: 'pointer', fontFamily: 'inherit', fontSize: 13, fontWeight: 600, backdropFilter: 'blur(8px)', transition: 'all .18s' }}>
            <IcBrain size={15} /> Memories
          </button>

          <div onClick={() => setShowProfile(s => !s)} style={{ width: 38, height: 38, borderRadius: '50%', background: 'oklch(0.35 0.10 145)', display: 'flex', alignItems: 'center', justifyContent: 'center', cursor: 'pointer', flexShrink: 0, boxShadow: '0 2px 10px oklch(0.35 0.10 145 / 0.35)', transition: 'transform .15s, box-shadow .15s' }}>
            <span style={{ fontSize: 13, fontWeight: 700, color: 'white', letterSpacing: '0.02em' }}>{initials}</span>
          </div>
        </header>

        {/* Main */}
        <main className="dash-main">
          <div style={{ marginBottom: 28 }}>
            <h1 style={{ fontSize: 'clamp(20px, 2.2vw, 30px)', fontWeight: 700, letterSpacing: '-0.03em', color: 'oklch(0.13 0.01 145)' }}>
              Good {new Date().getHours() < 12 ? 'morning' : new Date().getHours() < 17 ? 'afternoon' : 'evening'}, {firstName}. 👋
            </h1>
            <p style={{ fontSize: 14, color: 'oklch(0.58 0.02 200)', marginTop: 5 }}>
              {today} · Elara is active and listening.
            </p>
          </div>

          {/* Mode toggle */}
          <div style={{ display: 'flex', justifyContent: 'center', marginBottom: 28 }}>
            <div style={{ display: 'inline-flex', background: 'rgba(255,255,255,0.45)', backdropFilter: 'blur(16px)', border: '1px solid rgba(255,255,255,0.65)', borderRadius: 99, padding: 4, gap: 4, boxShadow: '0 2px 16px rgba(0,0,0,0.08)' }}>
              {(['hardware', 'chat'] as const).map(m => (
                <button key={m} onClick={() => setMode(m)} style={{ padding: '9px 28px', borderRadius: 99, border: 'none', cursor: 'pointer', fontFamily: 'inherit', fontSize: 14, fontWeight: 600, transition: 'all .22s', background: mode === m ? 'oklch(0.35 0.10 145)' : 'transparent', color: mode === m ? 'white' : 'oklch(0.58 0.02 200)', boxShadow: mode === m ? '0 2px 14px oklch(0.35 0.10 145 / 0.40)' : 'none' }}>
                  {m === 'hardware' ? 'Hardware Mode' : 'Browser Chat'}
                </button>
              ))}
            </div>
          </div>

          {mode === 'hardware'
            ? <HardwareMode userName={firstName} />
            : <BrowserChatMode userName={firstName} sessionId={sessionId} />}
        </main>
      </div>

      {showMemories && <MemoriesPanel onClose={() => setShowMemories(false)} sessionId={sessionId} />}
      {showProfile && <ProfilePanel onClose={() => setShowProfile(false)} user={user} onLogout={() => { logout(); setShowProfile(false) }} />}
    </div>
  )
}

export { Dashboard as SeniorChat }
