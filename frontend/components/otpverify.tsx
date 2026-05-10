"use client";

import { useRef, useState, KeyboardEvent, ClipboardEvent, ChangeEvent } from "react";
import { useRouter } from "next/navigation";

type Status = { type: "success" | "error" | "loading"; message: string } | null;

function Spinner() { return <span className="e-spinner" /> }

export default function OtpVerify() {
  const [digits, setDigits] = useState<string[]>(Array(6).fill(""));
  const [status, setStatus] = useState<Status>(null);
  const [verified, setVerified] = useState(false);
  const inputRefs = useRef<(HTMLInputElement | null)[]>([]);
  const router = useRouter();

  const otp = digits.join("");
  const canSubmit = otp.length === 6 && !verified && status?.type !== "loading";
  const filled = digits.filter(Boolean).length;

  function updateDigit(index: number, value: string) {
    const digit = value.replace(/\D/g, "").slice(-1);
    setDigits(prev => { const next = [...prev]; next[index] = digit; return next; });
    if (digit && index < 5) inputRefs.current[index + 1]?.focus();
    setStatus(null);
  }

  function handleChange(e: ChangeEvent<HTMLInputElement>, index: number) {
    updateDigit(index, e.target.value);
  }

  function handleKeyDown(e: KeyboardEvent<HTMLInputElement>, index: number) {
    if (e.key === "Backspace" && !digits[index] && index > 0) {
      setDigits(prev => { const next = [...prev]; next[index - 1] = ""; return next; });
      inputRefs.current[index - 1]?.focus();
    }
  }

  function handlePaste(e: ClipboardEvent<HTMLInputElement>) {
    e.preventDefault();
    const pasted = e.clipboardData.getData("text").replace(/\D/g, "").slice(0, 6);
    const next = Array(6).fill("");
    pasted.split("").forEach((ch, i) => (next[i] = ch));
    setDigits(next);
    inputRefs.current[Math.min(pasted.length, 5)]?.focus();
    setStatus(null);
  }

  async function handleVerify() {
    if (!canSubmit) return;
    setStatus({ type: "loading", message: "Verifying…" });
    try {
      const response = await fetch("/api/verify", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          user_id: "USER_01",
          user_input_otp: otp,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "OTP verification failed");
      }

      setStatus({ type: "success", message: "Identity verified! Redirecting…" });
      setVerified(true);
      localStorage.setItem("elara_initial_otp_verified", "true");
      setTimeout(() => router.replace("/login?mode=signup"), 1500);
    } catch (error) {
      setStatus({ type: "error", message: error instanceof Error ? error.message : "OTP verification failed" });
      setTimeout(() => {
        setDigits(Array(6).fill(""));
        setStatus(null);
        inputRefs.current[0]?.focus();
      }, 2000);
    }
  }

  const isError = status?.type === "error";

  return (
    <div className="e-anim-in" style={{ display: "flex", flexDirection: "column", gap: 20 }}>

      {/* Description */}
      <p style={{ fontSize: 14, color: "oklch(0.36 0.02 200)", lineHeight: 1.6, margin: 0 }}>
        Enter the 6-digit code sent to your registered device to continue.
      </p>

      {/* Digit inputs */}
      <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
        <label style={{ fontSize: 13, fontWeight: 500, color: "oklch(0.36 0.02 200)", letterSpacing: "0.01em" }}>
          One-time code
        </label>
        <div style={{ display: "flex", gap: 8 }}>
          {digits.map((digit, i) => (
            <input
              key={i}
              ref={el => { inputRefs.current[i] = el; }}
              type="text"
              inputMode="numeric"
              pattern="[0-9]*"
              maxLength={1}
              value={digit}
              disabled={verified}
              onChange={e => handleChange(e, i)}
              onKeyDown={e => handleKeyDown(e, i)}
              onPaste={i === 0 ? handlePaste : undefined}
              style={{
                flex: 1,
                aspectRatio: "1",
                maxWidth: 52,
                textAlign: "center",
                fontFamily: "var(--font-mono, monospace)",
                fontSize: 22,
                fontWeight: 500,
                border: `1.5px solid ${isError
                  ? "oklch(0.65 0.18 25 / 0.7)"
                  : digit
                  ? "oklch(0.35 0.10 145 / 0.6)"
                  : "rgba(255,255,255,0.55)"}`,
                borderRadius: 10,
                background: isError
                  ? "oklch(0.97 0.03 25 / 0.6)"
                  : digit
                  ? "oklch(0.35 0.10 145 / 0.08)"
                  : "rgba(255,255,255,0.45)",
                backdropFilter: "blur(8px)",
                color: isError ? "oklch(0.45 0.18 25)" : "oklch(0.13 0.01 145)",
                outline: "none",
                transition: "all .18s",
                boxShadow: digit && !isError ? "0 0 0 3px oklch(0.35 0.10 145 / 0.10)" : "none",
                opacity: verified ? 0.5 : 1,
                cursor: verified ? "not-allowed" : "text",
              }}
            />
          ))}
        </div>

        {/* Progress bar */}
        <div style={{ height: 2, background: "rgba(255,255,255,0.3)", borderRadius: 99, overflow: "hidden" }}>
          <div style={{
            height: "100%",
            width: `${(filled / 6) * 100}%`,
            background: isError ? "oklch(0.65 0.18 25)" : "oklch(0.35 0.10 145)",
            borderRadius: 99,
            transition: "width .2s, background .3s",
          }} />
        </div>
      </div>

      {/* Status message */}
      {status && (
        <div style={{
          display: "flex", alignItems: "center", gap: 8,
          fontSize: 13, fontWeight: 500,
          padding: "10px 14px", borderRadius: 10,
          background: status.type === "success"
            ? "oklch(0.35 0.10 145 / 0.12)"
            : status.type === "error"
            ? "oklch(0.65 0.18 25 / 0.10)"
            : "rgba(255,255,255,0.35)",
          border: `1px solid ${status.type === "success"
            ? "oklch(0.35 0.10 145 / 0.30)"
            : status.type === "error"
            ? "oklch(0.65 0.18 25 / 0.30)"
            : "rgba(255,255,255,0.50)"}`,
          color: status.type === "success"
            ? "oklch(0.26 0.09 145)"
            : status.type === "error"
            ? "oklch(0.45 0.18 25)"
            : "oklch(0.36 0.02 200)",
          backdropFilter: "blur(8px)",
        }}>
          {status.type === "loading" ? (
            <Spinner />
          ) : status.type === "success" ? (
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
              <path d="M20 6L9 17l-5-5" />
            </svg>
          ) : (
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
              <circle cx="12" cy="12" r="10" /><path d="M12 8v4m0 4h.01" />
            </svg>
          )}
          {status.message}
        </div>
      )}

      {/* Verify button */}
      <button
        className="e-btn-primary"
        onClick={handleVerify}
        disabled={!canSubmit}
        type="button"
      >
        {status?.type === "loading"
          ? <><Spinner /> Verifying…</>
          : "Verify code →"}
      </button>

      {/* Resend */}
      <p style={{ textAlign: "center", fontSize: 13, color: "oklch(0.58 0.02 200)", margin: 0 }}>
        Didn&apos;t receive a code?{" "}
        <button
          type="button"
          disabled={status?.type === "loading" || verified}
          onClick={() => alert("Resend OTP triggered")}
          style={{
            background: "none", border: "none", cursor: "pointer",
            fontSize: 13, color: "oklch(0.35 0.10 145)", fontWeight: 500,
            padding: "2px 0", textDecoration: "underline", textUnderlineOffset: 3,
            fontFamily: "inherit", opacity: (status?.type === "loading" || verified) ? 0.4 : 1,
          }}
        >
          Resend
        </button>
      </p>

    </div>
  );
}