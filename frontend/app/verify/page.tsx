"use client";

import OtpVerify from '@/components/otpverify';
import { useAuth } from '@/components/auth-context';
import { useRouter } from 'next/navigation';
import { useEffect } from 'react';

export default function VerifyOtpPage() {
  const { isAuthenticated, authLoading } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (authLoading) return;
    if (isAuthenticated) {
      router.replace('/');
      return;
    }
    const otpVerified = localStorage.getItem('elara_initial_otp_verified') === 'true';
    if (otpVerified) {
      router.replace('/login?mode=signup');
    }
  }, [authLoading, isAuthenticated, router]);

  if (authLoading) return null;

  return (
    <div className="auth-layout">
      <div className="auth-panel-left">
        <video
          src="/Cute_Robot.mp4"
          autoPlay
          muted
          loop
          playsInline
          style={{
            position: 'absolute',
            inset: 0,
            width: '100%',
            height: '100%',
            objectFit: 'cover',
            objectPosition: 'center',
          }}
        />
        <div className="auth-overlay">
          <div style={{ flexShrink: 0 }}>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src="/elara_white.png"
              alt="Elara"
              style={{
                height: 58,
                width: 'auto',
                objectFit: 'contain',
                objectPosition: 'left',
                filter: 'drop-shadow(0 1px 6px rgba(0,0,0,0.3))',
              }}
            />
          </div>
          <div style={{ flex: 1 }} />
          <div style={{ flexShrink: 0 }}>
            <h1
              style={{
                fontSize: 'clamp(24px, 2.8vw, 40px)',
                fontWeight: 700,
                color: 'white',
                lineHeight: 1.18,
                letterSpacing: '-0.03em',
                marginBottom: 14,
                textShadow: '0 2px 20px rgba(0,0,0,0.4)',
              }}
            >
              Secure your
              <br />
              first access.
            </h1>
            <p
              style={{
                fontSize: 15,
                color: 'rgba(255,255,255,0.75)',
                lineHeight: 1.65,
                maxWidth: 340,
                textShadow: '0 1px 8px rgba(0,0,0,0.4)',
                marginBottom: 8,
              }}
            >
              Complete one-time verification to unlock signup and future logins.
              Press the Button on your Bot to generate the OTP.
            </p>
          </div>
        </div>
      </div>

      <div className="auth-panel-right">
        <div
          style={{
            position: 'absolute',
            width: 480,
            height: 480,
            top: -120,
            right: -100,
            background: 'oklch(0.78 0.10 145 / 0.55)',
            borderRadius: '50%',
            filter: 'blur(65px)',
            zIndex: 0,
            animation: 'elaraBlob1 20s ease-in-out infinite alternate',
            pointerEvents: 'none',
          }}
        />
        <div
          style={{
            position: 'absolute',
            width: 380,
            height: 380,
            bottom: -80,
            left: -80,
            background: 'oklch(0.72 0.08 175 / 0.45)',
            borderRadius: '50%',
            filter: 'blur(65px)',
            zIndex: 0,
            animation: 'elaraBlob2 17s ease-in-out infinite alternate',
            pointerEvents: 'none',
          }}
        />
        <div className="auth-scroll-area">
          <div style={{ width: '100%' }}>
            <div className="auth-glass-card">
              <div style={{ marginBottom: 24 }}>
                <p style={{ fontSize: 13, color: 'oklch(0.58 0.02 200)', marginBottom: 6 }}>
                  Step into something new
                </p>
                <h2
                  style={{
                    fontSize: 'clamp(20px, 2vw, 28px)',
                    fontWeight: 700,
                    letterSpacing: '-0.025em',
                    color: 'oklch(0.13 0.01 145)',
                  }}
                >
                  Verify one-time code
                </h2>
              </div>
              <OtpVerify />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
