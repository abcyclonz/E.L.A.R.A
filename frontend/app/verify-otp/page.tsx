"use client";

import OtpVerify from '@/components/otpverify';
import { useAuth } from '@/components/auth-context';
import { useRouter } from 'next/navigation';
import { useEffect } from 'react';

export default function VerifyOtpPage() {
  const { isPendingOTP, authLoading } = useAuth();
  const router = useRouter();

  useEffect(() => {
    // Only allow access if user is pending OTP verification
    if (!authLoading && !isPendingOTP) {
      router.replace('/login');
    }
  }, [isPendingOTP, authLoading, router]);

  if (authLoading) return null;
  if (!isPendingOTP) return null;

  return <OtpVerify />;
}
