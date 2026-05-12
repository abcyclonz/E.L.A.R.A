"use client"

import { useAuth } from "@/components/auth-context"
import { Dashboard } from "@/components/senior-chat"
import { useRouter } from "next/navigation"
import { useEffect } from "react"

export default function Page() {
  const { isAuthenticated, authLoading } = useAuth()
  const router = useRouter()

  useEffect(() => {
    if (authLoading || isAuthenticated) return
    const otpVerified = localStorage.getItem('elara_initial_otp_verified') === 'true'
    router.replace(otpVerified ? '/login' : '/verify')
  }, [isAuthenticated, authLoading, router])

  if (authLoading) return null
  if (!isAuthenticated) return null
  return <Dashboard />
}
