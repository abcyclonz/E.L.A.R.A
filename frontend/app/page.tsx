"use client"

import { useAuth } from "@/components/auth-context"
import { Dashboard } from "@/components/senior-chat"
import { useRouter } from "next/navigation"
import { useEffect } from "react"

export default function Page() {
  const { isAuthenticated, authLoading } = useAuth()
  const router = useRouter()

  useEffect(() => {
    if (!authLoading && !isAuthenticated) router.replace('/login')
  }, [isAuthenticated, authLoading, router])

  if (authLoading) return null
  if (!isAuthenticated) return null
  return <Dashboard />
}
