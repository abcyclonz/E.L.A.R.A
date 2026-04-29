"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import { Button } from "@/components/ui/button"
import { ThemeToggle } from "@/components/theme-toggle"
import { useAuth } from "@/components/auth-context"
import Image from "next/image"
import { useTheme } from "next-themes"
import { useState, useEffect } from "react"

export function Navigation() {
  const pathname = usePathname()
  const { isAuthenticated, logout } = useAuth()
  const { theme } = useTheme()
  const [mounted, setMounted] = useState(false)
  useEffect(() => { setMounted(true) }, [])

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 shadow">
      <div className="container flex h-16 items-center justify-between">
        <div className="flex items-center space-x-4 ml-4">
          <Link href="/" className="flex items-center space-x-2">
            <div className="relative w-8 h-8 flex-shrink-0">
              {mounted && (
                <Image
                  src={theme === "dark" ? "/memora-dark.png" : "/memora-light.png"}
                  alt="ELARA Logo"
                  width={32}
                  height={32}
                  className="object-contain transition-smooth hover:scale-110"
                  priority
                />
              )}
            </div>
            <span className="text-xl font-bold">ELARA</span>
          </Link>
        </div>

        <div className="flex items-center space-x-4">
          <div className="hidden md:flex items-center space-x-4">
            {!isAuthenticated ? (
              <>
                {pathname === "/" && (
                  <>
                    <Link href="/signup">
                      <Button variant="outline">Sign Up</Button>
                    </Link>
                    <Link href="/login">
                      <Button>Sign In</Button>
                    </Link>
                  </>
                )}
                {pathname === "/login" && (
                  <Link href="/signup">
                    <Button variant="outline">Sign Up</Button>
                  </Link>
                )}
                {pathname === "/signup" && (
                  <Link href="/login">
                    <Button variant="outline">Sign In</Button>
                  </Link>
                )}
              </>
            ) : (
              <Button variant="outline" onClick={logout}>
                Sign Out
              </Button>
            )}
          </div>
          <ThemeToggle />
        </div>
      </div>
    </nav>
  )
}
