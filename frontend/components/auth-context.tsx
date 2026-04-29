"use client"

import { createContext, useContext, useState, useEffect, ReactNode } from "react"

interface User {
  id: string
  email?: string
  password?: string
  fullName: string
  age: string
  preferredLanguage: string
  background: string
  interests: string[]
  conversationPreferences: string[]
  technologyUsage: string
  conversationGoals: string[]
  additionalInfo: string
}

interface AuthContextType {
  user: User | null
  isAuthenticated: boolean
  authLoading: boolean
  login: (email: string, password: string) => Promise<void>
  logout: () => void
  signup: (userData: Omit<User, 'id'>) => Promise<void>
  sessionId: string | null
}

const AuthContext = createContext<AuthContextType | undefined>(undefined)

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null)
  const [isAuthenticated, setIsAuthenticated] = useState(false)
  const [authLoading, setAuthLoading] = useState(true)
  const [sessionId, setSessionId] = useState<string | null>(null)

  useEffect(() => {
    const storedUser = localStorage.getItem('memora_user')
    const storedToken = localStorage.getItem('memora_token')
    const storedSessionId = localStorage.getItem('memora_session_id')
    if (storedUser && storedToken && storedSessionId) {
      const userData = JSON.parse(storedUser)
      setUser(userData)
      setIsAuthenticated(true)
      setSessionId(storedSessionId)
    }
    setAuthLoading(false)
  }, [])

  const login = async (email: string, password: string) => {
    try {
      const response = await fetch('/api/auth/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ email, password }),
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || 'Login failed')
      }

      const data = await response.json()
      const user: User = {
        id: data.user.id,
        fullName: data.user.full_name,
        age: data.user.age,
        preferredLanguage: data.user.preferred_language,
        background: data.user.background,
        interests: data.user.interests,
        conversationPreferences: data.user.conversation_preferences,
        technologyUsage: data.user.technology_usage,
        conversationGoals: data.user.conversation_goals,
        additionalInfo: data.user.additional_info
      }
      setUser(user)
      setIsAuthenticated(true)
      localStorage.setItem('memora_user', JSON.stringify(user))
      localStorage.setItem('memora_token', data.access_token)
      localStorage.setItem('memora_session_id', user.id)
      setSessionId(user.id)
    } catch (error) {
      throw new Error(error instanceof Error ? error.message : 'Login failed')
    }
  }

  const logout = () => {
    setUser(null)
    setIsAuthenticated(false)
    setSessionId(null)
    localStorage.removeItem('memora_user')
    localStorage.removeItem('memora_token')
    localStorage.removeItem('memora_session_id')
  }

  const signup = async (userData: Omit<User, 'id'>) => {
    try {
      const response = await fetch('/api/auth/signup', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          email: userData.email || 'user@example.com',
          password: userData.password,
          full_name: userData.fullName,
          age: userData.age,
          preferred_language: userData.preferredLanguage,
          background: userData.background,
          interests: userData.interests,
          conversation_preferences: userData.conversationPreferences,
          technology_usage: userData.technologyUsage,
          conversation_goals: userData.conversationGoals,
          additional_info: userData.additionalInfo
        }),
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || 'Signup failed')
      }

      const data = await response.json()
      const newUser: User = {
        id: data.user.id,
        fullName: data.user.full_name,
        age: data.user.age,
        preferredLanguage: data.user.preferred_language,
        background: data.user.background,
        interests: data.user.interests,
        conversationPreferences: data.user.conversation_preferences,
        technologyUsage: data.user.technology_usage,
        conversationGoals: data.user.conversation_goals,
        additionalInfo: data.user.additional_info
      }
      setUser(newUser)
      setIsAuthenticated(true)
      localStorage.setItem('memora_user', JSON.stringify(newUser))
      localStorage.setItem('memora_token', data.access_token)
      localStorage.setItem('memora_session_id', newUser.id)
      setSessionId(newUser.id)
    } catch (error) {
      throw new Error(error instanceof Error ? error.message : 'Signup failed')
    }
  }

  return (
    <AuthContext.Provider value={{ user, isAuthenticated, authLoading, login, logout, signup, sessionId }}>
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  const context = useContext(AuthContext)
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider')
  }
  return context
}
