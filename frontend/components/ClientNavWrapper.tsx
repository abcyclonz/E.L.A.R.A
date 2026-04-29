"use client";
import { usePathname } from "next/navigation";
import { Navigation } from "@/components/navigation";

export function ClientNavWrapper() {
  const pathname = usePathname();
  const isChatPage = pathname === "/";
  const isAuthenticated = typeof window !== "undefined" && localStorage.getItem("memora_token");
  const showNav = !(isChatPage && isAuthenticated);

  if (!showNav) return null;
  return <Navigation />;
}
