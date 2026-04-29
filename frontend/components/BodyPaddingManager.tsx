"use client";
import { usePathname } from "next/navigation";
import { useEffect } from "react";

export function BodyPaddingManager() {
  const pathname = usePathname();
  useEffect(() => {
    document.body.removeAttribute('data-has-nav');
    const isChatPage = pathname === "/";
    const isAuthenticated = typeof window !== "undefined" && localStorage.getItem("memora_token");
    const showNav = !(isChatPage && isAuthenticated);
    if (showNav) {
      document.body.setAttribute('data-has-nav', 'true');
    }
  }, [pathname]);
  return null;
}
