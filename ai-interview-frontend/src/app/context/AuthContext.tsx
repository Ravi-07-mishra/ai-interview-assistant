"use client";
import { createContext, useContext, useState, useEffect } from "react";
import { useRouter } from "next/navigation";

const AuthContext = createContext<any>(null);

export function AuthProvider({ children }: any) {
  const router = useRouter();
  
  // IMPORTANT FIX:
  const [token, setToken] = useState<string | null | undefined>(undefined);
  const [user, setUser] = useState<any | undefined>(undefined);

  const loading = token === undefined; // <-- TRUE until localStorage loads

  useEffect(() => {
    const savedToken = localStorage.getItem("token");
    const savedUser = localStorage.getItem("user");

    if (savedToken) setToken(savedToken);
    else setToken(null);

    if (savedUser) setUser(JSON.parse(savedUser));
    else setUser(null);
  }, []);

  // -----------------
  // LOGIN
  // -----------------
  async function loginUser(email: string, password: string) {
    const res = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL}/auth/login`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password }),
    });

    const data = await res.json();
    if (!res.ok) throw new Error(data.message || "Login failed");

    localStorage.setItem("token", data.token);
    localStorage.setItem("user", JSON.stringify(data.user));

    setToken(data.token);
    setUser(data.user);

    router.push("/dashboard");
  }

  // -----------------
  // SIGNUP
  // -----------------
  async function signupUser(name: string, email: string, password: string) {
    const res = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL}/auth/signup`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, email, password }),
    });

    const data = await res.json();
    if (!res.ok) throw new Error(data.message || "Signup failed");

    localStorage.setItem("token", data.token);
    localStorage.setItem("user", JSON.stringify(data.user));

    setToken(data.token);
    setUser(data.user);

    router.push("/dashboard");
  }

  // -----------------
  // LOGOUT
  // -----------------
  function logout() {
    localStorage.removeItem("token");
    localStorage.removeItem("user");

    setToken(null);
    setUser(null);

    router.push("/auth/login");
  }

  return (
    <AuthContext.Provider value={{ user, token, loading, loginUser, signupUser, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  return useContext(AuthContext);
}
