"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState, type FormEvent } from "react";

import { Alert, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { api } from "@/lib/api";
import { useAuth } from "@/lib/auth";

export function AuthForm({ mode }: { mode: "login" | "register" }) {
  const { login } = useAuth();
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const isRegister = mode === "register";

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      if (isRegister) await api.register(email, password);
      await login(email, password);
      router.push("/dashboard");
    } catch (err) {
      setError(err instanceof Error ? err.message : "something went wrong");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="flex flex-1 items-center justify-center px-4 py-16">
      <Card className="w-full max-w-sm">
        <CardHeader>
          <CardTitle>{isRegister ? "Create your account" : "Welcome back"}</CardTitle>
          <CardDescription>
            {isRegister ? "Start recognizing hand signs in your browser." : "Sign in to continue to GestureFlow."}
          </CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={onSubmit} className="flex flex-col gap-4">
            <div className="flex flex-col gap-2">
              <Label htmlFor="email">Email</Label>
              <Input id="email" type="email" autoComplete="email" required value={email}
                     onChange={(e) => setEmail(e.target.value)} />
            </div>
            <div className="flex flex-col gap-2">
              <Label htmlFor="password">Password</Label>
              <Input id="password" type="password" required minLength={isRegister ? 8 : undefined}
                     autoComplete={isRegister ? "new-password" : "current-password"} value={password}
                     onChange={(e) => setPassword(e.target.value)} />
              {isRegister && <p className="text-xs text-muted-foreground">At least 8 characters.</p>}
            </div>
            {error && (
              <Alert variant="destructive">
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}
            <Button type="submit" disabled={busy}>
              {busy ? "Please wait…" : isRegister ? "Create account" : "Sign in"}
            </Button>
            {isRegister && (
              <p className="text-xs text-muted-foreground">
                We store your email and the letters you sign, never camera video.{" "}
                <Link href="/privacy" className="underline underline-offset-4">Privacy</Link>
              </p>
            )}
            <p className="text-center text-sm text-muted-foreground">
              {isRegister ? "Already have an account? " : "New here? "}
              <Link href={isRegister ? "/login" : "/register"} className="text-foreground underline-offset-4 hover:underline">
                {isRegister ? "Sign in" : "Create an account"}
              </Link>
            </p>
          </form>
        </CardContent>
      </Card>
    </div>
  );
}
