"use client";

import { Trash2 } from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState, type FormEvent } from "react";

import { formatDate, PageHeader } from "@/components/page-header";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { api } from "@/lib/api";
import { useAuth } from "@/lib/auth";

export default function SettingsPage() {
  const { user, logout } = useAuth();
  const router = useRouter();
  const [password, setPassword] = useState("");
  const [confirming, setConfirming] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function onDelete(e: FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      await api.deleteAccount(password);
      logout();
      router.push("/");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not delete the account.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
      <PageHeader title="Settings" />

      <Card>
        <CardHeader>
          <CardTitle>Account</CardTitle>
        </CardHeader>
        <CardContent className="flex flex-col gap-1 text-sm">
          <span>{user?.email}</span>
          {user && <span className="text-muted-foreground">Member since {formatDate(user.created_at)}</span>}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Delete account</CardTitle>
          <CardDescription>
            Permanently deletes your account, sessions, letters and calibration. This can&apos;t be undone. See{" "}
            <Link href="/privacy" className="underline underline-offset-4">Privacy</Link> for exactly what is stored.
          </CardDescription>
        </CardHeader>
        <CardContent>
          {!confirming ? (
            <Button variant="destructive" onClick={() => setConfirming(true)}>
              <Trash2 /> Delete my account
            </Button>
          ) : (
            <form onSubmit={onDelete} className="flex max-w-sm flex-col gap-3">
              <div className="flex flex-col gap-2">
                <Label htmlFor="password">Enter your password to confirm</Label>
                <Input id="password" type="password" autoComplete="current-password" required value={password}
                       onChange={(e) => setPassword(e.target.value)} />
              </div>
              {error && (
                <Alert variant="destructive">
                  <AlertDescription>{error}</AlertDescription>
                </Alert>
              )}
              <div className="flex gap-2">
                <Button type="submit" variant="destructive" disabled={busy || !password}>
                  {busy ? "Deleting…" : "Delete permanently"}
                </Button>
                <Button type="button" variant="ghost" onClick={() => setConfirming(false)}>
                  Cancel
                </Button>
              </div>
            </form>
          )}
        </CardContent>
      </Card>
    </>
  );
}
