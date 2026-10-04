"use client";

import type { ReactNode } from "react";

import { RequireAuth } from "@/lib/auth";

// Every page in this group needs a signed-in user.
export default function AppLayout({ children }: { children: ReactNode }) {
  return (
    <RequireAuth>
      <div className="mx-auto flex w-full max-w-6xl flex-1 flex-col gap-6 px-4 py-8">{children}</div>
    </RequireAuth>
  );
}
