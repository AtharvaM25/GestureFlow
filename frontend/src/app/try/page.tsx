import type { Metadata } from "next";

import { Recognizer } from "@/components/recognizer";

export const metadata: Metadata = { title: "Try it" };

// Public: no account needed. Edge mode only, so nothing is sent or stored.
export default function TryPage() {
  return (
    <div className="mx-auto flex w-full max-w-6xl flex-1 flex-col gap-6 px-4 py-8">
      <Recognizer guest />
    </div>
  );
}
