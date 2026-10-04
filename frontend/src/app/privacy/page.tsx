import type { Metadata } from "next";
import Link from "next/link";

export const metadata: Metadata = { title: "Privacy" };

// Keep this in step with what the code actually does (backend/models.py, docs/API.md).
export default function PrivacyPage() {
  return (
    <div className="mx-auto flex w-full max-w-3xl flex-col gap-8 px-4 py-12 text-sm leading-relaxed">
      <div className="flex flex-col gap-2">
        <h1 className="text-2xl font-semibold tracking-tight">Privacy</h1>
        <p className="text-muted-foreground">What GestureFlow stores, where, and who can see it.</p>
      </div>

      <section className="flex flex-col gap-2">
        <h2 className="text-base font-semibold">Your camera</h2>
        <p>
          Camera video never leaves your browser. Hand tracking runs on your device and turns each frame into 21 points on
          your hand. Video and images are never uploaded or stored.
        </p>
        <ul className="list-disc space-y-1 pl-5">
          <li>
            <strong>Edge mode</strong> (and <Link href="/try" className="underline underline-offset-4">Try it</Link> without an
            account): recognition runs in your browser too. Nothing is sent anywhere and nothing is saved.
          </li>
          <li>
            <strong>Server mode</strong>: the 21 points are sent to our server to recognize the letter, then discarded. Hand
            points are never stored.
          </li>
        </ul>
      </section>

      <section className="flex flex-col gap-2">
        <h2 className="text-base font-semibold">What we store when you have an account</h2>
        <ul className="list-disc space-y-1 pl-5">
          <li>Your email, and a one-way hash of your password (your password itself can&apos;t be read back).</li>
          <li>Your recognition sessions: the letters you committed, their confidence, timestamps, and any sentence you generated.</li>
          <li>
            If you calibrate: an average of the model&apos;s internal features for each letter (numbers, not pictures or hand
            points).
          </li>
        </ul>
        <p>It&apos;s kept in a PostgreSQL database run by our hosting provider, which encrypts it on disk.</p>
      </section>

      <section className="flex flex-col gap-2">
        <h2 className="text-base font-semibold">Who can see it</h2>
        <ul className="list-disc space-y-1 pl-5">
          <li>You: only your own data. Other users can&apos;t see it.</li>
          <li>The site&apos;s owner, who runs the server and database, for running and fixing the service.</li>
          <li>Our hosting providers run the hardware. They don&apos;t use your data, under their own privacy policies.</li>
        </ul>
      </section>

      <section className="flex flex-col gap-2">
        <h2 className="text-base font-semibold">Sentence generation</h2>
        <p>
          When you press <em>Make a sentence</em>, only the letters of that session (for example H, E, L, L, O) are sent to a
          language model to turn them into a sentence. Online this is a hosted model provider (Groq); no email or account
          details are included. It&apos;s limited per user per hour.
        </p>
      </section>

      <section className="flex flex-col gap-2">
        <h2 className="text-base font-semibold">Deleting your data</h2>
        <p>
          <Link href="/settings" className="underline underline-offset-4">Settings → Delete account</Link> permanently removes
          your account and everything listed above, straight away.
        </p>
      </section>
    </div>
  );
}
