import type { Metadata } from "next";

import { Recognizer } from "@/components/recognizer";

export const metadata: Metadata = { title: "Recognize" };

export default function RecognizePage() {
  return <Recognizer />;
}
