import type { Metadata } from "next";

import { CorpusView } from "@/components/corpus/CorpusView";

export const metadata: Metadata = {
  title: "Corpus · Varagity",
  description: "Upload, ingest, and manage the RAG corpus.",
};

export default function CorpusPage() {
  return <CorpusView />;
}
