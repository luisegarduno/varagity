import type { Metadata } from "next";

import { CorpusTabs } from "@/components/corpus/CorpusTabs";

export const metadata: Metadata = {
  title: "Corpus · Varagity",
  description: "Upload, ingest, and manage the RAG corpus and the message graph.",
};

/**
 * The corpus page: the document corpus and the Graph RAG corpus, tabbed.
 *
 * `?tab=graph` deep-links the second tab (the ⌘K palette's target), so the
 * initial tab is read here — server-side — rather than from a client-side
 * `useSearchParams`, which would need its own Suspense boundary to keep
 * the page renderable.
 */
export default async function CorpusPage({
  searchParams,
}: {
  searchParams: Promise<{ [key: string]: string | string[] | undefined }>;
}) {
  const { tab } = await searchParams;
  return <CorpusTabs initialTab={tab === "graph" ? "graph" : "rag"} />;
}
