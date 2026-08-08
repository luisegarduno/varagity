"use client";

import { FilesIcon, NetworkIcon } from "lucide-react";

import { CorpusView } from "@/components/corpus/CorpusView";
import { GraphCorpusView } from "@/components/corpus/GraphCorpusView";
import {
  Tabs,
  TabsIndicator,
  TabsList,
  TabsPanel,
  TabsTab,
} from "@/components/ui/tabs";

/** The two corpora, in tab order. */
const RAG = "rag";
const GRAPH = "graph";

/**
 * The corpus page's shell: one header, two corpora (spec_graphrag §4.4).
 *
 * **Documents** is the chunk-RAG corpus, unchanged — the same dropzone,
 * ingest panel, and document table it always was. **Message graph** is the
 * Graph RAG corpus: message archives in, a resumable extraction build,
 * and a graph out.
 *
 * The panels are the default Base UI ones, i.e. unmounted while hidden, so
 * sitting on Documents costs no graph polling and vice versa.
 */
export function CorpusTabs({ initialTab }: { initialTab: string }) {
  return (
    <div className="flex h-full min-h-0 flex-col overflow-y-auto">
      <Tabs defaultValue={initialTab === GRAPH ? GRAPH : RAG}>
        <header className="border-b border-border px-4 py-5 sm:px-6">
          <h1 className="font-heading text-2xl font-normal">Corpus</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            Manage what the assistant can ground on — the documents it
            retrieves chunks from, and the message archive it extracts a
            graph from.
          </p>
          <TabsList aria-label="Corpus" className="mt-4">
            <TabsIndicator />
            <TabsTab value={RAG}>
              <FilesIcon aria-hidden />
              Documents
            </TabsTab>
            <TabsTab value={GRAPH}>
              <NetworkIcon aria-hidden />
              Message graph
            </TabsTab>
          </TabsList>
        </header>

        <TabsPanel value={RAG}>
          <CorpusView />
        </TabsPanel>
        <TabsPanel value={GRAPH}>
          <GraphCorpusView />
        </TabsPanel>
      </Tabs>
    </div>
  );
}
