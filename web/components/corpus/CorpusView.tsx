"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { DocumentTable } from "@/components/corpus/DocumentTable";
import { IngestPanel } from "@/components/corpus/IngestPanel";
import { UploadDropzone } from "@/components/corpus/UploadDropzone";
import {
  ApiError,
  getConfig,
  getSettings,
  listDocuments,
  startIngest,
  streamIngestStatus,
  type ConfigResponse,
  type DocumentOut,
} from "@/lib/api";
import {
  initialIngestView,
  reduceIngestEvent,
  type IngestView,
} from "@/lib/ingest-reducer";
import { notifySettingsChanged, onSettingsChanged } from "@/lib/settings-bus";

/**
 * The corpus management page (spec_v2 §4.2): upload dropzone, live ingest
 * progress (the status SSE replays, so reloading mid-run re-renders the
 * same picture), the ingested-document table, and the stale-corpus
 * "Re-ingest to apply" affordance.
 */
export function CorpusView() {
  const [config, setConfig] = useState<ConfigResponse | null>(null);
  const [documents, setDocuments] = useState<DocumentOut[] | null>(null);
  const [ingest, setIngest] = useState<IngestView>(initialIngestView);
  const [corpusStale, setCorpusStale] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const followingRef = useRef(false);

  const refreshDocuments = useCallback(() => {
    listDocuments().then(
      (list) => {
        setDocuments(list);
        setError(null);
      },
      (failure: unknown) => {
        setError(
          failure instanceof ApiError
            ? failure.message
            : "API unreachable — is the stack up? (docker compose up -d)",
        );
      },
    );
  }, []);

  const refreshStale = useCallback(() => {
    getSettings().then(
      (catalog) => setCorpusStale(catalog.corpus_stale),
      () => undefined, // the banner is best-effort; the table error covers outages
    );
  }, []);

  // Follow the current (or last) run; the generator ends at the terminal
  // frame, after which the document table reflects the run's writes.
  const followIngest = useCallback(() => {
    if (followingRef.current) return;
    followingRef.current = true;
    void (async () => {
      try {
        let view = initialIngestView;
        setIngest(view);
        for await (const event of streamIngestStatus()) {
          view = reduceIngestEvent(view, event);
          setIngest(view);
        }
        if (view.run !== null && view.run.state !== "running") {
          refreshDocuments();
          refreshStale();
        }
      } catch {
        // Stream dropped (API restart, network): the panel keeps the last
        // known state; starting a run or reloading reconnects.
      } finally {
        followingRef.current = false;
      }
    })();
  }, [refreshDocuments, refreshStale]);

  useEffect(() => {
    getConfig().then(setConfig, () => undefined);
    refreshDocuments();
    refreshStale();
    followIngest();
  }, [refreshDocuments, refreshStale, followIngest]);

  useEffect(() => onSettingsChanged(refreshStale), [refreshStale]);

  const handleStartIngest = useCallback(
    async (reingest: boolean) => {
      try {
        await startIngest(reingest);
        setError(null);
        followIngest();
        if (reingest) notifySettingsChanged(); // the stale flag clears on completion
      } catch (failure) {
        setError(
          failure instanceof ApiError
            ? failure.message
            : "Could not start the ingest — is the stack up?",
        );
      }
    },
    [followIngest],
  );

  const isRunning = ingest.run?.state === "running";

  return (
    <div className="flex h-full min-h-0 flex-col overflow-y-auto">
      <header className="border-b border-border p-4">
        <h1 className="text-lg font-semibold tracking-tight">Corpus</h1>
        <p className="text-sm text-muted-foreground">
          Upload documents, ingest them into both stores, and manage what the
          assistant can ground on.
        </p>
      </header>

      <div className="mx-auto flex w-full max-w-4xl flex-col gap-6 p-4">
        {error && (
          <p
            role="alert"
            className="rounded-md border border-destructive/40 bg-destructive/10 p-3 text-sm"
          >
            {error}
          </p>
        )}

        {corpusStale && (
          <div
            role="status"
            className="flex items-center justify-between gap-3 rounded-md border border-amber-500/40 bg-amber-500/10 p-3 text-sm"
          >
            <span>
              Ingest-time settings changed since the last ingest — the corpus is
              <strong> stale</strong>. Re-ingest to apply them.
            </span>
            <button
              type="button"
              className="shrink-0 rounded-md border border-border bg-background px-3 py-1.5 text-sm font-medium hover:bg-accent disabled:opacity-50"
              disabled={isRunning}
              onClick={() => void handleStartIngest(true)}
            >
              Re-ingest to apply
            </button>
          </div>
        )}

        <UploadDropzone config={config} onUploaded={refreshDocuments} />

        <IngestPanel
          view={ingest}
          disabled={isRunning}
          onIngest={() => void handleStartIngest(false)}
          onReingest={() => void handleStartIngest(true)}
        />

        <DocumentTable documents={documents} onChanged={refreshDocuments} />
      </div>
    </div>
  );
}
