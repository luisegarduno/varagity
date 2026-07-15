"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { DocumentTable } from "@/components/corpus/DocumentTable";
import { IngestPanel } from "@/components/corpus/IngestPanel";
import { UploadDropzone } from "@/components/corpus/UploadDropzone";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
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
 * "Re-ingest to apply" affordance. An empty corpus gets the guided
 * upload → ingest → ask flow instead of a bare table.
 */
export function CorpusView() {
  const [config, setConfig] = useState<ConfigResponse | null>(null);
  const [documents, setDocuments] = useState<DocumentOut[] | null>(null);
  const [ingest, setIngest] = useState<IngestView>(initialIngestView);
  const [corpusStale, setCorpusStale] = useState(false);
  const [hasUploaded, setHasUploaded] = useState(false);
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

  const handleUploaded = useCallback(() => {
    setHasUploaded(true);
    refreshDocuments();
  }, [refreshDocuments]);

  const isRunning = ingest.run?.state === "running";
  // The guided first-run flow: corpus confirmed empty and nothing has run
  // (a live or terminal run renders the normal layout with its panel).
  const guided =
    documents !== null && documents.length === 0 && ingest.run === null;

  return (
    <div className="flex h-full min-h-0 flex-col overflow-y-auto">
      <header className="border-b border-border px-4 py-5 sm:px-6">
        <h1 className="font-heading text-2xl font-normal">Corpus</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Upload documents, ingest them into both stores, and manage what the
          assistant can ground on.
        </p>
      </header>

      <div className="mx-auto flex w-full max-w-4xl flex-col gap-6 p-4 sm:p-6">
        {error && (
          <p
            role="alert"
            className="rounded-lg border border-destructive/30 bg-destructive/5 px-4 py-3 text-sm text-destructive"
          >
            {error}
          </p>
        )}

        {corpusStale && (
          <div
            role="status"
            className="flex flex-wrap items-center justify-between gap-3 rounded-lg border border-amber-500/30 bg-amber-500/10 px-4 py-3 text-sm"
          >
            <span className="flex min-w-0 items-center gap-2.5">
              <Badge variant="warning">stale</Badge>
              <span>
                Ingest-time settings changed since the last ingest — re-ingest
                to apply them.
              </span>
            </span>
            <Button
              variant="outline"
              size="sm"
              disabled={isRunning}
              onClick={() => void handleStartIngest(true)}
            >
              Re-ingest to apply
            </Button>
          </div>
        )}

        {guided ? (
          <section
            aria-label="Getting started"
            className="flex flex-col gap-6 rounded-xl border border-border bg-card p-6 sm:p-8"
          >
            <div>
              <h2 className="font-heading text-xl font-normal">
                Build your corpus
              </h2>
              <p className="mt-1 text-sm text-muted-foreground">
                Nothing is ingested yet — ground the assistant in your own
                documents in three steps.
              </p>
            </div>

            <ol className="flex flex-col gap-2 text-sm text-muted-foreground sm:flex-row sm:items-center sm:gap-6">
              {["Upload documents", "Run an ingest", "Ask questions"].map(
                (step, index) => (
                  <li key={step} className="flex items-center gap-2">
                    <span className="flex size-5 shrink-0 items-center justify-center rounded-full border border-border font-mono text-[10px]">
                      {index + 1}
                    </span>
                    {step}
                  </li>
                ),
              )}
            </ol>

            <UploadDropzone config={config} onUploaded={handleUploaded} />

            {hasUploaded ? (
              <IngestPanel
                view={ingest}
                disabled={isRunning}
                onIngest={() => void handleStartIngest(false)}
                onReingest={() => void handleStartIngest(true)}
              />
            ) : (
              <p className="text-xs text-muted-foreground">
                Files already in the corpus directory?{" "}
                <button
                  type="button"
                  className="underline underline-offset-2 transition-colors hover:text-foreground"
                  onClick={() => void handleStartIngest(false)}
                >
                  Run an ingest
                </button>
                .
              </p>
            )}
          </section>
        ) : (
          <>
            <UploadDropzone config={config} onUploaded={handleUploaded} />

            <IngestPanel
              view={ingest}
              disabled={isRunning}
              onIngest={() => void handleStartIngest(false)}
              onReingest={() => void handleStartIngest(true)}
            />

            <DocumentTable documents={documents} onChanged={refreshDocuments} />
          </>
        )}
      </div>
    </div>
  );
}
