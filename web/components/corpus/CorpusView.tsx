"use client";

import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useCallback, useRef, useState } from "react";

import { DocumentTable } from "@/components/corpus/DocumentTable";
import { IngestPanel } from "@/components/corpus/IngestPanel";
import { UploadDropzone } from "@/components/corpus/UploadDropzone";
import { useSettingsCatalog } from "@/components/settings/use-settings";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { useMountEffect } from "@/hooks/use-mount-effect";
import { ApiError, startIngest, streamIngestStatus } from "@/lib/api";
import {
  initialIngestView,
  reduceIngestEvent,
  type IngestView,
} from "@/lib/ingest-reducer";
import { configQuery, documentsQuery, queryKeys } from "@/lib/queries";
import { notifySettingsChanged } from "@/lib/settings-bus";

const UNREACHABLE = "API unreachable — is the stack up? (docker compose up -d)";

/**
 * The corpus management page (spec_v2 §4.2): upload dropzone, live ingest
 * progress (the status SSE replays, so reloading mid-run re-renders the
 * same picture), the ingested-document table, and the stale-corpus
 * "Re-ingest to apply" affordance. An empty corpus gets the guided
 * upload → ingest → ask flow instead of a bare table.
 */
export function CorpusView() {
  const queryClient = useQueryClient();
  const { data: config = null } = useQuery(configQuery());
  const { data: documents = null, error: documentsError } =
    useQuery(documentsQuery());
  // The stale flag rides the shared settings catalog, so the drawer's
  // banner and this one always agree and refresh together.
  const { catalog } = useSettingsCatalog();
  const [ingest, setIngest] = useState<IngestView>(initialIngestView);
  const [ingestError, setIngestError] = useState<string | null>(null);
  const followingRef = useRef(false);

  const corpusStale = catalog?.corpus_stale ?? false;
  const [hasUploaded, setHasUploaded] = useState(false);

  const documentsErrorMessage =
    documentsError === null
      ? null
      : documentsError instanceof ApiError
        ? documentsError.message
        : UNREACHABLE;
  const error = documentsErrorMessage ?? ingestError;

  const refreshDocuments = useCallback(() => {
    void queryClient.invalidateQueries({ queryKey: queryKeys.documents });
  }, [queryClient]);

  // Follow the current (or last) run; the generator ends at the terminal
  // frame, after which the document table reflects the run's writes and the
  // stale flag reflects the settings it ingested under.
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
          void queryClient.invalidateQueries({ queryKey: queryKeys.documents });
          void queryClient.invalidateQueries({ queryKey: queryKeys.settings });
        }
      } catch {
        // Stream dropped (API restart, network): the panel keeps the last
        // known state; starting a run or reloading reconnects.
      } finally {
        followingRef.current = false;
      }
    })();
  }, [queryClient]);

  // The status stream replays a run from its start, so attaching once on
  // mount renders the same picture whether a run is live, already finished,
  // or absent entirely.
  useMountEffect(() => {
    followIngest();
  });

  const handleStartIngest = useCallback(
    async (reingest: boolean) => {
      try {
        await startIngest(reingest);
        setIngestError(null);
        followIngest();
        if (reingest) notifySettingsChanged(); // the stale flag clears on completion
      } catch (failure) {
        setIngestError(
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
