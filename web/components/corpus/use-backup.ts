"use client";

import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useCallback, useRef, useState } from "react";

import { useMountEffect } from "@/hooks/use-mount-effect";
import {
  ApiError,
  startExport,
  startRestore,
  streamBackupStatus,
  type BundleOut,
  type TargetCorpus,
} from "@/lib/api";
import {
  initialBackupView,
  reduceBackupEvent,
  type BackupView,
} from "@/lib/backup-reducer";
import { backupsQuery, queryKeys } from "@/lib/queries";

const UNREACHABLE = "API unreachable — is the stack up? (docker compose up -d)";

/** What `useBackupRun` hands a corpus tab (and, through it, `BackupPanel`). */
export interface BackupRunHandle {
  /** The folded backup-status stream. */
  view: BackupView;
  /** The corpus's bundles, newest first (`null` while loading). */
  bundles: BundleOut[] | null;
  /** Why the bundle list could not be read, if it couldn't. */
  bundlesError: string | null;
  /** Why the last start was refused (`409 corpus_busy`, `422`, …). */
  error: string | null;
  /** Whether a backup run is in flight. */
  running: boolean;
  /** Start an export; follows its stream on success. */
  startExport: () => Promise<void>;
  /**
   * Start restoring one bundle; follows its stream on success. Resolves to
   * `null` once the run started, or to why it was refused (the bundle list
   * is re-read either way: a run that fails on a damaged bundle is
   * remembered by the server, and its terminal frame re-reads the list, so
   * the row then says so).
   */
  startRestore: (name: string) => Promise<string | null>;
}

/**
 * One corpus's backup run (ADR-019): the bundle list plus the live status
 * stream, folded through the pure `reduceBackupEvent`.
 *
 * The `CorpusView.followIngest` pattern, lifted into a hook so both corpus
 * tabs can host the same panel: attach to the replaying stream once on
 * mount (a live, finished, or absent run all render correctly), re-attach
 * after a successful start, and invalidate the corpus's `["backups", …]`
 * list when a run ends — a finished export has a new bundle to show. A
 * finished restore also invalidates what it replaced, and the settings
 * catalog, whose stale flags the restore rewrote: for RAG the document
 * table; for the graph every `["graph", …]` key — its status and sizes, its
 * source listing, and the `/graph` view's cached slices and entities.
 *
 * `enabled: false` holds the bundle list back — the graph tab passes it
 * until the status poll says the graph is on, since a disabled graph's
 * backups answer `403 graph_disabled` (its panel isn't shown then). The
 * status stream is followed regardless: it is never refused.
 */
export function useBackupRun(
  corpus: TargetCorpus,
  { enabled = true }: { enabled?: boolean } = {},
): BackupRunHandle {
  const queryClient = useQueryClient();
  const { data: bundles = null, error: listError } = useQuery({
    ...backupsQuery(corpus),
    enabled,
  });
  const [view, setView] = useState<BackupView>(initialBackupView);
  const [error, setError] = useState<string | null>(null);
  const followingRef = useRef(false);

  const bundlesError =
    listError === null
      ? null
      : listError instanceof ApiError
        ? listError.message
        : UNREACHABLE;

  // Follow the current (or last) run; the generator ends at the terminal
  // frame, after which the bundle list reflects what the run wrote.
  const follow = useCallback(() => {
    if (followingRef.current) return;
    followingRef.current = true;
    void (async () => {
      try {
        let next = initialBackupView;
        setView(next);
        for await (const event of streamBackupStatus(corpus)) {
          next = reduceBackupEvent(next, event);
          setView(next);
        }
        if (next.run !== null && next.run.state !== "running") {
          void queryClient.invalidateQueries({ queryKey: queryKeys.backups(corpus) });
          if (next.run.operation === "restore") {
            void queryClient.invalidateQueries({
              queryKey: corpus === "graph" ? queryKeys.graph : queryKeys.documents,
            });
            void queryClient.invalidateQueries({ queryKey: queryKeys.settings });
          }
        }
      } catch {
        // Stream dropped (API restart, network): the panel keeps the last
        // known state; starting a run or reloading reconnects.
      } finally {
        followingRef.current = false;
      }
    })();
  }, [corpus, queryClient]);

  useMountEffect(() => {
    follow();
  });

  const start = useCallback(async () => {
    try {
      await startExport(corpus);
      setError(null);
      follow();
    } catch (failure) {
      setError(
        failure instanceof ApiError
          ? failure.message
          : "Could not start the export — is the stack up?",
      );
    }
  }, [corpus, follow]);

  const restore = useCallback(
    async (name: string): Promise<string | null> => {
      try {
        await startRestore(corpus, name);
        setError(null);
        follow();
        return null;
      } catch (failure) {
        const message =
          failure instanceof ApiError
            ? failure.message
            : "Could not start the restore — is the stack up?";
        setError(message);
        // The server may know more than the row shows (a damage a restore
        // run found, remembered): re-read the list so the row says so.
        void queryClient.invalidateQueries({ queryKey: queryKeys.backups(corpus) });
        return message;
      }
    },
    [corpus, follow, queryClient],
  );

  return {
    view,
    bundles,
    bundlesError,
    error,
    running: view.run?.state === "running",
    startExport: start,
    startRestore: restore,
  };
}
