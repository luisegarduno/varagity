"use client";

import { useQuery } from "@tanstack/react-query";
import { Maximize2Icon } from "lucide-react";
import { useState } from "react";

import { HighlightedText } from "@/components/provenance/ChunkCard";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Skeleton } from "@/components/ui/skeleton";
import { previewPageUrl } from "@/lib/api";
import type { EvidenceChunk } from "@/lib/evidence";
import { cssRects, previewFallbackLabel, type CssRect } from "@/lib/preview";
import { previewQuery } from "@/lib/queries";
import { cn } from "@/lib/utils";

/**
 * The rendered page with the chunk's highlight rects laid over it. The
 * wrapper hugs the image exactly, so the rects' percentage offsets track
 * it at any size — the inline rail and the enlarge dialog reuse this with
 * different image sizing. The fade-in is an `onLoad` state flip.
 */
function HighlightedPage({
  src,
  alt,
  rects,
  className,
  imgClassName,
}: {
  src: string;
  alt: string;
  rects: CssRect[];
  className?: string;
  imgClassName?: string;
}) {
  const [loaded, setLoaded] = useState(false);
  return (
    <div className={cn("relative", className)}>
      {/* eslint-disable-next-line @next/next/no-img-element -- the API
          renders and immutably caches the PNG; next/image would add an
          optimizer hop over an already-final image. */}
      <img
        src={src}
        alt={alt}
        onLoad={() => setLoaded(true)}
        className={cn(
          "w-full rounded-md border border-border motion-safe:transition-opacity motion-safe:duration-300",
          loaded ? "opacity-100" : "opacity-0",
          imgClassName,
        )}
      />
      {rects.map((rect, index) => (
        <div
          key={index}
          aria-hidden
          // The backdrop here is the rendered page, which is white paper in
          // either theme — so the blend mode must not track the app theme
          // (`screen` over white is always white).
          className="absolute rounded-[1px] bg-primary/25 mix-blend-multiply dark:bg-primary/30"
          style={rect}
        />
      ))}
    </div>
  );
}

/**
 * ★ The Kotaemon-style affordance (ADR-010): the one source page that
 * contains this chunk, with the chunk's text highlighted on it, plus an
 * enlarge dialog at reading size. Only mounted for eligible chunks (the
 * card gates on {@link previewEligible}), and only while the collapsible
 * panel is open — so mounting drives the locate fetch, and the cached
 * result makes every re-expand instant.
 *
 * Any degraded locate (`available:false`, transport error) falls back to
 * the exact full-text body ineligible chunks render, plus a muted line
 * naming the reason — never a dead panel.
 */
export function PagePreview({
  chunk,
  query,
}: {
  chunk: EvidenceChunk;
  query: string | null;
}) {
  // Eligibility guarantees docId; the assertion states that contract.
  const docId = chunk.docId!;
  const { data, isPending } = useQuery(
    previewQuery(docId, chunk.key, chunk.content),
  );

  if (isPending) {
    return (
      <Skeleton
        className={cn(
          "mt-1 w-full",
          chunk.fileType === "pptx" ? "aspect-video" : "aspect-[8.5/11]",
        )}
      />
    );
  }

  if (!data?.available || data.page == null) {
    return (
      <>
        <p className="mt-1 rounded-md bg-muted/50 p-2 text-xs leading-relaxed whitespace-pre-wrap">
          <HighlightedText text={chunk.content} query={query} />
        </p>
        <p className="mt-1 text-xs text-muted-foreground">
          {previewFallbackLabel(data?.reason ?? null)}
        </p>
      </>
    );
  }

  const rects = cssRects(data.rects);
  const src = previewPageUrl(docId, data.page);
  const name = chunk.fileName ?? chunk.source ?? "source document";
  const alt = `page ${data.page} of ${name}`;
  const caption =
    data.page_count != null
      ? `page ${data.page} of ${data.page_count}`
      : `page ${data.page}`;

  return (
    <figure className="mt-1 space-y-1.5">
      <HighlightedPage src={src} alt={alt} rects={rects} />
      <figcaption className="flex items-center justify-between gap-2">
        <span className="font-mono text-xs text-muted-foreground tabular-nums">
          {caption}
        </span>
        <Dialog>
          <DialogTrigger
            render={
              <Button
                variant="ghost"
                size="icon-xs"
                aria-label="Enlarge preview"
                title="Enlarge preview"
                className="-my-1 text-muted-foreground"
              />
            }
          >
            <Maximize2Icon />
          </DialogTrigger>
          <DialogContent className="w-fit max-w-[min(92vw,64rem)] sm:max-w-[min(92vw,64rem)]">
            <DialogTitle className="pr-8 text-base">
              {name} — {caption}
            </DialogTitle>
            <HighlightedPage
              src={src}
              alt={alt}
              rects={rects}
              className="mx-auto w-fit"
              imgClassName="max-h-[85vh] w-auto"
            />
          </DialogContent>
        </Dialog>
      </figcaption>
    </figure>
  );
}
