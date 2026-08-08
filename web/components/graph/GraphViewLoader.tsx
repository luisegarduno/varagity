"use client";

import dynamic from "next/dynamic";

import { Skeleton } from "@/components/ui/skeleton";

/**
 * The repo's **first** `next/dynamic` with `ssr: false`, and deliberately so.
 *
 * sigma renders in WebGL: importing it evaluates modules that reach for
 * `window`, so the graph view cannot be prerendered. `/map`'s
 * ref-callback pattern defers *work* to the browser but cannot defer a
 * library's module evaluation, which is the thing that has to wait here.
 *
 * `ssr: false` is only legal inside a Client Component (a Server Component
 * that used it would error), which is what this thin wrapper exists to be —
 * the page itself stays a Server Component and keeps its metadata.
 */
const GraphView = dynamic(
  () => import("@/components/graph/GraphView").then((module) => module.GraphView),
  {
    ssr: false,
    loading: () => (
      <div className="flex h-full items-center justify-center p-6">
        <Skeleton className="size-full max-h-[36rem] max-w-4xl rounded-xl" />
      </div>
    ),
  },
);

/** Client boundary for the WebGL graph view. */
export function GraphViewLoader() {
  return <GraphView />;
}
