import type { Metadata } from "next";

import { GraphViewLoader } from "@/components/graph/GraphViewLoader";

export const metadata: Metadata = {
  title: "Message graph · Varagity",
  description:
    "The entities and relations extracted from the message archive, drawn in full.",
};

/**
 * `/graph` — the extracted message graph (spec_graphrag §4.4).
 *
 * A Server Component so the route keeps its metadata; the view itself is
 * WebGL and loads client-only through {@link GraphViewLoader}.
 */
export default function GraphPage() {
  return <GraphViewLoader />;
}
