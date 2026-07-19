import type { Metadata } from "next";

import { CodebaseMapView } from "@/components/map/CodebaseMapView";

export const metadata: Metadata = {
  title: "Codebase Map · Varagity",
  description: "How Varagity fits together — entries, flows, engines, models, and stores.",
};

export default function MapPage() {
  return <CodebaseMapView />;
}
